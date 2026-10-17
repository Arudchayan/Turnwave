"""Mic capture + interruptible speaker playback (sounddevice + numpy).

Mic: 16 kHz mono float32 callback -> bounded queue (drops oldest on overflow,
so latency can't accumulate). Playback: 24 kHz mono, generation-counter
cancellation for instant barge-in stops.
"""
import queue
import sys
from contextlib import contextmanager
import threading
import time

import numpy as np
import sounddevice as sd

MIC_RATE = 16000
PLAY_RATE = 24000


@contextmanager
def audio_thread():
    """WASAPI capture needs COM on the thread opening its stream.

    PortAudio initialization on Python's main thread does not initialize
    worker threads. Keep the apartment alive until that worker closes audio.
    """
    initialized = False
    if sys.platform == 'win32':
        import ctypes
        result = ctypes.windll.ole32.CoInitializeEx(None, 0) & 0xffffffff
        initialized = result in (0, 1)
        if not initialized and result != 0x80010106:
            raise OSError('Unable to initialize audio thread COM: %08x' % result)
    try:
        yield
    finally:
        if initialized:
            ctypes.windll.ole32.CoUninitialize()


def list_inputs():
    """[(index, name)] for devices with input channels."""
    out = []
    try:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                out.append((i, d["name"][:48]))
    except Exception:
        pass
    return out


def preferred_device(kind):
    """Use WASAPI for the SAME system-default endpoint, when available.

    Do not silently redirect speakers/headphones to a different device.
    Explicit user-selected indices are always respected.
    """
    try:
        default_index = int(sd.default.device[0 if kind == 'input' else 1])
        original = sd.query_devices(default_index)
        if sd.query_hostapis(original['hostapi'])['name'] == 'Windows WASAPI':
            return default_index
        prefix = original['name'].rstrip().casefold()
        channel_key = 'max_input_channels' if kind == 'input' else 'max_output_channels'
        for index, candidate in enumerate(sd.query_devices()):
            if (candidate[channel_key] > 0
                    and sd.query_hostapis(candidate['hostapi'])['name'] == 'Windows WASAPI'
                    and candidate['name'].casefold().startswith(prefix)):
                return index
        return default_index
    except Exception:
        return None


class Mic:
    """16 kHz mono capture from the system default input.

    Intel SST arrays expose dead channels on ch0, so we capture ALL native
    channels at the device rate, track per-channel energy (slow EMA), read
    the loudest channel, and resample to 16 kHz mono (linear interp).
    Bounded queue drops oldest on overflow so latency can't accumulate.
    """

    def __init__(self, rate=MIC_RATE, device=None, max_buffer_s=0.5, gain=None):
        self.rate = rate
        self.device = device if device is not None else preferred_device('input')
        self.gain = float(gain) if gain is not None else 1.0
        dev = self.device
        try:
            if dev is None:
                dev = sd.default.device[0]
            info = sd.query_devices(dev)
            self.native_rate = int(info["default_samplerate"])
            self.native_ch = min(int(info["max_input_channels"]), 4)
            if gain is None and sd.query_hostapis(info["hostapi"])["name"] != "Windows WASAPI":
                self.gain = 6.0
        except Exception:
            self.native_rate = 44100
            self.native_ch = 1
        self.blocksize = max(160, round(self.native_rate * .02))
        self.q = queue.Queue(maxsize=max(2, round(max_buffer_s / .02)))
        self.stream = None
        self.level = 0.0
        self._ch_energy = None
        self.best_ch = 0
        self.dropped_blocks = 0
        self.capture_errors = 0
        self.clipped_fraction = 0.0
        self.latency_s = None
        self.effective_gain = self.gain

    def _cb(self, indata, frames, time_info, status):
        if status:
            self.capture_errors += 1
        data = np.asarray(indata, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        ch = data.shape[1]
        e = (data ** 2).mean(axis=0) + 1e-12
        if self._ch_energy is None or len(self._ch_energy) != ch:
            self._ch_energy = e
        else:
            self._ch_energy = 0.98 * self._ch_energy + 0.02 * e
        self.best_ch = int(np.argmax(self._ch_energy))
        mono_native = data[:, self.best_ch].copy()
        # resample native -> 16k
        n_out = int(len(mono_native) * self.rate / self.native_rate)
        if n_out < 8:
            return
        idx = np.linspace(0, len(mono_native) - 1, n_out)
        mono = np.interp(idx, np.arange(len(mono_native)), mono_native).astype(np.float32)
        peak = float(np.max(np.abs(mono)))
        self.effective_gain = min(self.gain, .95 / max(peak, 1e-6))
        amplified = mono * self.effective_gain
        self.clipped_fraction = float(np.mean(np.abs(mono) >= .999))
        mono = np.clip(amplified, -1.0, 1.0)
        self.level = float(np.sqrt(np.mean(mono ** 2) + 1e-12))
        try:
            self.q.put_nowait(mono)
        except queue.Full:
            self.dropped_blocks += 1
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            self.q.put_nowait(mono)

    def start(self):
        if self.stream is not None:
            if self.stream.active:
                return
            previous, self.stream = self.stream, None
            previous.close()
        self.stream = sd.InputStream(samplerate=self.native_rate,
                                     channels=self.native_ch,
                                     dtype="float32", device=self.device,
                                     blocksize=self.blocksize, latency='low', callback=self._cb)
        try:
            self.stream.start()
            self.latency_s = float(self.stream.latency)
        except Exception as failure:
            failed, self.stream = self.stream, None
            try:
                failed.close()
            except Exception as cleanup:
                failure.add_note('Microphone cleanup also failed: ' + str(cleanup))
            raise

    def stop(self):
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.stop()
            except Exception as failure:
                try:
                    stream.close()
                except Exception as cleanup:
                    failure.add_note('Microphone cleanup also failed: ' + str(cleanup))
                raise
            else:
                stream.close()

    def drain(self):
        chunks = []
        while True:
            try:
                chunks.append(self.q.get_nowait())
            except queue.Empty:
                break
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


class Player:
    """Low-latency playback with exact sample counts and immediate abort.

    A short PortAudio buffer bounds local interruption latency. Network chunk
    boundaries never introduce padding. on_play supplies the echo reference
    before each write; on_written reports source samples after a successful
    device write. Neither callback proves that buffered samples became audible.
    """
    def __init__(self, rate=PLAY_RATE, device=None):
        self.rate = rate
        self.device = device if device is not None else preferred_device('output')
        self.native_rate = rate
        self.native_channels = 1
        try:
            info = sd.query_devices(self.device, 'output')
            self.native_rate = int(info['default_samplerate'])
            self.native_channels = min(2, int(info['max_output_channels']))
        except Exception:
            pass
        self.stats = {}
        self._gen = 0
        self._lock = threading.Lock()
        self._stream = None
        self.playing = False

    def stop(self):
        with self._lock:
            self._gen += 1
            stream = self._stream
            self.playing = False
        if stream is not None:
            try:
                stream.abort()
            except sd.PortAudioError:
                pass

    def play_chunks(self, chunk_iter, cancel=None, on_play=None, on_start=None, on_written=None):
        if cancel is not None and cancel.is_set():
            return
        with self._lock:
            self._gen += 1
            generation = self._gen
        stream = None
        started = False
        self.stats = dict(underflows=0, source_samples=0, device_samples=0, max_gap_ms=0.0, latency_s=None)
        previous_write = None
        previous_duration = 0.0
        try:
            for chunk in chunk_iter:
                audio = np.asarray(chunk, dtype=np.float32).ravel()
                for offset in range(0, len(audio), 480):
                    with self._lock:
                        obsolete = generation != self._gen
                    if obsolete or (cancel is not None and cancel.is_set()):
                        return
                    if stream is None:
                        stream = sd.OutputStream(samplerate=self.native_rate, channels=self.native_channels,
                                                 dtype='float32', device=self.device,
                                                 blocksize=round(self.native_rate * .02), latency='low')
                        with self._lock:
                            if generation != self._gen:
                                return
                            self._stream = stream
                            self.playing = True
                        stream.start()
                        self.stats['latency_s'] = float(getattr(stream, 'latency', 0))
                    piece = audio[offset:offset + 480]
                    if not started:
                        started = True
                        if on_start is not None:
                            on_start()
                    if on_play is not None:
                        on_play(piece)
                    try:
                        device_audio = piece
                        if self.native_rate != self.rate:
                            positions = np.arange(round(len(piece) * self.native_rate / self.rate)) * self.rate / self.native_rate
                            device_audio = np.interp(positions, np.arange(len(piece)), piece).astype(np.float32)
                        if self.native_channels > 1:
                            device_audio = np.repeat(device_audio[:, None], self.native_channels, axis=1)
                        now = time.perf_counter()
                        if previous_write is not None:
                            gap = max(0, now - previous_write - previous_duration)
                            self.stats['max_gap_ms'] = max(self.stats['max_gap_ms'], round(gap * 1000, 2))
                        underflow = stream.write(device_audio)
                        if underflow and previous_write is not None:
                            self.stats['underflows'] += 1
                        previous_write = now
                        previous_duration = len(piece) / self.rate
                        self.stats['source_samples'] += len(piece)
                        self.stats['device_samples'] += len(device_audio)
                        if on_written is not None:
                            on_written(piece)
                    except sd.PortAudioError:
                        with self._lock:
                            interrupted = generation != self._gen
                        if interrupted or (cancel is not None and cancel.is_set()):
                            return
                        raise
            if stream is not None:
                with self._lock:
                    uninterrupted = generation == self._gen
                if uninterrupted:
                    stream.stop()
        finally:
            try:
                if stream is not None:
                    stream.close()
            finally:
                # A device-close error must remain visible to the caller,
                # but cannot leave this completed playback marked active.
                with self._lock:
                    if self._stream is stream:
                        self._stream = None
                    if generation == self._gen:
                        self.playing = False
