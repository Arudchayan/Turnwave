"""Streaming client for the pinned Liquid audio runner.

One lock owns the server context for the entire lifetime of each request.
Closing a stream cancels local consumption; backend abort latency varies by build.
"""
import base64
import io
import json
import os
import struct
import threading
import time
import urllib.request
import urllib.error

import numpy as np

TTS_VOICE = "Perform TTS. Use the UK female voice."
CHAT_SYS_BASE = "Respond with interleaved text and audio."
CHAT_SYS = os.getenv('LOCALVOICE_CHAT_SYSTEM', CHAT_SYS_BASE +
    " Answer briefly and directly. Follow the user's instructions exactly."
    " When the user tells you their name, remember it as the user's name.")
ASR_SYS = "Perform ASR."


def wav16k_bytes(samples_float32):
    pcm = (np.clip(np.asarray(samples_float32, dtype=np.float32), -1, 1) * 32767).astype('<i2')
    buf = io.BytesIO()
    buf.write(b'RIFF' + struct.pack('<I', 36 + pcm.nbytes) + b'WAVEfmt ')
    buf.write(struct.pack('<IHHIIHH', 16, 1, 1, 16000, 32000, 2, 16))
    buf.write(b'data' + struct.pack('<I', pcm.nbytes) + pcm.tobytes())
    return buf.getvalue()


def extract_audio(delta):
    """Decode either runner schema into finite float32 mono audio at 24 kHz."""
    legacy = delta.get('audio_chunk')
    modern = delta.get('audio')
    if legacy and legacy.get('data'):
        pcm = np.frombuffer(base64.b64decode(legacy['data'], validate=True), dtype='<f4').copy()
    elif modern and modern.get('data'):
        pcm = np.frombuffer(base64.b64decode(modern['data'], validate=True), dtype='<i2').astype(np.float32) / 32768
        rate = int(modern.get('sample_rate', 24000))
        if not 8000 <= rate <= 192000:
            raise ValueError('Invalid audio sample rate')
        if rate != 24000 and len(pcm):
            positions = np.arange(round(len(pcm) * 24000 / rate)) * rate / 24000
            pcm = np.interp(positions, np.arange(len(pcm)), pcm).astype(np.float32)
    else:
        return None
    if not np.isfinite(pcm).all():
        raise ValueError('Non-finite audio samples')
    return np.clip(pcm, -1, 1)


def sse_events(response, cancel=None):
    """Read available bytes, never wait for a 64 KiB batch of SSE events.

    read1 returns after one underlying read, including on chunked responses.
    Accept LF/CRLF and split UTF-8 sequences only after a whole event arrives.
    """
    buf = b''
    read = getattr(response, 'read1', None) or response.read
    while cancel is None or not cancel.is_set():
        chunk = read(16384)
        if not chunk:
            if buf.strip():
                raise ValueError('Truncated event stream')
            if cancel is None or not cancel.is_set():
                raise ConnectionError('Stream ended before completion')
            return
        buf += chunk
        # A trailing CR can belong to a CRLF split between reads.
        buf = buf.replace(b'\r\n', b'\n')
        if len(buf) > 4 * 1024 * 1024:
            raise ValueError('Oversized event stream frame')
        while b'\n\n' in buf:
            frame, buf = buf.split(b'\n\n', 1)
            data = b'\n'.join(line[5:].lstrip(b' ') for line in frame.split(b'\n') if line.startswith(b'data:'))
            if not data:
                continue
            if data == b'[DONE]':
                return
            event = json.loads(data)
            error = event.get('error')
            if error is not None:
                raise RuntimeError(str(error) if error else 'The server reported an error.')
            yield event


class Turn:
    def __init__(self):
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.t_req = 0.0
        self.ttft = None  # first decoded audio, not first text
        self.first_event = None
        self.queue_s = 0.0
        self.headers_s = None
        self.text_parts = []
        self.audio_frames = 0
        self.audio_samples = 0
        self.error = None
        self.finish_reason = None
        self.request_observer = None
        self._response = None
        self._lock = threading.Lock()

    def abort(self):
        self.cancel.set()
        # Do not close a buffered HTTPResponse on the UI/mic thread: close()
        # may wait for the reader's lock. Socket shutdown wakes read1 instead.
        with self._lock:
            resp = self._response
        if resp is not None:
            try:
                import socket
                sock = resp.fp.raw._sock
                sock.shutdown(socket.SHUT_RDWR)
                # urllib marks its socket closed while a makefile still owns
                # the handle. On Windows shutdown alone does not wake recv.
                # Close that handle now; later BufferedReader.close is safe.
                sock._real_close()
            except (AttributeError, OSError):
                pass

    @property
    def truncated(self):
        return self.finish_reason == 'length'

    @property
    def text(self):
        return ''.join(self.text_parts)

    @property
    def degenerate(self):
        text = self.text
        if len(text) < 60:
            return False
        tail = text[-240:]
        if len(set(tail)) <= 3:
            return True
        for size in range(1, min(80, len(tail) // 3) + 1):
            repeats = max(3, (60 + size - 1) // size)
            if tail.endswith(tail[-size:] * repeats):
                return True
        return False


class ServerClient:
    def __init__(self, base='http://localhost:8080', timeout=30):
        # Windows may try an unreachable ::1 first, adding ~2 seconds per
        # request to WSL localhost forwarding. Use the measured IPv4 path.
        self.base = base.rstrip('/').replace('http://localhost:', 'http://127.0.0.1:')
        self.timeout = timeout
        self.request_lock = threading.Lock()

    def _post(self, payload):
        req = urllib.request.Request(self.base + '/v1/chat/completions',
                                     data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        return urllib.request.urlopen(req, timeout=self.timeout)

    def health(self):
        try:
            with urllib.request.urlopen(self.base + '/health', timeout=1) as response:
                return response.status == 200
        except urllib.error.HTTPError as exc:
            # The pinned runner has no GET /health. Its default 404 carries
            # an identifying header, which proves the intended HTTP server
            # is reachable without touching its conversation context.
            with exc:
                return exc.code == 404 and exc.headers.get('Server') == 'lfm2-audio-server'
        except (OSError, ValueError):
            return False

    def _stream(self, payload, turn, on_audio=None):
        acquired = False
        resp = None
        queued_at = time.perf_counter()
        try:
            while not turn.cancel.is_set():
                if self.request_lock.acquire(timeout=0.05):
                    acquired = True
                    break
            if not acquired or turn.cancel.is_set():
                return
            if turn.request_observer is not None:
                turn.request_observer(payload)
            turn.t_req = time.perf_counter()
            turn.queue_s = turn.t_req - queued_at
            resp = self._post(payload)
            turn.headers_s = time.perf_counter() - turn.t_req
            with turn._lock:
                turn._response = resp
            if turn.cancel.is_set():
                return
            for event in sse_events(resp, turn.cancel):
                if turn.cancel.is_set():
                    break
                elapsed = time.perf_counter() - turn.t_req
                if turn.first_event is None:
                    turn.first_event = elapsed
                choices = event.get('choices') or []
                if not choices:
                    continue
                if choices[0].get('finish_reason') is not None:
                    turn.finish_reason = choices[0]['finish_reason']
                delta = choices[0].get('delta') or {}
                if delta.get('content'):
                    turn.text_parts.append(delta['content'])
                    if turn.degenerate:
                        raise RuntimeError('The reply became repetitive. Please try again.')
                audio = extract_audio(delta)
                if audio is not None and len(audio):
                    if turn.ttft is None:
                        turn.ttft = elapsed
                    turn.audio_frames += 1
                    turn.audio_samples += len(audio)
                    if on_audio is not None:
                        on_audio(audio)
        except Exception as exc:
            if not turn.cancel.is_set():
                turn.error = str(exc)
        finally:
            with turn._lock:
                turn._response = None
            try:
                if resp is not None:
                    resp.close()
            except Exception:
                # Cleanup must not replace completed audio, a primary stream
                # error, or intentional cancellation with a secondary failure.
                pass
            finally:
                if acquired:
                    self.request_lock.release()
                turn.done.set()

    def asr_text(self, wav_bytes, max_tokens=128):
        # Use on a separate server instance: this resets its context.
        turn = Turn()
        self._stream(self._payload(ASR_SYS, self._audio_input(wav_bytes), True, max_tokens, audio=False), turn)
        if turn.error:
            raise RuntimeError(turn.error)
        return turn.text.strip()

    @staticmethod
    def _audio_input(wav_bytes):
        return [{'type': 'input_audio', 'input_audio': {
            'data': base64.b64encode(wav_bytes).decode(), 'format': 'wav'}}]

    @staticmethod
    def _payload(system, content, first, max_tokens, audio=True):
        messages = ([{'role': 'system', 'content': system}] if first else [])
        messages.append({'role': 'user', 'content': content})
        body = {'model': '', 'messages': messages, 'stream': True,
                'max_tokens': max_tokens, 'reset_context': bool(first)}
        # TTS must be audio-only: interleaved mode invents text instead
        # of reading the supplied words. Chat needs both modalities.
        body['modalities'] = (['audio'] if system.startswith('Perform TTS.')
                              else ['text', 'audio'] if audio else ['text'])
        return body

    def tts(self, text, turn, max_tokens=128, on_audio=None):
        self._stream(self._payload(TTS_VOICE, text, True, max_tokens), turn, on_audio)

    def chat_turn(self, wav_bytes, turn, first, max_tokens=512,
                  on_audio=None, require_audio=True, _attempt=1):
        # Never silently replay input into a stateful context. Missing audio
        # is surfaced to the app; the next turn can recover with a reset.
        self._stream(self._payload(CHAT_SYS, self._audio_input(wav_bytes), first, max_tokens), turn, on_audio)
        if require_audio and not turn.cancel.is_set() and not turn.error and not turn.audio_samples:
            turn.error = ('The reply reached its length limit before producing audio.' if turn.truncated
                          else 'The server returned no audio. Please try again.')
