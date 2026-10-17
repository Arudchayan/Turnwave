"""Thread-safe echo cancellation and speech endpointing.

The reference is actual device-bound playback, resampled to 16 kHz. All timing
inside the detector uses audio time so offline and real-time tests agree.
"""
import threading
import numpy as np

try:
    from pywebrtc_audio import AudioProcessor
except ImportError:
    AudioProcessor = None

F_AEC = 160
F_VAD = 512


class DuplexFrontend:
    def __init__(self, stream_delay_ms=40, vad_thresh=0.5, start_chunks=3,
                 end_silence_ms=480, aec=True, holdoff_s=0.35, residual_floor=.005,
                 onset_window_chunks=None):
        self.vad_thresh = vad_thresh
        self.start_chunks = int(start_chunks)
        if self.start_chunks < 1:
            raise ValueError('Speech confirmation needs at least one frame')
        self.onset_window_chunks = self.start_chunks + 2 if onset_window_chunks is None else int(onset_window_chunks)
        if self.onset_window_chunks < self.start_chunks:
            raise ValueError('Confirmation window is smaller than the required evidence')
        self._onset_mask = (1 << self.onset_window_chunks) - 1
        self._onset_bits = 0
        self.end_chunks = max(1, round(end_silence_ms / 32))
        self.holdoff_s = holdoff_s
        self.residual_floor = max(0.0, float(residual_floor))
        self._lock = threading.RLock()
        self._sample_clock = 0
        self._play_t0 = -1e9
        self._far_active = False
        self._since_far = 16000
        self.aec_error = None
        self.aec = None
        if aec:
            if AudioProcessor is None:
                self.aec_error = 'Echo cancellation unavailable; use headphones.'
            else:
                try:
                    self.aec = AudioProcessor(sample_rate=16000, num_channels=1,
                                              echo_cancellation=True, noise_suppression=False)
                    self.aec.stream_delay_ms = stream_delay_ms
                except Exception as exc:
                    self.aec_error = str(exc)
        self._mic_buf = np.zeros(0, np.float32)
        self._far_buf = np.zeros(0, np.float32)
        self._far_history = np.zeros(0, np.float32)
        self._near_history = np.zeros(0, np.float32)
        self._clean_history = np.zeros(0, np.float32)
        self.echo_score = 0.0
        self._vad_buf = np.zeros(0, np.float32)
        self._raw_vad_buf = np.zeros(0, np.float32)
        self._far_vad_buf = np.zeros(0, np.float32)
        self.residual_rejections = 0
        self._model = None
        self.speaking = False
        self._hot = self._quiet = 0
        self.probability = 0.0
        self.level = 0.0

    def warmup(self):
        with self._lock:
            self._vad_model()
            self._prob(np.zeros(512, np.float32))
            self.reset()

    def _vad_model(self):
        if self._model is None:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad(onnx=True)
        return self._model

    def push_far(self, samples, rate=16000):
        """Append actual playback samples; caller specifies their sample rate."""
        audio = np.asarray(samples, dtype=np.float32).ravel()
        if not len(audio):
            return
        if rate != 16000:
            positions = np.arange(round(len(audio) * 16000 / rate)) * rate / 16000
            audio = np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)
        with self._lock:
            active = float(np.sqrt(np.mean(audio ** 2))) >= 0.005
            if active and not self._far_active:
                self._play_t0 = self._sample_clock / 16000
            self._far_active = self._far_active or active
            self._since_far = 0
            self._far_buf = np.concatenate((self._far_buf, audio))[-8000:]

    def _echo_similarity(self):
        # Before AEC has converged, the speaker can still sound like speech.
        # Match raw mic against recent render history across unknown device /
        # room delays. Averaging to 4 kHz makes this inexpensive and robust to
        # sub-frame scheduling. Double talk lowers the normalized correlation.
        if len(self._near_history) < 1600 or len(self._far_history) < 1600:
            return 0.0
        near = self._near_history[-1600:].reshape(-1, 4).mean(axis=1)
        nfar = len(self._far_history) // 4 * 4
        far = self._far_history[-nfar:].reshape(-1, 4).mean(axis=1)
        near = near - near.mean()
        near_energy = float(np.dot(near, near))
        if near_energy < 1e-8:
            return 0.0
        n = len(near)
        sums = np.concatenate(([0.0], np.cumsum(far, dtype=np.float64)))
        squares = np.concatenate(([0.0], np.cumsum(far * far, dtype=np.float64)))
        energy = squares[n:] - squares[:-n] - (sums[n:] - sums[:-n]) ** 2 / n
        dots = np.correlate(far, near, mode='valid')
        scores = np.abs(dots) / np.sqrt(np.maximum(energy * near_energy, 1e-12))
        return min(1.0, float(np.max(scores)))

    def set_delay(self, milliseconds):
        with self._lock:
            if self.aec is not None:
                self.aec.stream_delay_ms = max(0, min(500, int(milliseconds)))

    def clear_playback(self):
        with self._lock:
            # Drop queued, unplayed reference, but keep AEC running with zero
            # render for the delayed acoustic tail. Bypassing immediately on
            # interrupt leaks the old reply into the user's replacement clip.
            self._far_buf = np.zeros(0, np.float32)
            if self._far_active:
                self._since_far = 0

    def reset(self):
        with self._lock:
            # Discard partial speech decisions without creating a gap in the
            # acoustic timeline. AEC keeps its state and sub-10 ms mic frame.
            self._advance_history(self._raw_vad_buf, self._far_vad_buf, self._vad_buf)
            self.speaking = False
            self._hot = self._quiet = 0
            self._onset_bits = 0
            self._vad_buf = np.zeros(0, np.float32)
            self._raw_vad_buf = np.zeros(0, np.float32)
            self._far_vad_buf = np.zeros(0, np.float32)
            if self._model is not None:
                self._model.reset_states()

    def _advance_history(self, raw, far, clean):
        self._near_history = np.concatenate((self._near_history, raw))[-1600:]
        self._far_history = np.concatenate((self._far_history, far))[-12000:]
        self._clean_history = np.concatenate((self._clean_history, clean))[-1600:]
        self._sample_clock += len(raw)

    def _prob(self, chunk):
        import torch
        with torch.no_grad():
            probability = self._vad_model()(torch.from_numpy(chunk[None, :]), 16000)
        return float(probability.detach().cpu().numpy().ravel()[0])

    def update(self, samples):
        with self._lock:
            return self._update(samples)

    def _update(self, samples):
        incoming = np.asarray(samples, dtype=np.float32).ravel()
        self._mic_buf = np.concatenate((self._mic_buf, incoming))
        out = []
        raw_out = []
        far_out = []
        while len(self._mic_buf) >= F_AEC:
            near, self._mic_buf = self._mic_buf[:F_AEC], self._mic_buf[F_AEC:]
            raw_out.append(near.copy())
            far = np.zeros(F_AEC, np.float32)
            count = min(F_AEC, len(self._far_buf))
            far[:count] = self._far_buf[:count]
            self._far_buf = self._far_buf[count:]
            far_out.append(far)
            if self.aec is not None and (self._far_active or np.sqrt(np.mean(far ** 2)) >= 0.005):
                try:
                    near = np.asarray(self.aec.process(near, far)).ravel().astype(np.float32)
                except Exception as exc:
                    self.aec_error = str(exc)
            out.append(near)
        clean = np.concatenate(out) if out else np.zeros(0, np.float32)
        self._since_far += len(incoming)
        if self._since_far > 4800 and not len(self._far_buf):
            self._far_active = False
        self._vad_buf = np.concatenate((self._vad_buf, clean))
        raw_clean = np.concatenate(raw_out) if raw_out else np.zeros(0, np.float32)
        self._raw_vad_buf = np.concatenate((self._raw_vad_buf, raw_clean))
        self._far_vad_buf = np.concatenate((self._far_vad_buf, np.concatenate(far_out) if far_out else np.zeros(0, np.float32)))
        started = ended = False
        if len(clean):
            self.level = float(np.sqrt(np.mean(clean ** 2)))
        while len(self._vad_buf) >= F_VAD:
            chunk, self._vad_buf = self._vad_buf[:F_VAD], self._vad_buf[F_VAD:]
            raw_chunk, self._raw_vad_buf = self._raw_vad_buf[:F_VAD], self._raw_vad_buf[F_VAD:]
            far_chunk, self._far_vad_buf = self._far_vad_buf[:F_VAD], self._far_vad_buf[F_VAD:]
            # All correlation histories end at this exact VAD boundary,
            # including zero render during gaps. Queued AEC/VAD samples must
            # not enter an earlier decision just because input blocks vary.
            self._advance_history(raw_chunk, far_chunk, chunk)
            self.echo_score = self._echo_similarity() if self._far_active else 0.0
            self.probability = self._prob(chunk)
            hot = self.probability >= self.vad_thresh
            # H3 quiet-room recording: AEC reduced 0.028 RMS echo to 0.001,
            # yet Silero called the residual speech. Only reject an onset
            # when BOTH strong AEC suppression and tiny residual energy agree.
            # Low-volume speech without that suppression remains eligible.
            if hot and not self.speaking and self._far_active:
                clean_rms = float(np.sqrt(np.mean(chunk ** 2)))
                raw_rms = float(np.sqrt(np.mean(raw_chunk ** 2)))
                if clean_rms < self.residual_floor and clean_rms < raw_rms * .2:
                    hot = False
                    self.residual_rejections += 1
            if not self.speaking and self._far_active and self.echo_score >= .6:
                hot = False
            if not self.speaking:
                # Brief echo masking need not erase genuine speech evidence.
                # The default requires three eligible frames in five (160 ms).
                self._onset_bits = ((self._onset_bits << 1) | int(hot)) & self._onset_mask
                self._hot = self._onset_bits.bit_count()
                if hot and self._hot >= self.start_chunks:
                    if self._sample_clock / 16000 - self._play_t0 >= self.holdoff_s:
                        self.speaking = True
                        started = True
                        self._quiet = 0
            else:
                self._quiet = 0 if hot else self._quiet + 1
                if self._quiet >= self.end_chunks:
                    self.speaking = False
                    ended = True
                    self._hot = 0
                    self._onset_bits = 0
        return self.speaking, started, ended, clean
