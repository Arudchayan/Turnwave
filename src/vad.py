"""Energy VAD with adaptive noise floor + barge-in detector (stdlib+numpy only).

No webrtcvad dependency, no ML. Honest limits: energy-based, so loud
non-speech trips it; no AEC — during playback the mic hears the speakers,
so barge-in uses a raised threshold + headphones are recommended.
"""
import numpy as np


class Vad:
    def __init__(self, rate=16000, frame_ms=30, thresh_mult=2.5,
                 start_frames=4, end_silence_ms=800):
        self.rate = rate
        self.frame = int(rate * frame_ms / 1000)
        self.thresh_mult = thresh_mult
        self.start_frames = start_frames
        self.end_frames = int(end_silence_ms / frame_ms)
        self.noise = 1e-4
        self.speaking = False
        self._hot = 0
        self._quiet = 0

    def reset(self):
        self.speaking = False
        self._hot = 0
        self._quiet = 0

    def update(self, samples):
        """Feed mono float32 samples. Returns (speaking, started, ended)."""
        started = ended = False
        n = self.frame
        for i in range(0, len(samples) - n + 1, n):
            rms = float(np.sqrt(np.mean(samples[i:i + n] ** 2) + 1e-12))
            self.noise = 0.995 * self.noise + 0.005 * min(rms, 0.02)
            hot = rms > max(self.noise * self.thresh_mult, 0.008)
            if not self.speaking:
                self._hot = self._hot + 1 if hot else 0
                if self._hot >= self.start_frames:
                    self.speaking = True
                    started = True
                    self._quiet = 0
            else:
                if hot:
                    self._quiet = 0
                else:
                    self._quiet += 1
                    if self._quiet >= self.end_frames:
                        self.speaking = False
                        ended = True
                        self._hot = 0
        return self.speaking, started, ended


class BargeIn:
    """Raised-threshold speech detector for use DURING assistant playback."""

    def __init__(self, rate=16000, frame_ms=30, thresh_mult=6.0,
                 trigger_frames=6, grace_ms=150):
        self.rate = rate
        self.frame = int(rate * frame_ms / 1000)
        self.thresh_mult = thresh_mult
        self.trigger_frames = trigger_frames
        self.grace = int(rate * grace_ms / 1000)
        self.noise = 1e-4
        self._seen = 0
        self._hot = 0

    def reset(self):
        self._seen = 0
        self._hot = 0

    def update(self, samples):
        """Returns True once when user speech likely starts over playback."""
        n = self.frame
        for i in range(0, len(samples) - n + 1, n):
            self._seen += n
            if self._seen < self.grace:
                continue
            rms = float(np.sqrt(np.mean(samples[i:i + n] ** 2) + 1e-12))
            self.noise = 0.99 * self.noise + 0.01 * min(rms, 0.05)
            if rms > max(self.noise * self.thresh_mult, 0.02):
                self._hot += 1
                if self._hot >= self.trigger_frames:
                    return True
            else:
                self._hot = max(0, self._hot - 1)
        return False
