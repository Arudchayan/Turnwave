"""Replay synthetic overlapping speech through the app's real capture loop.

No audio device or model backend is opened. This is a diagnostic with a simple
simulated acoustic path, not a room, headset, or conversational evaluation.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import queue
import sys
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from app import Assistant, Job
from duplex import DuplexFrontend
from synth.echo_tail_probe import human_near, read_wav, resample


class StepQueue(queue.Queue):
    """Acknowledge each block when the capture loop requests the next one."""
    def __init__(self):
        super().__init__(maxsize=1)
        self.finished = threading.Condition()
        self.completed = 0
        self.previous = False

    def get(self, *args, **kwargs):
        with self.finished:
            if self.previous:
                self.completed += 1
                self.previous = False
                self.finished.notify_all()
        result = super().get(*args, **kwargs)
        with self.finished:
            self.previous = True
        return result

    def wait_completed(self, count):
        with self.finished:
            if not self.finished.wait_for(lambda: self.completed >= count, timeout=10):
                raise TimeoutError('Capture loop did not finish a frame')


class VirtualMic:
    device = -1
    def __init__(self): self.q = StepQueue()
    def stop(self): pass
    def drain(self): pass


class VirtualSpeaker:
    def __init__(self):
        self.playing = True
        self.gain = 1.0
        self.sample_clock = 0
        self.stop_sample = None
        self.gain_events = []

    def set_gain(self, gain, *args):
        self.gain = gain
        self.gain_events.append([round(self.sample_clock / 16, 1), gain])

    def stop(self):
        if self.playing and self.stop_sample is None:
            self.stop_sample = self.sample_clock
        self.playing = False


class NoBackend:
    base = 'offline://capture-only'


class CaptureAssistant(Assistant):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.captured_samples = []

    def submit_audio(self, audio):
        # Measure the app's cancellation/submission decision without generating
        # a reply; keep the same interruption side effects as a new turn.
        self.captured_samples.append(len(audio))
        self.interrupt('new turn', announce=False)


def build_tracks(kind, far, near):
    size = round(4.6 * 16000)
    playback = np.zeros(size, np.float32)
    voice = np.zeros(size, np.float32)
    far_length = min(len(far), round(3.3 * 16000))
    playback[8000:8000 + far_length] = far[:far_length]
    if kind == 'utterance':
        count = min(len(near), size - 24000)
        voice[24000:24000 + count] = near[:count]
    elif kind != 'echo_only':
        length = {'short_fragment': .22, 'sustained': .9, 'mid_pause': .4}[kind]
        count = round(length * 16000)
        voice[24000:24000 + count] = near[:count]
        if kind == 'mid_pause':
            begin = round(2.4 * 16000)
            voice[begin:begin + count] = near[8000:8000 + count]
    return playback, voice


def run_case(kind, mode, gain, far, near, vad_threshold=.5, start_chunks=3,
             echo_threshold=.6):
    playback, voice = build_tracks(kind, far, near * gain)
    rendered = np.zeros_like(playback)
    noise = np.random.default_rng(20261028).normal(0, .0015, len(playback)).astype(np.float32)
    class ExperimentalFrontend(DuplexFrontend):
        def _echo_similarity(self):
            # The app gates at 0.6. Scaling the score changes only the
            # experimental gate; production frontend defaults stay untouched.
            return super()._echo_similarity() * (.6 / echo_threshold)
    front = ExperimentalFrontend(aec=True, holdoff_s=0, vad_thresh=vad_threshold,
                                 start_chunks=start_chunks)
    front.set_delay(80)
    front.warmup()
    mic, speaker = VirtualMic(), VirtualSpeaker()
    options = {'mic': mic, 'player': speaker, 'front': front, 'client': NoBackend()}
    if mode != 'current':
        if 'floor_mode' not in inspect.signature(Assistant).parameters:
            raise ValueError('This checkout does not include the requested experimental floor mode')
        options['floor_mode'] = mode
    app = CaptureAssistant(**options)
    job = Job(id=1)
    app._active = job
    app.status = 'speaking'
    app._spawn(app._loop)
    try:
        for offset in range(0, len(playback), 320):
            speaker.sample_clock = offset + 320
            if speaker.playing:
                # Instant gain is an approximation; the real player ramps gain.
                rendered[offset:offset + 320] = playback[offset:offset + 320] * speaker.gain
                front.push_far(rendered[offset:offset + 320])
                job.played_samples += 480
            block = rendered[offset:offset + 320] + voice[offset:offset + 320] + noise[offset:offset + 320]
            if offset >= 1280:
                block += rendered[offset - 1280:offset - 960] * .45
            mic.q.put(block, timeout=10)
            mic.q.wait_completed(offset // 320 + 1)
        return {'case': kind, 'mode': mode, 'near_gain': gain,
                'cancel_ms': speaker.stop_sample / 16 if speaker.stop_sample is not None else None,
                'captured_samples': list(app.captured_samples), 'gain_events': list(speaker.gain_events),
                'aec_error': front.aec_error, 'error': app.error}
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    near_source = parser.add_mutually_exclusive_group(required=True)
    near_source.add_argument('--near', help='mono PCM16 WAV for the near-end speech')
    near_source.add_argument('--public-human', action='store_true', help='download/load the public LibriSpeech dummy clip')
    parser.add_argument('--mode', choices=('current', 'reversible'), default='current')
    parser.add_argument('--utterance', action='store_true', help='replay the full near-end utterance at two gains')
    parser.add_argument('--trim-silence', action='store_true', help='trim leading and trailing low-energy 20 ms frames')
    parser.add_argument('--vad-threshold', type=float, default=.5, help='experimental speech probability threshold')
    parser.add_argument('--start-chunks', type=int, default=3, help='experimental onset confirmation frame count')
    parser.add_argument('--echo-threshold', type=float, default=.6, help='experimental onset echo-correlation threshold')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if not 0 < args.vad_threshold <= 1 or not 1 <= args.start_chunks <= 10 or not 0 < args.echo_threshold <= 1:
        parser.error('Thresholds must be in (0, 1] and start chunks in [1, 10]')
    # Experiment mode is explicit and does not inherit application settings.
    os.environ.pop('LOCALVOICE_FLOOR_MODE', None)
    os.environ.pop('LOCALVOICE_EOT_MODEL', None)
    far, rate = read_wav(ROOT / 'research/examples/pass-reply.wav')
    far = resample(far, rate)
    near, rate = human_near() if args.public_human else read_wav(Path(args.near))
    near = resample(near, rate)
    if len(far) < 16000 or len(near) < 16000:
        raise ValueError('Far and near fixtures must contain at least 1 second of audio each')
    if args.trim_silence:
        hot = [i for i in range(0, len(near) - 320 + 1, 320)
               if float(np.sqrt(np.mean(near[i:i + 320] ** 2))) > .003]
        if not hot:
            raise ValueError('Near fixture contains no speech above the trim threshold')
        near = near[max(0, hot[0] - 320):min(len(near), hot[-1] + 640)]
    for signal, target in ((far, .05), (near, .06)):
        rms = float(np.sqrt(np.mean(signal ** 2)))
        if rms <= 0:
            raise ValueError('Fixture contains no audio')
        signal *= target / rms
    cases = ('utterance',) if args.utterance else ('echo_only', 'short_fragment', 'sustained', 'mid_pause')
    rows = [run_case(kind, args.mode, gain, far, near, args.vad_threshold, args.start_chunks, args.echo_threshold)
            for kind in cases
            for gain in (1.0, .5)]
    report = {'scope': 'real app capture loop, virtual devices, one synthetic echo path and one speech pair',
              'mode': args.mode, 'near_source': 'librispeech-dummy' if args.public_human else 'supplied WAV',
              'vad_threshold': args.vad_threshold, 'start_chunks': args.start_chunks,
              'echo_threshold': args.echo_threshold,
              'near_duration_ms': round(len(near) / 16, 1), 'trimmed': args.trim_silence,
              'near_normalized_sha256': hashlib.sha256(near.tobytes()).hexdigest(),
              'far_normalized_sha256': hashlib.sha256(far.tobytes()).hexdigest(),
              'onset_ms': 1500, 'echo_delay_ms': 80, 'frame_ms': 20,
              'limits': 'One clipped speech pair, instant simulated gain, no room, device, backend, or semantic labels.',
              'rows': rows}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))
    if any(row['error'] or row['aec_error'] for row in rows):
        raise SystemExit('Capture or echo processing reported an error')


if __name__ == '__main__':
    main()
