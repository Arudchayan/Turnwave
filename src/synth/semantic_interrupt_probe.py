"""Compare local ASR on clean synthetic utterances and captured overlapping audio.

This diagnostic makes local model calls, writes synthetic transcripts, and does
not open a microphone or loudspeaker. Use only generated/public fixtures; it
is not an evaluation of natural speech or a production intent classifier.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from backend_session import backend_session
from client import wav16k_bytes
from lfm3_client import LFM3Client
from synth import duplex_barge_probe as probe

FIXTURES = ('david-ack.wav', 'david-yes.wav', 'david-stop.wav', 'david-wait.wav',
            'zira-ack.wav', 'zira-yes.wav', 'zira-stop.wav', 'zira-wait.wav')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', required=True, help='folder of generated benchmark WAVs')
    parser.add_argument('--url', default='http://127.0.0.1:18085')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    folder = Path(args.fixtures)
    for name in FIXTURES:
        if not (folder / name).is_file():
            parser.error('Missing generated fixture: ' + name)
    far, rate = probe.read_wav(ROOT / 'research/examples/pass-reply.wav')
    far = probe.resample(far, rate)
    far *= .05 / np.sqrt(np.mean(far ** 2))
    captured = []
    base_assistant = probe.CaptureAssistant

    class CapturingAssistant(base_assistant):
        def submit_audio(self, audio):
            captured.append(audio.copy())
            super().submit_audio(audio)

    # The benchmark is single-threaded; the application still owns its capture
    # worker. Only the copied, synthetic capture is sent to the ASR endpoint.
    probe.CaptureAssistant = CapturingAssistant
    os.environ.pop('LOCALVOICE_FLOOR_MODE', None)
    os.environ.pop('LOCALVOICE_EOT_MODEL', None)
    rows = []
    with backend_session(args.url, test=True):
        voice = LFM3Client(args.url, timeout=180)
        try:
            for name in FIXTURES:
                path = folder / name
                near, rate = probe.read_wav(path)
                near = probe.resample(near, rate)
                hot = [i for i in range(0, len(near) - 320 + 1, 320)
                       if float(np.sqrt(np.mean(near[i:i + 320] ** 2))) > .003]
                if not hot:
                    raise ValueError('Fixture has no speech above the trim threshold: ' + name)
                near = near[max(0, hot[0] - 320):min(len(near), hot[-1] + 640)]
                near *= .06 / np.sqrt(np.mean(near ** 2))
                started = time.monotonic()
                clean = voice.asr_text(wav16k_bytes(near))
                clean_s = time.monotonic() - started
                captured.clear()
                result = probe.run_case('utterance', 'current', 1.0, far, near, echo_threshold=.9)
                noisy, noisy_s = None, None
                if captured:
                    started = time.monotonic()
                    noisy = voice.asr_text(wav16k_bytes(captured[0]))
                    noisy_s = time.monotonic() - started
                rows.append({'fixture': name, 'fixture_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                             'intended': name.split('-')[1].split('.')[0],
                             'clean_asr': clean, 'clean_seconds': round(clean_s, 2),
                             'capture_asr': noisy, 'capture_seconds': round(noisy_s, 2) if noisy_s is not None else None,
                             'captured_samples': len(captured[0]) if captured else 0,
                             'cancel_ms': result['cancel_ms'], 'capture_error': result['error'],
                             'aec_error': result['aec_error']})
                print(json.dumps(rows[-1]), flush=True)
        finally:
            voice.close()
            probe.CaptureAssistant = base_assistant
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'scope': 'synthetic utterances, virtual capture, LFM3 local ASR',
                                  'echo_threshold': .9, 'rows': rows}, indent=2) + '\n', encoding='utf-8')
    if any(row['capture_error'] or row['aec_error'] for row in rows):
        raise SystemExit('Capture or echo processing reported an error')


if __name__ == '__main__':
    main()
