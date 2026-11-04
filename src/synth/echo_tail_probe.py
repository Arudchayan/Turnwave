"""Compare delayed echo-tail and near-speech detection without devices.

The far-end is a bundled generated reply; the near-end is one public speech
clip or a generated fallback. This synthetic delay-and-gain mixture is not a
Bluetooth, room, or real-device measurement.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from duplex import DuplexFrontend
from synth.floor_acoustic import human_near, read_wav, resample


def run_case(far, near, delay_ms, case, extended):
    settings = {'echo_tail_ms': 1500, 'echo_history_ms': 1600} if extended else {}
    front = DuplexFrontend(aec=True, holdoff_s=0, **settings)
    count, start, speech_at = 88000, 8000, 48800
    playback = np.zeros(count, np.float32)
    playback[start:start + len(far)] = far
    echo = np.zeros(count, np.float32)
    delay_samples = round(delay_ms * 16)
    echo[delay_samples:] = playback[:-delay_samples] * .5
    speech = np.zeros(count, np.float32)
    speech[speech_at:speech_at + len(near)] = near
    gain = .25 if case.endswith('quiet') else 1.0
    mic = (echo if case.startswith('overlap') or case == 'echo_only' else 0)
    if case != 'echo_only':
        mic = mic + speech * gain
    starts = []
    started = time.perf_counter()
    for offset in range(0, count, 320):
        if offset < start + len(far) and offset + 320 > start:
            front.push_far(playback[offset:offset + 320], 16000)
        _, onset, _, _ = front.update(mic[offset:offset + 320])
        if onset:
            starts.append(round(offset / 16000, 3))
    return dict(profile='long' if extended else 'standard', delay_ms=delay_ms,
                case=case, onsets_s=starts, speech_start_s=speech_at / 16000,
                runtime_s=round(time.perf_counter() - started, 3), aec_error=front.aec_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    far, rate = read_wav(ROOT / 'research/examples/pass-reply.wav')
    far = resample(far, rate)[:48000]
    try:
        near, rate = human_near()
        near_source = 'librispeech-dummy'
    except Exception:
        near, rate = read_wav(ROOT / 'research/examples/failure-reply.wav')
        near_source = 'generated-failure-reply'
    near = resample(near, rate)[:16000]
    for signal in (far, near):
        rms = float(np.sqrt(np.mean(signal ** 2)))
        if rms <= 0:
            raise ValueError('Fixture contains no audio')
        signal *= (.08 if signal is far else .06) / rms
    rows = []
    for extended in (False, True):
        for delay in (80, 300, 500, 800, 1100):
            for case in ('echo_only', 'near_full', 'near_quiet', 'overlap_full', 'overlap_quiet'):
                row = run_case(far, near, delay, case, extended)
                rows.append(row)
                print(json.dumps(row), flush=True)
    report = dict(scope='One generated far-end and one near-end clip; synthetic delay/gain, no device or room. Fixed profile order.',
                  near_source=near_source, rows=rows)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
