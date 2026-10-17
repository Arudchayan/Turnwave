"""Offline queued-render and playback-gap controls on the real AEC and VAD."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from duplex import DuplexFrontend
from synth.acoustic_stress import read


def run(folder, baseline_ref):
    if (folder / 'clock-stress.json').exists():
        raise FileExistsError('Use a new experiment folder')
    baseline_source = subprocess.check_output(['git', 'show', baseline_ref + ':src/duplex.py'], cwd=ROOT)
    (folder / 'baseline-duplex.py').write_bytes(baseline_source)
    baseline = types.ModuleType('baseline_duplex')
    exec(compile(baseline_source, 'baseline-duplex.py', 'exec'), baseline.__dict__)
    far = np.tile(read('_assistant.wav'), 3)[:8 * 16000]
    far = np.pad(far, (0, (-len(far)) % 512))
    user = read('user_barge.wav')
    first = np.flatnonzero(abs(user) > .02)[0]
    user = user[max(0, first - 160):]
    rows = []
    for gaps in (False, True):
        render = far.copy()
        if gaps:
            render[48*512:56*512] = 0
            render[96*512:102*512] = 0
        chunks = render.reshape(-1, 512)
        # Network stalls do not call push_far with zeros. Within an active
        # segment, allow the output device to have several queued chunks.
        for ahead in (0, 4):
            deliveries = {}
            segment_start = 0
            for index, chunk in enumerate(chunks):
                if not np.any(chunk):
                    segment_start = index + 1
                    continue
                tick = max(segment_start, index - ahead)
                deliveries.setdefault(tick, []).append(chunk)
            for overlap in (False, True):
                microphone = np.zeros(len(render), np.float32)
                for delay_ms, gain in ((80, .8), (105, .24), (150, .12)):
                    offset = delay_ms * 16
                    microphone[offset:] += render[:-offset] * gain
                rng = np.random.default_rng(20261016)
                microphone += rng.normal(0, .0005, len(render)).astype(np.float32)
                onset = 112 * 512
                if overlap:
                    count = min(len(user), len(render) - onset)
                    microphone[onset:onset+count] += .8 * user[:count]
                for label, frontend in [('baseline', baseline.DuplexFrontend), ('candidate', DuplexFrontend)]:
                    front = frontend(stream_delay_ms=80)
                    front.warmup()
                    starts = []
                    for tick, near in enumerate(microphone.reshape(-1, 512)):
                        for reference in deliveries.get(tick, []):
                            front.push_far(reference)
                        _, started, _, _ = front.update(near)
                        if started:
                            starts.append((tick + 1) * .032)
                    onset_s = onset / 16000
                    false = [value for value in starts if not overlap or value < onset_s]
                    latency = next((value - onset_s for value in starts if value >= onset_s), None) if overlap else None
                    row = dict(version=label, gaps=gaps, queued_ahead_ms=ahead*32,
                               overlap=overlap, starts=starts, false_starts=len(false),
                               barge_s=round(latency, 3) if latency is not None else None,
                               aec_error=front.aec_error)
                    row['passed'] = not false and not front.aec_error and (not overlap or latency is not None and latency < .5)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    report = dict(cases=rows, baseline_ref=baseline_ref, seed=20261016,
                  note='Real AEC/VAD; deterministic synthetic device queue and network stalls. No hardware capture or playback.')
    (folder / 'clock-stress.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--baseline-ref', default='a2cbf85')
    args = parser.parse_args()
    run(args.folder, args.baseline_ref)
