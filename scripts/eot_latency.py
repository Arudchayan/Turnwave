"""Measure Smart Turn query latency on this machine.

This is a diagnostic for whether the 300 ms first-query schedule is feasible on
the target CPU. It times the actual ONNX scorer on public 8 s clips.
"""
import json
import pathlib
import statistics
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / 'src'))
from synth.eot_bench import SmartTurn, _decode, _spans

SOURCE = ROOT / 'outputs' / 'eot_bench.json'
DEST = ROOT / 'outputs' / 'eot_latency.json'


def main(clips=20):
    meta = json.loads(SOURCE.read_text(encoding='utf-8'))
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(meta['dataset']['repo'], meta['dataset']['file'],
                           repo_type='dataset')
    frame = pd.read_parquet(path)
    model = SmartTurn()
    samples, labels = [], []
    for _, row in frame.iterrows():
        spans = _spans(row)
        final = next((span for span in spans if span['final']), None)
        if final is None:
            continue
        audio = _decode(row['audio'])
        decision = final['start'] + .3
        samples.append((audio, decision))
        labels.append(str(row['id']))
        if len(samples) >= clips:
            break
    timings = []
    for index, (audio, decision) in enumerate(samples):
        start = time.perf_counter()
        model.score(audio, decision)
        elapsed = (time.perf_counter() - start) * 1000
        if index:  # discard the first warm-up call
            timings.append(elapsed)
    timings.sort()
    report = dict(
        schema_version=1,
        model=dict(repo=meta['model']['repo'], file=meta['model']['file'],
                   sha256=model.sha256),
        clips=len(samples),
        measured=len(timings),
        cpu_threads='onnxruntime default (inter_op=1)',
        mean_ms=round(statistics.mean(timings), 2) if timings else None,
        median_ms=round(statistics.median(timings), 2) if timings else None,
        p90_ms=round(timings[int(.9 * (len(timings) - 1))], 2) if timings else None,
        min_ms=round(min(timings), 2) if timings else None,
        max_ms=round(max(timings), 2) if timings else None,
        note='One local CPU run; not a hardware-independent benchmark.',
    )
    DEST.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8',
                    newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
