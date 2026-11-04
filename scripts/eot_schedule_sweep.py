"""Sweep two-threshold EOT schedules on saved Smart Turn query scores.

A high first-query threshold can avoid premature cut-offs; a lower threshold on
later re-queries can still recover a final turn. This uses the saved query
scores only and runs no model.
"""
import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs' / 'eot_bench.json'
DEST = ROOT / 'outputs' / 'eot_schedule_sweep.json'
MIN_MS = 300


def main():
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    observations = source['temporal_policy']['observations']
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(source['dataset']['repo'], source['dataset']['file'],
                           repo_type='dataset')
    frame = pd.read_parquet(path).head(source['dataset']['rows'])
    spans = []
    for _, row in frame.iterrows():
        duration = float(row['duration'])
        for span in row['silence_spans']:
            start, end = float(span['start']), float(span['end'])
            if end - start > 0:
                spans.append((start, end, duration, end >= duration - .05))
    eligible = [i for i, (start, end, _, _) in enumerate(spans)
                if (end - start) * 1000 >= MIN_MS]
    labels = [observations['labels'][i] for i in eligible]
    rows = []
    for first in (.5, .6, .7, .8, .9):
        for later in (.3, .4, .5, .6, .7):
            for timeout in (1000, 2000):
                decisions, latencies = [], []
                for i in eligible:
                    scores = observations['queries'][i]
                    final = spans[i][3]
                    duration_ms = (spans[i][1] - spans[i][0]) * 1000
                    commit = None
                    if scores and scores[0] >= first:
                        commit = 300
                    else:
                        for index, score in enumerate(scores[1:], start=1):
                            if score >= later:
                                commit = 300 + index * 200
                                break
                    if commit is None:
                        if final or duration_ms >= timeout:
                            commit = timeout
                    decisions.append(1 if commit is not None else 0)
                    latencies.append(commit)
                fp = sum(1 for y, d in zip(labels, decisions) if not y and d)
                tn = sum(1 for y, d in zip(labels, decisions) if not y and not d)
                tp = sum(1 for y, d in zip(labels, decisions) if y and d)
                fn = sum(1 for y, d in zip(labels, decisions) if y and not d)
                final_latencies = [lat for lat, y in zip(latencies, labels)
                                   if y and lat is not None]
                rows.append(dict(
                    first_threshold=first, later_threshold=later,
                    timeout_ms=timeout,
                    tp=tp, fp=fp, fn=fn, tn=tn,
                    false_cutoff=round(fp / max(1, fp + tn), 4),
                    eot_recall=round(tp / max(1, tp + fn), 4),
                    precision=round(tp / max(1, tp + fp), 4),
                    median_latency_ms=(sorted(final_latencies)[len(final_latencies) // 2]
                                       if final_latencies else None),
                    p90_latency_ms=(sorted(final_latencies)[int(.9 * (len(final_latencies) - 1))]
                                    if final_latencies else None),
                ))
    # Keep only schedules with full recall and rank by false cut-off, then p90.
    full = [row for row in rows if row['eot_recall'] == 1.0]
    full.sort(key=lambda row: (row['false_cutoff'], row['p90_latency_ms']))
    report = dict(
        schema_version=1,
        protocol='saved 300/500/700/900 ms scores; two-threshold schedule',
        pauses=len(eligible),
        rows=rows,
        full_recall_top=full[:15],
        note='Screening only; not a new model run.',
    )
    DEST.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8',
                    newline='\n')
    print(json.dumps(report['full_recall_top'], indent=2))


if __name__ == '__main__':
    main()
