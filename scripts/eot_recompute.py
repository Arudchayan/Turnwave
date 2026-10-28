"""Recompute EOT policy metrics on one common pause denominator.

The first EOT run let short pauses enter the fixed and hybrid denominators
without being eligible for a decision, which diluted their false cut-off rates.
This script aligns every policy to pauses of at least 300 ms, the first query
point, using the saved query scores. It runs no model.
"""
import json
import math
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs' / 'eot_bench.json'
DEST = ROOT / 'outputs' / 'eot_recomputed.json'
MIN_MS = 300
THRESHOLD = .5


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0, centre - half), 4), round(min(1, centre + half), 4)]


def metrics(labels, decisions):
    tp = sum(1 for y, d in zip(labels, decisions) if y and d)
    fp = sum(1 for y, d in zip(labels, decisions) if not y and d)
    fn = sum(1 for y, d in zip(labels, decisions) if y and not d)
    tn = sum(1 for y, d in zip(labels, decisions) if not y and not d)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                false_cutoff=round(fp / max(1, fp + tn), 4),
                false_cutoff_ci=wilson(fp, fp + tn),
                eot_recall=round(tp / max(1, tp + fn), 4),
                eot_recall_ci=wilson(tp, tp + fn),
                precision=round(tp / max(1, tp + fp), 4))


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
    if len(spans) != len(observations['labels']):
        raise SystemExit('Saved observations do not align with the dataset')
    eligible = [i for i, (start, end, _, _) in enumerate(spans)
                if (end - start) * 1000 >= MIN_MS]
    labels = [observations['labels'][i] for i in eligible]
    fixed = [1 if (spans[i][1] - spans[i][0]) * 1000 >= 480 else 0
             for i in eligible]
    temporal = []
    for i in eligible:
        scores = observations['queries'][i]
        temporal.append(1 if any(score >= THRESHOLD for score in scores) else 0)
    hybrids = {}
    for timeout in (600, 800, 1000, 1200, 1500, 2000):
        decisions, latencies = [], []
        for i in eligible:
            scores = observations['queries'][i]
            commit_index = next((j for j, score in enumerate(scores)
                                 if score >= THRESHOLD), None)
            final = spans[i][3]
            duration_ms = (spans[i][1] - spans[i][0]) * 1000
            if commit_index is not None:
                decisions.append(1)
                latencies.append(MIN_MS + commit_index * 200)
            elif final:
                decisions.append(1)
                latencies.append(timeout)
            elif duration_ms >= timeout:
                decisions.append(1)
                latencies.append(timeout)
            else:
                decisions.append(0)
        row = metrics(labels, decisions)
        row['timeout_ms'] = timeout
        row['final_latency_ms'] = dict(
            median=round(sorted(latencies)[len(latencies) // 2], 1) if latencies else None,
            p90=round(sorted(latencies)[int(.9 * (len(latencies) - 1))], 1) if latencies else None,
        )
        hybrids[str(timeout)] = row
    # Single 300 ms query, using the separately saved eligible observations.
    single = dict(source['budgets']['300']['at_0_5'])
    single['false_cutoff_ci'] = wilson(single['fp'], single['fp'] + single['tn'])
    single['eot_recall_ci'] = wilson(single['tp'], single['tp'] + single['fn'])
    report = dict(
        schema_version=1,
        protocol='all policies scored on pauses >= 300 ms; saved scores only',
        pauses=len(eligible),
        mid_pauses=sum(1 for y in labels if not y),
        final_pauses=sum(1 for y in labels if y),
        fixed_480ms=metrics(labels, fixed),
        single_300ms=single,
        temporal_no_timeout=metrics(labels, temporal),
        hybrid_timeouts=hybrids,
        note='Descriptive recomputation; no model was run by this script.',
    )
    DEST.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8',
                    newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
