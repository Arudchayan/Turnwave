"""Held-out check for the two-threshold EOT schedule.

Rows are split deterministically by row index (even/odd). Thresholds are chosen
on the even half and scored once on the odd half. The saved Smart Turn scores are
reused; no model is run.
"""
import json
import math
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs' / 'eot_bench.json'
DEST = ROOT / 'outputs' / 'eot_holdout.json'
MIN_MS = 300


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


def decide(indexes, spans, observations, first, later, timeout):
    decisions, latencies = [], []
    for i in indexes:
        scores = observations['queries'][i]
        final = spans[i][4]
        duration_ms = (spans[i][2] - spans[i][1]) * 1000
        commit = None
        if scores and scores[0] >= first:
            commit = 300
        else:
            for position, score in enumerate(scores[1:], start=1):
                if score >= later:
                    commit = 300 + position * 200
                    break
        if commit is None and (final or duration_ms >= timeout):
            commit = timeout
        decisions.append(1 if commit is not None else 0)
        latencies.append(commit)
    return decisions, latencies


def main():
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    observations = source['temporal_policy']['observations']
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(source['dataset']['repo'], source['dataset']['file'],
                           repo_type='dataset')
    frame = pd.read_parquet(path).head(source['dataset']['rows'])
    spans = []
    for row_index, row in frame.iterrows():
        duration = float(row['duration'])
        for span in row['silence_spans']:
            start, end = float(span['start']), float(span['end'])
            if end - start > 0:
                spans.append((row_index, start, end, duration,
                              end >= duration - .05))
    eligible = [i for i, (_, start, end, _, _) in enumerate(spans)
                if (end - start) * 1000 >= MIN_MS]
    train = [i for i in eligible if spans[i][0] % 2 == 0]
    test = [i for i in eligible if spans[i][0] % 2 == 1]
    labels = lambda indexes: [observations['labels'][i] for i in indexes]
    report = dict(
        schema_version=1,
        protocol='deterministic even/odd row split; thresholds chosen on train',
        pauses=dict(total=len(eligible), train=len(train), test=len(test)),
        train_mid=sum(1 for y in labels(train) if not y),
        test_mid=sum(1 for y in labels(test) if not y),
        baselines={},
        selected={},
    )
    # Fixed 480 and single 300 ms query on the held-out half.
    fixed = [1 if (spans[i][2] - spans[i][1]) * 1000 >= 480 else 0 for i in test]
    report['baselines']['fixed_480ms'] = metrics(labels(test), fixed)
    single = [1 if observations['queries'][i] and observations['queries'][i][0] >= .5
              else 0 for i in test]
    report['baselines']['single_300ms_0.5'] = metrics(labels(test), single)
    # Choose one schedule per timeout on train, then score it once on test.
    for timeout in (1000, 2000):
        candidates = []
        for first in (.5, .6, .7, .8, .9):
            for later in (.3, .4, .5, .6, .7):
                decisions, latencies = decide(train, spans, observations,
                                              first, later, timeout)
                row = metrics(labels(train), decisions)
                if row['eot_recall'] == 1.0:
                    final_latencies = [lat for lat, y in zip(latencies, labels(train))
                                       if y and lat is not None]
                    row.update(first_threshold=first, later_threshold=later,
                               timeout_ms=timeout,
                               p90_latency_ms=sorted(final_latencies)[
                                   int(.9 * (len(final_latencies) - 1))])
                    candidates.append(row)
        if not candidates:
            continue
        candidates.sort(key=lambda row: (row['false_cutoff'], row['p90_latency_ms']))
        chosen = candidates[0]
        decisions, latencies = decide(test, spans, observations,
                                      chosen['first_threshold'],
                                      chosen['later_threshold'], timeout)
        test_row = metrics(labels(test), decisions)
        final_latencies = [lat for lat, y in zip(latencies, labels(test))
                           if y and lat is not None]
        test_row.update(first_threshold=chosen['first_threshold'],
                        later_threshold=chosen['later_threshold'],
                        timeout_ms=timeout,
                        p90_latency_ms=sorted(final_latencies)[
                            int(.9 * (len(final_latencies) - 1))])
        report['selected'][str(timeout)] = dict(
            train=chosen, test=test_row)
    report['limits'] = ('One deterministic split, English only, saved offline '
                        'scores; not a live or multi-seed validation.')
    DEST.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8',
                    newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
