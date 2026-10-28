"""Zero-shot cross-lingual EOT check with the English-selected schedule.

The thresholds (0.9 first, 0.7 later) are fixed from English; no tuning is done
on the target language. This measures whether the policy transfers.
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from synth.eot_bench import SmartTurn, _decode, _spans

DEST = ROOT / 'outputs'
DATASET = 'livekit/eot-bench-data'
THRESHOLDS = ((.9, .7), (.5, .5))
TIMEOUTS = (1000, 2000)
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


def evaluate(language, limit, model):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(DATASET,
                           f'data/{language}/validation-00000-of-00001.parquet',
                           repo_type='dataset')
    frame = pd.read_parquet(path)
    if limit:
        frame = frame.head(limit)
    rows = []
    for _, row in frame.iterrows():
        audio = _decode(row['audio'])
        duration = float(row['duration'])
        for span in _spans(row):
            pause_ms = (span['end'] - span['start']) * 1000
            if pause_ms < MIN_MS:
                continue
            scores = [model.score(audio, span['start'] + wait / 1000.0)
                      for wait in (300, 500, 700, 900)
                      if wait <= pause_ms]
            rows.append(dict(label=1 if span['final'] else 0,
                             duration_ms=pause_ms, scores=scores))
    labels = [row['label'] for row in rows]
    fixed = [1 if row['duration_ms'] >= 480 else 0 for row in rows]
    report = dict(
        schema_version=1,
        language=language,
        dataset=dict(repo=DATASET,
                     file=f'data/{language}/validation-00000-of-00001.parquet',
                     rows=len(frame)),
        model=dict(repo='pipecat-ai/smart-turn-v3',
                   file='smart-turn-v3.2-cpu.onnx', sha256=model.sha256),
        pauses=len(rows),
        mid_pauses=sum(1 for y in labels if not y),
        final_pauses=sum(1 for y in labels if y),
        policies=dict(fixed_480ms=metrics(labels, fixed)),
        note='Thresholds fixed from English; zero-shot language transfer.',
    )
    for first, later in THRESHOLDS:
        single = [1 if row['scores'] and row['scores'][0] >= first else 0
                  for row in rows]
        report['policies'][f'single_300_{first}'] = metrics(labels, single)
        for timeout in TIMEOUTS:
            decisions, latencies = [], []
            for row in rows:
                commit = None
                if row['scores'] and row['scores'][0] >= first:
                    commit = 300
                else:
                    for index, score in enumerate(row['scores'][1:], start=1):
                        if score >= later:
                            commit = 300 + index * 200
                            break
                if commit is None and (row['label'] == 1
                                       or row['duration_ms'] >= timeout):
                    commit = timeout
                decisions.append(1 if commit is not None else 0)
                latencies.append(commit)
            result = metrics(labels, decisions)
            final_latencies = [lat for lat, y in zip(latencies, labels)
                               if y and lat is not None]
            result['p90_latency_ms'] = (sorted(final_latencies)[
                int(.9 * (len(final_latencies) - 1))] if final_latencies else None)
            report['policies'][f'two_{first}_{later}_{timeout}'] = result
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--languages', nargs='+', default=['de', 'es'])
    parser.add_argument('--limit', type=int, default=400)
    args = parser.parse_args()
    model = SmartTurn()
    reports = {}
    for language in args.languages:
        print('evaluating', language, flush=True)
        report = evaluate(language, args.limit, model)
        reports[language] = report
        path = DEST / f'eot_crosslingual_{language}.json'
        path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8',
                        newline='\n')
        summary = {key: (row['false_cutoff'], row['eot_recall'],
                         row.get('p90_latency_ms'))
                   for key, row in report['policies'].items()}
        print(json.dumps(dict(language=language, pauses=report['pauses'],
                              policies=summary), indent=2), flush=True)
    combined = DEST / 'eot_crosslingual.json'
    combined.write_text(json.dumps(reports, indent=2) + '\n', encoding='utf-8',
                        newline='\n')
    print('wrote', combined)


if __name__ == '__main__':
    main()
