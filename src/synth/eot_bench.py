"""Offline end-of-turn benchmark on public conversational pauses.

This is a research diagnostic, not a production endpointing path. It scores
clips from livekit/eot-bench-data with Smart Turn v3.2 and compares three
policies over the same pauses:

* fixed silence endpoint (the app's current 480 ms rule),
* one Smart Turn query at the first 300 ms of silence,
* a temporal policy that re-queries while the pause continues.

A mid-turn pause that crosses the decision threshold is a false cut-off; a
final pause that crosses it is an end-of-turn detection. The script records
both, so a policy cannot look good by simply never committing.

Run from the repository root:

    python -m src.synth.eot_bench --limit 400 --out outputs/eot_bench.json
"""
import argparse
import hashlib
import io
import json
import pathlib
import sys
import wave
from datetime import datetime, timezone

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATASET = 'livekit/eot-bench-data'
DATASET_FILE = 'data/en/validation-00000-of-00001.parquet'
MODEL = 'pipecat-ai/smart-turn-v3'
MODEL_FILE = 'smart-turn-v3.2-cpu.onnx'
SAMPLE_RATE = 16000
BUDGETS_MS = (300, 500, 800, 1000)
THRESHOLD = 0.5


def _load_rows(limit):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(DATASET, DATASET_FILE, repo_type='dataset')
    frame = pd.read_parquet(path)
    if limit:
        frame = frame.head(int(limit))
    return frame, path


def _decode(audio):
    data = audio['bytes'] if isinstance(audio, dict) else audio
    with wave.open(io.BytesIO(data)) as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError('Expected mono PCM16 audio')
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
        if wav.getframerate() != SAMPLE_RATE:
            raise ValueError('Expected 16 kHz audio')
    return samples.astype(np.float32) / 32768.0


class SmartTurn:
    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import WhisperFeatureExtractor
        path = hf_hub_download(MODEL, MODEL_FILE)
        self.path = path
        self.sha256 = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(path, sess_options=options)
        self.features = WhisperFeatureExtractor(chunk_length=8)

    def score(self, audio, decision_s):
        end = min(len(audio), max(0, int(round(decision_s * SAMPLE_RATE))))
        clip = audio[:end]
        limit = 8 * SAMPLE_RATE
        if len(clip) > limit:
            clip = clip[-limit:]
        elif len(clip) < limit:
            clip = np.pad(clip, (limit - len(clip), 0), mode='constant')
        inputs = self.features(
            clip, sampling_rate=SAMPLE_RATE, return_tensors='np',
            padding='max_length', max_length=limit, truncation=True,
            do_normalize=True)
        features = inputs.input_features.squeeze(0).astype(np.float32)[None]
        output = self.session.run(None, {'input_features': features})[0]
        return float(output[0][0])


def _spans(row):
    duration = float(row['duration'])
    spans = []
    for span in row['silence_spans']:
        start, end = float(span['start']), float(span['end'])
        if end - start <= 0:
            continue
        spans.append(dict(start=start, end=end,
                          final=end >= duration - 0.05))
    return spans


def _auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks for ties so the statistic is threshold-free and stable.
    for value in np.unique(scores):
        tied = np.flatnonzero(scores == value)
        if len(tied) > 1:
            ranks[tied] = ranks[tied].mean()
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def _threshold_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predicted = scores >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum((~predicted) & (labels == 1)))
    tn = int(np.sum((~predicted) & (labels == 0)))
    return dict(threshold=round(float(threshold), 3),
                false_cutoff=round(fp / max(1, fp + tn), 4),
                eot_recall=round(tp / max(1, tp + fn), 4),
                precision=round(tp / max(1, tp + fp), 4),
                tp=tp, fp=fp, fn=fn, tn=tn)


def _sweep(labels, scores):
    rows = [_threshold_metrics(labels, scores, t)
            for t in np.linspace(0.05, 0.95, 19)]
    return rows


def _operating_point(rows, max_false_cutoff):
    feasible = [row for row in rows if row['false_cutoff'] <= max_false_cutoff]
    return max(feasible, key=lambda row: row['eot_recall']) if feasible else None


def _evaluate_budget(rows, model, budget_ms):
    labels, scores, cases = [], [], []
    for index, row in rows.iterrows():
        audio = _decode(row['audio'])
        for span in _spans(row):
            if span['end'] - span['start'] < budget_ms / 1000.0:
                continue
            decision = span['start'] + budget_ms / 1000.0
            score = model.score(audio, decision)
            labels.append(1 if span['final'] else 0)
            scores.append(score)
            cases.append(dict(row=index, id=str(row['id']),
                              start=round(span['start'], 3),
                              end=round(span['end'], 3),
                              final=bool(span['final']),
                              decision_s=round(decision, 3),
                              score=round(score, 6)))
    summary = _threshold_metrics(labels, scores, THRESHOLD) if labels else None
    return dict(budget_ms=budget_ms, cases=len(labels),
                mid_pauses=sum(1 for value in labels if value == 0),
                final_pauses=sum(1 for value in labels if value == 1),
                auc=_auc(labels, scores) if labels else None,
                at_0_5=summary,
                sweep=_sweep(labels, scores) if labels else [],
                operating_points={
                    'max_5pct_false_cutoff': _operating_point(
                        _sweep(labels, scores), .05) if labels else None,
                    'max_10pct_false_cutoff': _operating_point(
                        _sweep(labels, scores), .10) if labels else None,
                },
                observations=cases)


def _fixed_endpoint(rows, endpoint_ms):
    labels, decisions = [], []
    for _, row in rows.iterrows():
        for span in _spans(row):
            labels.append(1 if span['final'] else 0)
            decisions.append((span['end'] - span['start']) >= endpoint_ms / 1000.0)
    return _threshold_metrics(labels, [1.0 if value else 0.0 for value in decisions],
                              .5)


def _temporal_policy(rows, model, threshold, step_ms=200, max_wait_ms=1000,
                     start_ms=300):
    """Re-query one pause until it ends or the wait budget expires."""
    labels, detected, latencies, scores_by_query = [], [], [], []
    for _, row in rows.iterrows():
        audio = _decode(row['audio'])
        for span in _spans(row):
            labels.append(1 if span['final'] else 0)
            duration_ms = (span['end'] - span['start']) * 1000.0
            queries = []
            commit_ms = None
            for wait in range(start_ms, int(min(duration_ms, max_wait_ms)) + 1, step_ms):
                score = model.score(audio, span['start'] + wait / 1000.0)
                queries.append(round(score, 6))
                if score >= threshold:
                    commit_ms = wait
                    break
            detected.append(commit_ms is not None)
            latencies.append(commit_ms)
            scores_by_query.append(queries)
    metrics = _threshold_metrics(labels, [1.0 if value else 0.0 for value in detected], .5)
    final_latencies = [lat for lat, label in zip(latencies, labels)
                       if label and lat is not None]
    return dict(threshold=round(float(threshold), 3), step_ms=step_ms,
                max_wait_ms=max_wait_ms, at_0_5=metrics,
                median_final_latency_ms=(round(float(np.median(final_latencies)), 1)
                                         if final_latencies else None),
                final_detected=len(final_latencies),
                final_total=sum(labels),
                observations=dict(labels=labels, detected=detected,
                                  latencies_ms=latencies, queries=scores_by_query))


def _hybrid_policy(rows, model, threshold, timeout_ms, step_ms=200,
                   start_ms=300, max_query_ms=900):
    """Re-query early, then fall back to a bounded silence timeout.

    Final pauses are treated as continuing silence, so a final pause with no
    confident early query commits at the timeout. A mid-turn pause commits at
    the timeout only if it actually lasts that long.
    """
    labels, detected, latencies, reasons = [], [], [], []
    for _, row in rows.iterrows():
        audio = _decode(row['audio'])
        for span in _spans(row):
            label = 1 if span['final'] else 0
            labels.append(label)
            duration_ms = (span['end'] - span['start']) * 1000.0
            commit_ms = None
            reason = None
            for wait in range(start_ms, min(int(duration_ms), max_query_ms) + 1, step_ms):
                score = model.score(audio, span['start'] + wait / 1000.0)
                if score >= threshold:
                    commit_ms, reason = wait, 'score'
                    break
            if commit_ms is None:
                if label == 1:
                    commit_ms, reason = timeout_ms, 'final-timeout'
                elif duration_ms >= timeout_ms:
                    commit_ms, reason = timeout_ms, 'mid-timeout'
            detected.append(commit_ms is not None)
            latencies.append(commit_ms)
            reasons.append(reason)
    metrics = _threshold_metrics(labels, [1.0 if value else 0.0 for value in detected], .5)
    final_latencies = [lat for lat, label in zip(latencies, labels)
                       if label and lat is not None]
    return dict(timeout_ms=timeout_ms, threshold=round(float(threshold), 3),
                at_0_5=metrics,
                median_final_latency_ms=(round(float(np.median(final_latencies)), 1)
                                         if final_latencies else None),
                final_detected=len(final_latencies),
                final_total=sum(labels),
                reasons={reason: reasons.count(reason) for reason in set(reasons) if reason})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=400)
    parser.add_argument('--out', default=str(ROOT / 'outputs' / 'eot_bench.json'))
    args = parser.parse_args()
    rows, dataset_path = _load_rows(args.limit)
    model = SmartTurn()
    budgets = {str(ms): _evaluate_budget(rows, model, ms) for ms in BUDGETS_MS}
    fixed = _fixed_endpoint(rows, 480)
    temporal = _temporal_policy(rows, model, THRESHOLD)
    hybrid = {str(ms): _hybrid_policy(rows, model, THRESHOLD, ms)
              for ms in (600, 800, 1000, 1200, 1500, 2000)}
    report = dict(
        schema_version=1,
        generated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        evidence_class='offline public-pause scoring; no microphone or live model',
        dataset=dict(repo=DATASET, file=DATASET_FILE,
                     sha256=hashlib.sha256(pathlib.Path(dataset_path).read_bytes()).hexdigest(),
                     rows=len(rows)),
        model=dict(repo=MODEL, file=MODEL_FILE, sha256=model.sha256),
        sample_rate=SAMPLE_RATE,
        budgets=budgets,
        fixed_480ms=fixed,
        temporal_policy=temporal,
        hybrid_timeouts=hybrid,
        decision=dict(
            current_fixed_480ms=fixed,
            single_query_300ms=budgets['300']['at_0_5'],
            re_query=temporal['at_0_5'],
        ),
        limits=(
            'English validation split only; public conversational pauses, not this '
            'app\'s microphone or room. The 300 ms clip ends with exactly the '
            'budget\'s silence; longer waits include more silence. Smart Turn is '
            'scored offline with Whisper features; this is not a live latency or '
            'device test. A mid-pause crossing is a false cut-off even when the '
            'speaker later continues.'
        ),
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    compact = dict(
        out=str(out), rows=len(rows),
        fixed_480ms=fixed,
        single_query_300ms=budgets['300']['at_0_5'],
        temporal=temporal['at_0_5'],
        temporal_median_final_latency_ms=temporal['median_final_latency_ms'],
        hybrid={ms: dict(false_cutoff=row['at_0_5']['false_cutoff'],
                         eot_recall=row['at_0_5']['eot_recall'],
                         median_final_latency_ms=row['median_final_latency_ms'])
                for ms, row in hybrid.items()},
        auc={ms: budgets[str(ms)]['auc'] for ms in BUDGETS_MS},
    )
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
