"""Compare fixed pause durations on a local, annotated Parquet file.

This scores annotations, not the app's VAD, playback, or a physical device.
The input is supplied locally; no dataset or model is downloaded.
"""
import argparse
import hashlib
import json
from pathlib import Path

PRESETS = (480, 800, 1120)


def evaluate(rows, presets=PRESETS):
    counts = {ms: {'mid_cutoffs': 0, 'mid_total': 0, 'final_reached': 0, 'final_total': 0}
              for ms in presets}
    for row in rows:
        duration = float(row['duration'])
        for span in row['silence_spans']:
            start, end = float(span['start']), float(span['end'])
            if end <= start:
                continue
            final = end >= duration - .05
            for ms, result in counts.items():
                crossed = end - start >= ms / 1000.0
                if final:
                    result['final_total'] += 1
                    result['final_reached'] += int(crossed)
                else:
                    result['mid_total'] += 1
                    result['mid_cutoffs'] += int(crossed)
    for result in counts.values():
        result['mid_cutoff_rate'] = (round(result['mid_cutoffs'] / result['mid_total'], 4)
                                     if result['mid_total'] else None)
        result['final_reached_rate'] = (round(result['final_reached'] / result['final_total'], 4)
                                        if result['final_total'] else None)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, help='local annotated Parquet file')
    parser.add_argument('--limit', type=int, default=400, help='first N rows (positive integer)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error('--limit must be positive')
    import pandas as pd
    path = Path(args.dataset)
    frame = pd.read_parquet(path, columns=['duration', 'silence_spans']).head(args.limit)
    rows = frame.to_dict('records')
    report = {'scope': 'annotated pauses only; no VAD, room, playback, or model',
              'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'rows': len(rows), 'presets_ms': evaluate(rows),
              'limits': 'First N rows only, per-pause observations; final pauses may be clipped by the dataset.'}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
