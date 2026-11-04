"""Describe per-turn cut-offs on the EOT replay, without re-running a model.

This uses the saved 300 ms observations and the same cached dataset. It is a
descriptive denominator check, not a replacement for the per-pause metric.
"""
import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'eot_bench.json'
DEST = ROOT / 'outputs' / 'eot_per_turn.json'


def main():
    source = json.loads(OUT.read_text(encoding='utf-8'))
    observations = source['budgets']['300']['observations']
    mid_rows = {row['row'] for row in observations if not row['final']}
    single_cut_rows = {row['row'] for row in observations
                       if not row['final'] and row['score'] >= .5}
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(source['dataset']['repo'], source['dataset']['file'],
                           repo_type='dataset')
    frame = pd.read_parquet(path).head(source['dataset']['rows'])
    fixed_cut = 0
    for index, row in frame.iterrows():
        if index not in mid_rows:
            continue
        duration = float(row['duration'])
        pauses = [float(span['end']) - float(span['start'])
                  for span in row['silence_spans']
                  if float(span['end']) < duration - .05
                  and float(span['end']) - float(span['start']) >= .3]
        if any(pause >= .48 for pause in pauses):
            fixed_cut += 1
    result = dict(
        schema_version=1,
        budget_ms=300,
        denominator='turns with at least one mid-turn pause >= 300 ms',
        turns=len(mid_rows),
        fixed_480ms_cut=fixed_cut,
        fixed_480ms_rate=round(fixed_cut / max(1, len(mid_rows)), 4),
        single_smart_turn_cut=len(single_cut_rows),
        single_smart_turn_rate=round(len(single_cut_rows) / max(1, len(mid_rows)), 4),
        note='Descriptive subset; not the per-pause primary metric.',
    )
    DEST.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8',
                    newline='\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
