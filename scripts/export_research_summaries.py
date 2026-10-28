"""Export compact, publishable summaries from local research outputs.

The full outputs contain per-case observations and are intentionally ignored.
This script keeps only aggregate policy results and provenance in research/.
It performs no inference.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
RESEARCH = ROOT / 'research'


def read(name):
    path = OUT / name
    if not path.exists():
        raise SystemExit('Missing local output: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def write(name, value):
    path = RESEARCH / name
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8',
                    newline='\n')
    return path


def floor_summary(source):
    return dict(
        schema_version=source.get('schema_version', 1),
        generated_at=source.get('generated_at'),
        evidence_class=source.get('evidence_class'),
        step_ms=source.get('step_ms'),
        confirm_ms=source.get('confirm_ms'),
        endpoint_ms=source.get('endpoint_ms'),
        source_sha256=source.get('source_sha256'),
        summary=source.get('summary'),
        limits=source.get('limits'),
    )


def eot_summary(source):
    per_turn = None
    per_turn_path = OUT / 'eot_per_turn.json'
    if per_turn_path.exists():
        per_turn = json.loads(per_turn_path.read_text(encoding='utf-8'))
    budgets = {}
    for key, row in source.get('budgets', {}).items():
        budgets[key] = dict(
            cases=row.get('cases'), mid_pauses=row.get('mid_pauses'),
            final_pauses=row.get('final_pauses'), auc=row.get('auc'),
            at_0_5=row.get('at_0_5'),
            operating_points=row.get('operating_points'),
        )
    recomputed_path = OUT / 'eot_recomputed.json'
    recomputed = (json.loads(recomputed_path.read_text(encoding='utf-8'))
                  if recomputed_path.exists() else None)
    schedule_path = OUT / 'eot_schedule_sweep.json'
    schedule = (json.loads(schedule_path.read_text(encoding='utf-8'))
                if schedule_path.exists() else None)
    holdout_path = OUT / 'eot_holdout.json'
    holdout = (json.loads(holdout_path.read_text(encoding='utf-8'))
               if holdout_path.exists() else None)
    selected_schedule = None
    if schedule:
        selected = [row for row in schedule.get('rows', [])
                    if row.get('first_threshold') == .9
                    and row.get('later_threshold') == .7]
        selected_schedule = dict(
            protocol=schedule.get('protocol'), pauses=schedule.get('pauses'),
            rows=selected,
            note='Screened on the same pauses; not held-out validation.')
    if recomputed:
        fixed = recomputed.get('fixed_480ms')
        single = recomputed.get('single_300ms')
        temporal = recomputed.get('temporal_no_timeout')
        hybrid = recomputed.get('hybrid_timeouts')
        protocol = recomputed.get('protocol')
    else:
        fixed = source.get('fixed_480ms')
        single = source.get('budgets', {}).get('300', {}).get('at_0_5')
        temporal = {key: value for key, value in
                    source.get('temporal_policy', {}).items()
                    if key != 'observations'}
        hybrid = {key: dict(timeout_ms=row.get('timeout_ms'),
                            at_0_5=row.get('at_0_5'),
                            median_final_latency_ms=row.get('median_final_latency_ms'))
                  for key, row in source.get('hybrid_timeouts', {}).items()}
        protocol = 'original per-policy denominators'
    return dict(
        schema_version=source.get('schema_version', 1),
        generated_at=source.get('generated_at'),
        evidence_class=source.get('evidence_class'),
        protocol=protocol,
        dataset=source.get('dataset'),
        model=source.get('model'),
        sample_rate=source.get('sample_rate'),
        budgets=budgets,
        fixed_480ms=fixed,
        single_300ms=single,
        temporal_no_timeout=temporal,
        hybrid_timeouts=hybrid,
        two_threshold_schedule=selected_schedule,
        holdout=holdout,
        per_turn_300ms=per_turn,
        decision=source.get('decision'),
        limits=source.get('limits'),
    )


def main():
    paths = [
        write('floor_replay_summary.json', floor_summary(read('floor_replay.json'))),
        write('eot_bench_summary.json', eot_summary(read('eot_bench.json'))),
    ]
    for path in paths:
        print(path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
