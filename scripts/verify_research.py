"""Recompute public research claims from included rows; no model or devices."""
import hashlib
import json
import math
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(name):
    return json.loads((ROOT / 'research' / name).read_text(encoding='utf-8'))


def verify():
    results = load('results.json')
    timing = load('short-turn-cases.json')
    screen = load('candidate-screen-cases.json')
    trows, srows = timing['cases'], screen['cases']
    require(len(trows) == len(srows) == 16, 'Expected 16 observations per study')
    tids, sids = [r['id'] for r in trows], [r['id'] for r in srows]
    require(len(set(tids)) == 16 and set(tids) == set(sids), 'Case IDs differ/duplicate')
    for role in ('complete', 'incomplete_prefix'):
        require(sum(r['authored_role'] == role for r in trows) == 8, 'Timing denominator')
        require(sum(r['authored_role'] == role for r in srows) == 8, 'Screen denominator')
    require(all(math.isfinite(r['extra_wall_ms']) and r['extra_wall_ms'] >= 0 for r in trows), 'Invalid timing')
    complete = sum(r['authored_role'] == 'complete' and r['extra_wall_ms'] <= timing['complete_budget_ms'] for r in trows)
    held = sum(r['authored_role'] == 'incomplete_prefix' and r['extra_wall_ms'] >= timing['unfinished_minimum_hold_ms'] for r in trows)
    require(complete == results['short_turn_streaming']['complete_within_160ms'] == 5, 'Timing complete count')
    require(held == results['short_turn_streaming']['unfinished_held_2000ms'] == 8, 'Timing prefix count')
    for row in srows:
        require(math.isfinite(row['audio_score']) and 0 <= row['audio_score'] <= 1, 'Invalid probability')
        require(row['combined_positive'] == (row['namo_label'] == 1 or row['audio_score'] > .5), 'OR rule mismatch')
    counts = {}
    for method, predicate in [('namo', lambda r: r['namo_label'] == 1), ('combined', lambda r: r['combined_positive'])]:
        counts[method + '_complete_positive'] = sum(r['authored_role'] == 'complete' and predicate(r) for r in srows)
        counts[method + '_unfinished_held'] = sum(r['authored_role'] == 'incomplete_prefix' and not predicate(r) for r in srows)
    require(counts == {k: results['fixed_endpoint_audio_or'][k] for k in counts}, 'Screen summary mismatch')
    capture = results['public_capture_comparison']
    require(capture['completed_cases'] == len(capture['cases']) == 6 and capture['paired_arms'] == 12, 'Capture denominator')
    require(capture['paired_result']['candidate_added_capture_samples'] == 0, 'Capture adoption claim changed')
    examples = load('examples/examples.json')['examples']
    require(len(examples) == 2, 'Example scope changed')
    for example in examples:
        relative = Path(example['audio'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe example path')
        path = ROOT / 'research' / relative
        require(hashlib.sha256(path.read_bytes()).hexdigest() == example['audio_sha256'], 'Example hash mismatch')
        require(example['whole_reply'] and not example['contains_human_recording'], 'Example review flags changed')
        with wave.open(str(path), 'rb') as wav:
            require((wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (24000, 1, 2), 'Example format')
            require(wav.getnframes() == example['frames'], 'Example frame count')
            require(wav.getnframes() / wav.getframerate() == example['duration_s'], 'Example duration')
    return {'timing_cases': len(trows), 'screen_counts': counts, 'capture_pairs': 6, 'whole_reply_examples': len(examples)}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
