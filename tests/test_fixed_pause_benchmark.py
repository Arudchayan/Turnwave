from src.synth.fixed_pause_benchmark import evaluate


def test_fixed_presets_count_mid_and_final_pauses():
    rows = [
        {'duration': 4.0, 'silence_spans': [
            {'start': 1.0, 'end': 1.6},
            {'start': 2.0, 'end': 2.9},
            {'start': 3.5, 'end': 4.0}]},
        {'duration': 5.0, 'silence_spans': [
            {'start': 1.0, 'end': 2.2},
            {'start': 3.0, 'end': 5.0}]},
    ]
    result = evaluate(rows)
    assert [result[ms]['mid_cutoffs'] for ms in (480, 800, 1120)] == [3, 2, 1]
    assert [result[ms]['mid_total'] for ms in (480, 800, 1120)] == [3, 3, 3]
    assert [result[ms]['final_reached'] for ms in (480, 800, 1120)] == [2, 1, 1]
    assert all(result[ms]['final_total'] == 2 for ms in (480, 800, 1120))


def test_empty_denominators_do_not_claim_rates():
    result = evaluate([{'duration': 1, 'silence_spans': []}])
    assert all(row['mid_cutoff_rate'] is None and row['final_reached_rate'] is None
               for row in result.values())
