"""The offline floor replay must keep its own A/B accounting honest."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from floor import HARD, REVERSIBLE
from synth.floor_replay import CASES, replay_case, summarize


def row(case_id, mode):
    case = next(case for case in CASES if case[0] == case_id)
    return replay_case(case, mode)


def test_hard_cancels_a_backchannel_but_reversible_preserves_it():
    hard = row('backchannel-180', HARD)
    reversible = row('backchannel-180', REVERSIBLE)
    assert hard['cancelled'] and hard['false_cancel']
    assert reversible['ducked'] and reversible['resumed']
    assert not reversible['cancelled'] and not reversible['false_cancel']


def test_reversible_still_commits_a_sustained_floor_take():
    hard = row('floor-take-900', HARD)
    reversible = row('floor-take-900', REVERSIBLE)
    assert hard['cancelled'] and not hard['missed_cancel']
    assert reversible['cancelled'] and not reversible['missed_cancel']
    assert reversible['cancel_from_onset_ms'] >= 100


def test_mid_pause_is_reported_as_a_known_limitation_not_hidden():
    case = row('mid-pause-400', REVERSIBLE)
    assert case['false_cancel']
    assert case['limitation'] == 'known EOT limitation'


def test_summary_accepts_the_candidate_on_the_frozen_cases():
    rows = [replay_case(case, mode) for mode in (HARD, REVERSIBLE)
            for case in CASES]
    summary = summarize(rows)
    assert summary['decision'] == 'candidate for real-device A/B'
    assert summary['non_limitation_false_cancels']['reversible'] < \
        summary['non_limitation_false_cancels']['hard']
    assert summary['reversible']['missed_cancels'] == 0
