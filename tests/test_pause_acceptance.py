"""Acceptance checks protect the actual failure and per-case regressions."""
import sys
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from synth.pause_acceptance import assess, duration_candidate_source, load_frontend, run, validate_roster


def result(starts=(.1,), ends=(1.5,)):
    return dict(starts_s=list(starts), ends_s=list(ends))


def test_same_fragmentation_cannot_pass_on_unchanged_totals():
    case = dict(kind='pause', protected_until_s=3)
    assert assess(case, result(), result())['reasons'] == ['premature_endpoint_remains']
    assert assess(case, result(), result(ends=(3.5,)))['passed']


def test_pause_requires_reproduced_baseline_and_candidate_completion():
    case = dict(kind='pause', protected_until_s=3)
    assert 'baseline_failure_not_reproduced' in assess(case, result(ends=(4,)), result(ends=(4,)))['reasons']
    assert 'candidate_never_completes' in assess(case, result(), result(ends=()))['reasons']


def test_short_command_latency_and_lost_clip_are_rejected_individually():
    case = dict(kind='short')
    assert assess(case, result(), result(ends=(1.532,)))['passed']
    assert not assess(case, result(), result(ends=(2.14,)))['passed']
    assert not assess(case, result(), result(ends=()))['passed']


def test_quiet_control_cannot_hide_baseline_or_candidate_false_onset():
    case = dict(kind='quiet')
    quiet = result(starts=(), ends=())
    assert assess(case, quiet, quiet)['passed']
    assert not assess(case, quiet, result())['passed']
    assert not assess(case, result(), quiet)['passed']


def test_duration_candidate_fails_closed_when_source_changes():
    with pytest.raises(ValueError, match='anchor changed'):
        duration_candidate_source('class UnknownFrontend: pass\n')


def test_existing_experiment_is_never_modified_on_failed_run(tmp_path):
    existing = tmp_path / 'experiment'
    existing.mkdir()
    sentinel = existing / 'failure.txt'
    sentinel.write_text('Original failed outcome')
    with pytest.raises(FileExistsError):
        run(tmp_path / 'missing.json', existing, tmp_path / 'missing.py')
    assert sentinel.read_text() == 'Original failed outcome'
    assert list(existing.iterdir()) == [sentinel]


def test_generated_duration_rule_restores_short_endpoint_after_long_turn(tmp_path):
    import numpy as np

    source = Path(__file__).resolve().parents[1] / 'src/duplex.py'
    candidate = tmp_path / 'candidate.py'
    candidate.write_text(duration_candidate_source(source.read_text()), encoding='utf-8')
    front = load_frontend(candidate, 'test_pause_candidate')(aec=False, holdoff_s=0)
    front._prob = lambda chunk: float(np.mean(chunk) > .02)
    for speech_s, expected_delay in [(3.2, 1.12), (.7, .48)]:
        for _ in range(round(speech_s * 100)):
            front.update(np.full(160, .1, np.float32))
        begin = front._sample_clock
        for _ in range(200):
            _, _, ended, _ = front.update(np.zeros(160, np.float32))
            if ended:
                break
        assert ended
        assert abs((front._sample_clock - begin) / 16000 - expected_delay) <= .032 + 1e-9
        assert front.end_chunks == 15
        assert front._utterance_start_clock is None


@pytest.mark.parametrize('cases', [
    [], None, {},
    [dict(id='../escape', kind='quiet')],
    [dict(id='..\\escape', kind='quiet')],
    [dict(id='C:escape', kind='quiet')],
    [dict(id='/absolute', kind='quiet')],
    [dict(id='', kind='quiet')],
    [dict(id=None, kind='quiet')],
    [dict(id=123, kind='quiet')],
    [dict(id='same', kind='quiet'), dict(id='same', kind='quiet')],
    [dict(id='Same', kind='quiet'), dict(id='same', kind='quiet')],
    [dict(id='safe', kind='unknown')],
    [dict(id='safe')],
    ['not-a-case'],
])
def test_invalid_roster_is_rejected_before_source_load_or_output(tmp_path, cases):
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps(dict(cases=cases, scope='unit control')))
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='case|Case'):
        run(protocol, output, tmp_path / 'source-must-not-be-read.py')
    assert not output.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ['protocol.json']


def test_unknown_kind_is_rejected_by_direct_assessment():
    with pytest.raises(ValueError, match='case kind'):
        assess(dict(kind='unknown'), result(), result())


def test_safe_roster_accepts_all_supported_case_kinds():
    validate_roster(dict(cases=[dict(id='pause-1', kind='pause'),
                                dict(id='short_2', kind='short'),
                                dict(id='quiet3', kind='quiet')]))
