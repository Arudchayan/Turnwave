"""Accounting tests for the public-pause EOT diagnostic."""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from synth.eot_bench import (_auc, _fixed_endpoint, _operating_point, _spans,
                             _threshold_metrics)


def test_spans_marks_only_trailing_silence_final():
    row = dict(duration=10.0, silence_spans=[
        {'start': 2.0, 'end': 3.0},
        {'start': 8.0, 'end': 10.0},
    ])
    spans = _spans(row)
    assert [span['final'] for span in spans] == [False, True]


def test_threshold_counts_are_explicit():
    result = _threshold_metrics([0, 0, 1, 1], [.1, .6, .7, .2], .5)
    assert (result['tp'], result['fp'], result['fn'], result['tn']) == (1, 1, 1, 1)
    assert result['false_cutoff'] == pytest.approx(.5)
    assert result['eot_recall'] == pytest.approx(.5)


def test_auc_is_threshold_free_and_handles_ties():
    assert _auc([0, 1], [0, 1]) == pytest.approx(1.0)
    assert _auc([0, 1], [1, 0]) == pytest.approx(0.0)
    assert _auc([0, 1], [.5, .5]) == pytest.approx(.5)


def test_fixed_endpoint_commits_only_after_its_silence_budget():
    rows = pd.DataFrame([
        dict(duration=10.0, silence_spans=[{'start': 2.0, 'end': 2.4},
                                           {'start': 8.0, 'end': 10.0}]),
        dict(duration=10.0, silence_spans=[{'start': 2.0, 'end': 3.0}]),
    ])
    result = _fixed_endpoint(rows, 480)
    # 2.4 s mid-pause is below the endpoint; 3.0 s mid-pause is above it; the
    # final 2.0 s span is above it.
    assert (result['tp'], result['fp'], result['fn'], result['tn']) == (1, 1, 0, 1)


def test_operating_point_respects_the_false_cutoff_budget():
    rows = [
        dict(threshold=.2, false_cutoff=.02, eot_recall=.4),
        dict(threshold=.5, false_cutoff=.04, eot_recall=.7),
        dict(threshold=.8, false_cutoff=.09, eot_recall=.9),
    ]
    assert _operating_point(rows, .05)['threshold'] == .5
    assert _operating_point(rows, .02)['threshold'] == .2
    assert _operating_point(rows, .01) is None
