"""The completion policy is testable without a model or audio device."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from eot import CompletionPolicy


class FakeScorer:
    def __init__(self, score):
        self.value = score
        self.calls = 0

    def score(self, audio):
        self.calls += 1
        return self.value


def test_confident_score_commits_at_the_first_query():
    scorer = FakeScorer(.9)
    policy = CompletionPolicy(scorer, first_query_ms=300, step_ms=200,
                              timeout_ms=1000)
    assert policy.update([0], 299) is None
    assert policy.update([0], 300) == 'commit'
    assert scorer.calls == 1


def test_uncertain_score_is_requeried_then_times_out():
    scorer = FakeScorer(.1)
    policy = CompletionPolicy(scorer, first_query_ms=300, step_ms=200,
                              timeout_ms=1000)
    assert policy.update([0], 300) == 'hold'
    assert policy.update([0], 350) == 'hold'   # throttled, no model call
    assert policy.update([0], 500) == 'hold'
    assert policy.update([0], 1000) == 'commit'
    assert scorer.calls == 3
    assert policy.last_score == pytest.approx(.1)


def test_speech_resets_the_waiting_state():
    scorer = FakeScorer(.1)
    policy = CompletionPolicy(scorer, first_query_ms=300, step_ms=200,
                              timeout_ms=1000)
    policy.update([0], 300)
    policy.update([0], 500)
    assert policy.update([0], 600, speaking=True) is None
    assert policy.queries == 0
    assert policy.update([0], 100) is None
    assert policy.update([0], 300) == 'hold'
    assert policy.queries == 1


def test_smart_turn_scorer_can_be_constructed_with_injected_components():
    from types import SimpleNamespace
    from eot import SmartTurnScorer

    class Features:
        def __call__(self, audio, **kwargs):
            return SimpleNamespace(input_features=np.zeros((1, 80, 800), np.float32))

    class Session:
        def run(self, names, feeds):
            assert feeds['input_features'].shape == (1, 80, 800)
            return [np.array([[.7]], dtype=np.float32)]

    scorer = SmartTurnScorer(model_path='fake', session=Session(),
                             feature_extractor=Features())
    assert scorer.score(np.zeros(16000, np.float32)) == pytest.approx(.7)


def test_first_query_is_stricter_than_later_queries():
    scorer = FakeScorer(.8)
    policy = CompletionPolicy(scorer, threshold=.7, first_threshold=.9,
                              first_query_ms=300, step_ms=200,
                              timeout_ms=1000)
    assert policy.update([0], 300) == 'hold'   # .8 < .9
    assert policy.update([0], 500) == 'commit'  # .8 >= .7
    assert policy.queries == 2


def test_due_tracks_the_query_schedule_without_calling_the_model():
    scorer = FakeScorer(.1)
    policy = CompletionPolicy(scorer, first_query_ms=300, step_ms=200,
                              timeout_ms=1000)
    assert policy.due(299) is False
    assert policy.due(300) is True
    policy.update([0], 300)
    assert policy.due(350) is False
    assert policy.due(500) is True
    assert scorer.calls == 1


def test_invalid_timing_is_rejected():
    with pytest.raises(ValueError):
        CompletionPolicy(FakeScorer(.5), threshold=1.5)
    with pytest.raises(ValueError):
        CompletionPolicy(FakeScorer(.5), first_query_ms=500, timeout_ms=400)
