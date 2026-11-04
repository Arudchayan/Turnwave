"""Reversible floor decisions and the playback gain envelope they rely on."""
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import audio_io
from audio_io import Player
from duplex import DuplexFrontend
from floor import (BACKCHANNEL, CANDIDATE, FLOOR, HARD, IDLE, REVERSIBLE,
                   FloorPolicy)


def frames(policy, count, *, t0=0.0, step=.02, speaking=True, ended=False,
           assistant_active=True, audible=True, block_s=.02):
    """Feed ``count`` blocks after an explicit onset and return the actions."""
    actions = [policy.update(t0, speaking=True, started=True,
                             assistant_active=assistant_active,
                             audible=audible, block_s=block_s)]
    now = t0
    for _ in range(count - 1):
        now += step
        actions.append(policy.update(now, speaking=speaking, ended=False,
                                     assistant_active=assistant_active,
                                     audible=audible, block_s=block_s))
    if ended:
        now += step
        actions.append(policy.update(now, speaking=False, ended=True,
                                     assistant_active=assistant_active,
                                     audible=audible, block_s=0.0))
    return actions


def test_hard_mode_cancels_immediately_on_onset():
    policy = FloorPolicy(mode=HARD)
    action = policy.update(0.0, speaking=True, started=True,
                           assistant_active=True, audible=True)
    assert action.cancel and not action.duck
    assert action.state == FLOOR


def test_reversible_ducks_and_preserves_a_short_backchannel():
    policy = FloorPolicy(mode=REVERSIBLE, backchannel_max_ms=250,
                         floor_confirm_ms=300)
    actions = frames(policy, 10, ended=True)  # 200 ms, then silence
    assert actions[0].duck and not actions[0].cancel
    assert actions[0].state == CANDIDATE
    assert actions[-1].resume and not actions[-1].cancel
    assert actions[-1].state == BACKCHANNEL
    assert policy.state == IDLE


def test_reversible_commits_sustained_speech_before_endpoint():
    policy = FloorPolicy(mode=REVERSIBLE, backchannel_max_ms=250,
                         floor_confirm_ms=300)
    actions = frames(policy, 16)  # 320 ms of continuous speech
    assert any(action.cancel for action in actions)
    committed = next(action for action in actions if action.cancel)
    assert committed.reason == 'sustained-speech'
    assert committed.speech_ms >= 300
    assert policy.state == FLOOR


def test_reversible_commits_a_short_command_at_endpoint():
    policy = FloorPolicy(mode=REVERSIBLE, backchannel_max_ms=250,
                         floor_confirm_ms=300)
    actions = frames(policy, 13, ended=True)  # 260 ms, then silence
    assert not any(action.cancel for action in actions[:-1])
    assert actions[-1].cancel
    assert actions[-1].reason == 'ended-speech'


def test_reversible_cancels_when_assistant_is_not_audible():
    policy = FloorPolicy(mode=REVERSIBLE)
    action = policy.update(0.0, speaking=True, started=True,
                           assistant_active=True, audible=False)
    assert action.cancel and not action.duck
    assert action.reason == 'not-audible'


def test_assistant_finishing_promotes_the_candidate_without_cancelling():
    policy = FloorPolicy(mode=REVERSIBLE)
    first = policy.update(0.0, speaking=True, started=True,
                          assistant_active=True, audible=True)
    assert first.duck
    promoted = policy.update(.1, speaking=True, assistant_active=False,
                             audible=False)
    assert promoted.promote and not promoted.cancel
    assert policy.state == IDLE


def test_reset_discards_candidate_state():
    policy = FloorPolicy(mode=REVERSIBLE)
    policy.update(0.0, speaking=True, started=True,
                  assistant_active=True, audible=True)
    policy.reset()
    assert policy.state == IDLE
    assert policy.snapshot()['speech_ms'] == 0


def test_frontend_reports_speech_inside_the_confirmation_window():
    front = DuplexFrontend(aec=False, holdoff_s=0)
    front._prob = lambda chunk: 1.0
    started = False
    for _ in range(4):
        _, started, _, _ = front.update(np.ones(512, np.float32) * .1)
    assert started
    assert 96 <= front.onset_speech_ms <= 160


def test_real_frontend_and_policy_preserve_a_short_onset():
    front = DuplexFrontend(aec=False, holdoff_s=0)
    probability = [1.0]
    front._prob = lambda chunk: probability[0]
    policy = FloorPolicy(mode=REVERSIBLE)
    actions = []
    now = 0.0
    started = False
    for _ in range(6):  # four blocks confirm onset, two continue it
        speaking, started, ended, clean = front.update(np.ones(512, np.float32) * .1)
        now += len(clean) / 16000
        actions.append(policy.update(now, speaking=front.hot, started=started,
                                     ended=ended, block_s=len(clean) / 16000,
                                     onset_ms=getattr(front, 'onset_speech_ms', None),
                                     assistant_active=True, audible=True))
    probability[0] = 0.0
    for _ in range(20):
        speaking, started, ended, clean = front.update(np.zeros(512, np.float32))
        now += len(clean) / 16000
        actions.append(policy.update(now, speaking=front.hot, started=started,
                                     ended=ended, block_s=len(clean) / 16000,
                                     assistant_active=True, audible=True))
        if ended:
            break
    assert actions[0].duck or any(action.duck for action in actions)
    assert not any(action.cancel for action in actions)
    assert any(action.resume for action in actions)
    assert policy.state == IDLE


def test_invalid_policy_parameters_are_rejected():
    with pytest.raises(ValueError):
        FloorPolicy(mode='sometimes')
    with pytest.raises(ValueError):
        FloorPolicy(mode=REVERSIBLE, backchannel_max_ms=400,
                    floor_confirm_ms=300)
    with pytest.raises(ValueError):
        FloorPolicy(mode=REVERSIBLE, floor_confirm_ms=0)


def bare_player(rate=24000):
    player = Player.__new__(Player)
    player.rate = rate
    player.native_rate = rate
    player.native_channels = 1
    player._lock = threading.Lock()
    player._gain = 1.0
    player._gain_target = 1.0
    player._gain_step = 0.0
    return player


def test_gain_ramp_reaches_the_requested_level_without_a_step():
    player = bare_player()
    player.set_gain(.25, ramp_ms=20)  # 480 samples at 24 kHz
    rendered = player._render_gain(np.ones(480, np.float32))
    assert rendered[0] > .99
    assert rendered[-1] == pytest.approx(.25, abs=1e-6)
    assert player._render_gain(np.ones(480, np.float32))[-1] == pytest.approx(.25)


def test_gain_is_clamped_and_can_be_restored():
    player = bare_player()
    player.set_gain(2.0, ramp_ms=0)
    assert player._render_gain(np.ones(1, np.float32))[0] == pytest.approx(1.0)
    player.set_gain(-1.0, ramp_ms=0)
    assert player._render_gain(np.ones(1, np.float32))[0] == pytest.approx(0.0)
    player.set_gain(1.0, ramp_ms=0)
    assert player._render_gain(np.ones(1, np.float32))[0] == pytest.approx(1.0)


def test_playback_duck_is_applied_to_device_and_echo_reference(monkeypatch):
    written, references = [], []

    class Output:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def write(self, audio):
            written.append(audio.copy())

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr('audio_io.sd.OutputStream', Output)
    player = Player()
    player.native_rate = 24000
    player.native_channels = 1

    def chunks():
        yield np.ones(480, np.float32)
        player.set_gain(.25, ramp_ms=20)
        yield np.ones(480, np.float32)

    player.play_chunks(chunks(), on_play=lambda audio: references.append(audio.copy()))
    assert written[0][0] == pytest.approx(1.0)
    assert written[1][-1] == pytest.approx(.25, abs=1e-6)
    np.testing.assert_allclose(np.concatenate(references), np.concatenate(written))

@pytest.mark.parametrize('hot_frames, expected_reason', [
    (6, 'backchannel'), (7, 'backchannel'), (8, 'ended-speech'),
    (9, 'ended-speech'), (10, 'sustained-speech'),
])
@pytest.mark.parametrize('block_samples', [160, 320, 480, 640, 960])
def test_frame_evidence_is_independent_of_capture_block_size(hot_frames, expected_reason, block_samples):
    # The real frontend consumes 32 ms VAD frames, while capture delivers
    # blocks of another size. Use deterministic classifications to isolate
    # counting; these shapes are not labelled natural speech.
    front = DuplexFrontend(aec=False, holdoff_s=0)
    front._prob = lambda chunk: float(np.mean(chunk) > .5)
    policy = FloorPolicy(mode=REVERSIBLE)
    samples = np.concatenate((np.ones(512 * hot_frames, dtype=np.float32),
                              np.zeros(16000, dtype=np.float32)))
    for offset in range(0, len(samples), block_samples):
        _, started, ended, clean = front.update(samples[offset:offset + block_samples])
        action = policy.update(offset / 16000, speaking=front.hot, started=started,
                               ended=ended, onset_ms=front.onset_speech_ms,
                               evidence_ms=front.speech_evidence_ms,
                               block_s=len(clean) / 16000)
        if action.cancel or action.resume:
            break
    assert action.reason == expected_reason
    assert action.speech_ms == pytest.approx(hot_frames * 32)
