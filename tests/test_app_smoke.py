"""The offline smoke speaker follows the app's playback callback contract."""
import pathlib
import sys
import threading

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from synth.app_smoke import PacedSpeaker, assess_interruption


def test_paced_speaker_reports_written_samples_and_applies_gain():
    player = PacedSpeaker()
    cancel = threading.Event()
    references, written, started = [], [], []
    def on_start():
        started.append(True)
        player.set_gain(.2)
    player.play_chunks([np.ones(480, np.float32)], cancel, on_start,
                       references.append, written.append)
    assert started == [True]
    np.testing.assert_array_equal(written[0], references[0])
    np.testing.assert_allclose(written[0], .2)
    assert not player.playing


def test_screen_rejects_a_split_or_missing_interruption():
    report = dict(barge_duck_from_onset_s=.15, barge_cancel_after_fixture_s=.5,
                  premature_cancel_s=0, user_turns=2, expected_words_missing=[])
    assert assess_interruption(report)
    for change in [dict(user_turns=3), dict(premature_cancel_s=.2),
                   dict(expected_words_missing=['wait']), dict(barge_duck_from_onset_s=.6)]:
        assert not assess_interruption({**report, **change})
