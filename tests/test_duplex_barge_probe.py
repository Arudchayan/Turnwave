"""Contract checks for capture-loop benchmark inputs and frame synchronization."""
import threading
import numpy as np
import pytest
from src.synth.duplex_barge_probe import StepQueue, build_tracks


def test_synthetic_tracks_keep_silence_and_pause_boundaries():
    far = np.ones(48000, dtype=np.float32)
    near = np.ones(16000, dtype=np.float32)
    playback, voice = build_tracks('mid_pause', far, near)
    assert not np.any(playback[:8000])
    assert np.count_nonzero(voice) == 12800
    assert np.all(voice[24000:30400] == 1)
    assert not np.any(voice[30400:38400])
    assert np.all(voice[38400:44800] == 1)
    _, echo_only = build_tracks('echo_only', far, near)
    assert not np.any(echo_only)
    _, utterance = build_tracks('utterance', far, near[:9600])
    assert np.count_nonzero(utterance) == 9600


def test_frame_acknowledgment_waits_for_next_read():
    q = StepQueue()
    q.put('frame')
    assert q.get() == 'frame'
    with pytest.raises(TimeoutError):
        with q.finished:
            if not q.finished.wait_for(lambda: q.completed >= 1, timeout=.01):
                raise TimeoutError()
    done = threading.Event()
    def consume():
        q.get()
        done.set()
    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    q.wait_completed(1)
    q.put('end')
    assert done.wait(1)
    thread.join(timeout=1)
