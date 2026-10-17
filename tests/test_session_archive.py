import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from src.session_archive import SessionArchive


def result(**kwargs):
    fields = dict(text='A clear reply.', error=None, finish_reason='stop',
                  cancel=threading.Event(), audio_samples=4, ttft=0.25)
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def test_exact_inputs_exclusive_paths_and_requests(tmp_path):
    session = SessionArchive(tmp_path / 'recorded')
    original = b'RIFF exact input bytes including unknown chunks\x00\xff'
    turn = session.begin_turn(7, original, first=True)
    payload = {'messages': [{'content': 'Café\nexact', 'role': 'user'}],
               'nested': {'max_tokens': 512, 'extra': None}}
    turn.record_request(payload)
    assert (turn.folder / 'input.wav').read_bytes() == original
    assert load(turn.folder / 'request.json') == payload
    with pytest.raises(FileExistsError):
        turn.record_request({'changed': True})
    with pytest.raises(FileExistsError):
        session.begin_turn(7, b'different', False)
    with pytest.raises(FileExistsError):
        SessionArchive(session.folder)
    with pytest.raises(ValueError):
        session.begin_turn('../escape', b'', False)
    assert not (tmp_path / 'escape').exists()
    assert load(turn.folder / 'request.json') == payload


@pytest.mark.parametrize('playback_first', [False, True])
def test_readable_incremental_audio_finalization_and_hashes(tmp_path, playback_first):
    turn = SessionArchive(tmp_path / 'recorded').begin_turn('one', b'', False)
    all_samples = np.array([-2, -1, -.25, 0, .25, 1, 2], dtype=np.float32)
    consumed = 0
    for chunk in [all_samples[:2], all_samples[2:3], all_samples[3:]]:
        turn.write_audio(chunk)
        consumed += len(chunk)
        # Reader can open it while generation remains active.
        with wave.open(str(turn.folder / 'generated.wav'), 'rb') as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (24000, 1, 2)
            assert wav.getnframes() == consumed
            actual = np.frombuffer(wav.readframes(consumed), dtype='<i2')
        expected = np.clip(np.rint(np.clip(all_samples[:consumed], -1, 1) * 32768), -32768, 32767).astype('<i2')
        np.testing.assert_array_equal(actual, expected)
        assert (turn.folder / 'generated.f32').read_bytes() == all_samples[:consumed].astype('<f4').tobytes()
    assert not (turn.folder / 'manifest.json').exists()
    assert not turn.status()['checkpoint_safe']
    if playback_first:
        turn.finish_playback(3, True)
        assert not turn.status()['checkpoint_safe']
        turn.finish_generation(result(audio_samples=7))
    else:
        turn.finish_generation(result(audio_samples=7))
        assert not turn.status()['checkpoint_safe']
        turn.finish_playback(3, True)
    status = load(turn.folder / 'status.json')
    assert status['state'] == 'finalized'
    assert status['checkpoint_safe']
    assert status['generated_samples'] == 7
    assert status['played_samples'] == 3
    assert status['playback']['cancelled']
    roster = load(turn.folder / 'manifest.json')['files']
    assert set(roster) == {'input.wav', 'turn.json', 'generated.wav', 'generated.f32', 'generation.json', 'playback.json'}
    for name, entry in roster.items():
        data = (turn.folder / name).read_bytes()
        assert entry == {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    with pytest.raises(RuntimeError):
        turn.write_audio(np.ones(1))
    with pytest.raises(RuntimeError):
        turn.finish_generation(result())
    with pytest.raises(RuntimeError):
        turn.finish_playback(7, False)
    assert not (turn.folder / 'READY').exists()


def test_cancelled_partial_generation_and_playback_error_preserved(tmp_path):
    turn = SessionArchive(tmp_path / 'recorded').begin_turn('partial', b'input', True)
    turn.write_audio(np.array([0.1, -0.2], dtype=np.float32))
    cancelled = threading.Event()
    cancelled.set()
    turn.finish_generation(result(text='Partial', cancel=cancelled, error='transport failed'))
    turn.finish_playback(1, True, error='speaker disconnected')
    status = turn.status()
    assert status['generation']['text'] == 'Partial'
    assert status['generation']['cancelled']
    assert status['generation']['error'] == 'transport failed'
    assert status['playback']['error'] == 'speaker disconnected'
    assert status['checkpoint_safe']  # Finished failed evidence, not a successful reply.


def test_empty_test_turn_missing_audio_is_explicit(tmp_path):
    turn = SessionArchive(tmp_path / 'recorded').begin_turn('test', b'', False, test=True)
    turn.write_audio(np.array([], dtype=np.float32))
    turn.finish_generation(result(text='', error='no audio', audio_samples=0))
    turn.finish_playback(0, False)
    assert load(turn.folder / 'turn.json')['test']
    assert not (turn.folder / 'generated.wav').exists()
    assert not turn.status()['generation']['audio_present']
    assert turn.status()['generation']['error'] == 'no audio'


def test_invalid_audio_cannot_damage_existing_partial(tmp_path):
    turn = SessionArchive(tmp_path / 'recorded').begin_turn('one', b'', True)
    turn.write_audio(np.array([0.2], dtype=np.float32))
    original = (turn.folder / 'generated.wav').read_bytes()
    for bad in [np.array([np.nan]), np.array([[0.5]])]:
        with pytest.raises(ValueError):
            turn.write_audio(bad)
    assert (turn.folder / 'generated.wav').read_bytes() == original
    with pytest.raises(ValueError):
        turn.finish_playback(-1, False)
    assert not (turn.folder / 'playback.json').exists()
    turn.finish_generation(result())
    turn.finish_playback(1, False)


def test_process_exit_preserves_readable_partial_without_claiming_final(tmp_path):
    folder = tmp_path / 'crashed'
    script = '''
import os, sys
import numpy as np
from src.session_archive import SessionArchive
turn = SessionArchive(sys.argv[1]).begin_turn('partial', b'original', True)
turn.write_audio(np.array([0.25, -0.25, 0.5], dtype=np.float32))
os._exit(17)
'''
    run = subprocess.run([sys.executable, '-c', script, str(folder)],
                         cwd=Path(__file__).resolve().parents[1], capture_output=True)
    assert run.returncode == 17, run.stderr.decode(errors='replace')
    turn_folder = folder / 'turn-partial'
    with wave.open(str(turn_folder / 'generated.wav'), 'rb') as wav:
        assert wav.getnframes() == 3
        assert len(wav.readframes(3)) == 6
    assert not load(turn_folder / 'status.json')['checkpoint_safe']
    assert not (turn_folder / 'generation.json').exists()
    assert not (turn_folder / 'manifest.json').exists()
    np.testing.assert_array_equal(np.fromfile(turn_folder / 'generated.f32', dtype='<f4'),
                                  np.array([0.25, -0.25, 0.5], dtype=np.float32))


def test_native_pcm16_roundtrip_and_exact_float_stream(tmp_path):
    turn = SessionArchive(tmp_path / 'recorded').begin_turn('one', b'', True)
    native = np.array([-32768, -1, 0, 1, 32767], dtype='<i2')
    decoded = native.astype(np.float32) / 32768
    arbitrary = np.array([0.1234567, -0.9876543, 1.0001], dtype=np.float32)
    turn.write_audio(decoded[:2])
    turn.write_audio(decoded[2:])
    turn.write_audio(arbitrary)
    assert (turn.folder / 'generated.f32').read_bytes() == decoded.astype('<f4').tobytes() + arbitrary.astype('<f4').tobytes()
    with wave.open(str(turn.folder / 'generated.wav'), 'rb') as wav:
        np.testing.assert_array_equal(np.frombuffer(wav.readframes(5), dtype='<i2'), native)
    turn.finish_generation(result(audio_samples=8))
    turn.finish_playback(8, False)
    fmt = load(turn.folder / 'generation.json')['generated_audio_format']
    assert fmt == {'authoritative': 'generated.f32', 'dtype': '<f4', 'channels': 1,
                   'sample_rate': 24000, 'samples': 8, 'preview': 'generated.wav'}
