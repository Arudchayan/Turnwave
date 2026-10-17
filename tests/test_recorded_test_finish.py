import json
import hashlib
from pathlib import Path
import wave
from types import SimpleNamespace

import pytest

from src.synth.finish_recorded_test import finish as _finish, _mute
from src.synth.checkpoint_experiment import checkpoint, verify_checkpoint
from src.session_archive import SessionArchive


def finish(*args, **kwargs):
    """Legacy finalizer fixtures explicitly simulate the new app handshake.

    No injected-state test may fall through to the real running app's HTTP API.
    Dedicated fence tests exercise refused/unsupported/real fake-app seals.
    """
    if 'read_state' in kwargs and 'seal_archive' not in kwargs:
        read_state = kwargs['read_state']
        sealed = False
        def seal(expected):
            nonlocal sealed
            sealed = True
            return {'ok': True, 'archive_session': expected, 'archive_sealed': True}
        def state():
            value = read_state()
            return dict(value, archive_sealed=True) if sealed else value
        kwargs.update(read_state=state, seal_archive=seal)
    return _finish(*args, **kwargs)


class Clock:
    def __init__(self):
        self.now = 0
        self.callbacks = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        for callback in self.callbacks:
            callback()


def session(tmp_path):
    folder = tmp_path / 'experiment'
    audio = folder / 'audio'
    audio.mkdir(parents=True)
    devices = {name: dict(maxInputChannels=1, defaultSampleRate=24000)
               for name in ('microphone', 'speaker')}
    (audio / 'audio-session.json').write_text(json.dumps(dict(devices=devices)))
    return folder, audio


def close_recording(audio, truncate=False):
    for track in ('microphone', 'speaker'):
        with wave.open(str(audio / (track + '.wav')), 'wb') as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(24000)
            writer.writeframes(b'\0' * 40)
    if truncate:
        path = audio / 'speaker.wav'
        path.write_bytes(path.read_bytes()[:-2])
    (audio / 'audio-finished.json').write_text(json.dumps(dict(frames={'microphone': 20, 'speaker': 20})))


def idle(**kwargs):
    return dict(status='listening', assistant_partial='', mic_on=False, active_turn=False,
                recording_safe=True, pending_turns=0, playback_active=False, archive_pending=0, **kwargs)


def test_timeout_keeps_recorder_running_and_retains_unique_attempts(tmp_path):
    folder, audio = session(tmp_path)
    for _ in range(2):
        clock = Clock()
        with pytest.raises(TimeoutError, match='recorder was not stopped'):
            finish(folder, timeout_s=.5, read_state=lambda: dict(idle(), active_turn=True),
                   clock=clock.time, sleep=clock.sleep,
                   make_checkpoint=lambda _: pytest.fail('Premature checkpoint'))
    assert not (audio / 'STOP_AUDIO').exists()
    attempts = list(folder.glob('finalization-*'))
    assert len(attempts) == 2
    assert all(json.loads((p / 'status.json').read_text())['phase'] == 'error' for p in attempts)


def test_mute_then_drain_and_checkpoint_closed_files(tmp_path):
    folder, audio = session(tmp_path)
    clock = Clock()
    state = dict(idle(), mic_on=True, active_turn=True, status='speaking', assistant_partial='reply')
    calls = []

    def mute():
        calls.append('mute')
        state['mic_on'] = False

    def tick():
        if clock.now >= .4:
            state.update(idle())
        if (audio / 'STOP_AUDIO').exists():
            close_recording(audio)

    clock.callbacks.append(tick)
    def save(source):
        assert clock.now >= .4
        assert (audio / 'audio-finished.json').exists()
        return checkpoint(source, tmp_path / 'backups')

    result = finish(folder, read_state=lambda: state.copy(), mute_mic=mute,
                    clock=clock.time, sleep=clock.sleep, make_checkpoint=save)
    assert calls == ['mute']
    assert result['phase'] == 'verified-checkpoint'
    verify_checkpoint(result['checkpoint'])
    assert (Path(result['checkpoint']) / 'finalization-receipt.json').exists()
    assert json.loads((Path(result['attempt']) / 'status.json').read_text())['phase'] == 'recording-verified'


def test_unknown_active_state_and_unavailable_app_do_not_stop_audio(tmp_path):
    folder, audio = session(tmp_path)
    clock = Clock()
    state = idle()
    del state['active_turn']
    with pytest.raises(TimeoutError):
        finish(folder, timeout_s=.3, read_state=lambda: state, clock=clock.time, sleep=clock.sleep)
    assert not (audio / 'STOP_AUDIO').exists()
    def unavailable():
        raise OSError('App unavailable')
    with pytest.raises(OSError):
        finish(folder, read_state=unavailable)
    assert not (audio / 'STOP_AUDIO').exists()


def test_truncated_recording_is_not_checkpointed(tmp_path):
    folder, audio = session(tmp_path)
    close_recording(audio, truncate=True)
    with pytest.raises(ValueError, match='truncated'):
        finish(folder, read_state=idle, make_checkpoint=lambda _: pytest.fail('Corrupt audio checkpointed'))


def test_waits_for_explicit_observer_finish(tmp_path):
    folder, audio = session(tmp_path)
    close_recording(audio)
    (folder / 'experiment.json').write_text(json.dumps(dict(watch_minutes=10)))
    clock = Clock()
    def tick():
        if (folder / 'STOP').exists():
            (folder / 'finished.json').write_text('{}')
    clock.callbacks.append(tick)
    result = finish(folder, read_state=idle, clock=clock.time, sleep=clock.sleep,
                    make_checkpoint=lambda p: checkpoint(p, tmp_path / 'backups'))
    assert (folder / 'finished.json').exists()
    assert result['phase'] == 'verified-checkpoint'


def test_exact_folder_and_local_address_required_before_writes(tmp_path):
    folder, audio = session(tmp_path)
    for selected in (tmp_path, tmp_path / 'new'):
        with pytest.raises((ValueError, FileNotFoundError)):
            finish(selected)
    with pytest.raises(ValueError):
        finish(folder, app_base='https://example.org')
    assert not list(folder.glob('finalization-*'))
    assert not (audio / 'STOP_AUDIO').exists()


@pytest.mark.parametrize('field,value', [('recording_safe', False), ('pending_turns', 1),
                                        ('playback_active', True), ('archive_pending', 1)])
def test_background_producer_playback_or_archive_prevents_stop(tmp_path, field, value):
    folder, audio = session(tmp_path)
    clock = Clock()
    state = idle()
    state[field] = value
    with pytest.raises(TimeoutError):
        finish(folder, timeout_s=.3, read_state=lambda: state, clock=clock.time, sleep=clock.sleep)
    assert not (audio / 'STOP_AUDIO').exists()


def test_ambiguous_observer_is_never_signalled_or_checkpointed(tmp_path):
    folder, audio = session(tmp_path)
    close_recording(audio)
    observer = folder / 'observer'
    observer.mkdir()
    (folder / 'observer-process.json').write_text('{"pid":123}')
    clock = Clock()
    with pytest.raises(TimeoutError, match='Observer completion is unverified'):
        finish(folder, timeout_s=.3, read_state=idle, clock=clock.time, sleep=clock.sleep,
               make_checkpoint=lambda _: pytest.fail('Live observer checkpointed'))
    assert not (observer / 'STOP').exists()


def make_archive(folder, complete=True, speaker_test=False, empty_input=False, request=True, cancelled=False):
    archive = SessionArchive(folder / 'turn-recording')
    import io
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b'\0' * 40)
    turn = archive.begin_turn('one', b'' if empty_input else buffer.getvalue(), True, test=speaker_test)
    if complete:
        if request:
            turn.record_request({'messages': []})
        if not cancelled:
            turn.write_audio([.1, .2, .3])
        turn.finish_generation(SimpleNamespace(text='' if cancelled else 'test', error=None, cancel=cancelled))
        turn.finish_playback(0 if cancelled else 3, cancelled)
    return archive.folder


def test_archive_error_does_not_stop_recorder(tmp_path):
    folder, audio = session(tmp_path)
    with pytest.raises(RuntimeError, match='Per-turn archive failed'):
        finish(folder, read_state=lambda: dict(idle(), archive_error='disk unavailable'))
    assert not (audio / 'STOP_AUDIO').exists()


@pytest.mark.parametrize('failure', ['unfinished', 'hash', 'status', 'external'])
def test_archive_must_be_closed_complete_and_inside_checkpoint(tmp_path, failure):
    folder, audio = session(tmp_path)
    archive = make_archive(tmp_path if failure == 'external' else folder, complete=failure != 'unfinished')
    if failure == 'hash':
        (archive / 'turn-one' / 'generated.wav').write_bytes(b'corrupted')
    if failure == 'status':
        (archive / 'turn-one' / 'status.json').unlink()
    state = dict(idle(), archive_session=str(archive))
    with pytest.raises((ValueError, FileNotFoundError)):
        finish(folder, read_state=lambda: state,
               make_checkpoint=lambda _: pytest.fail('Unsafe turn archive checkpointed'))
    assert not (audio / 'STOP_AUDIO').exists()


def test_complete_turn_archive_is_verified_and_included(tmp_path):
    folder, audio = session(tmp_path)
    archive = make_archive(folder)
    close_recording(audio)
    state = dict(idle(), archive_session=str(archive), archive_error=None)
    result = finish(folder, read_state=lambda: state,
                    make_checkpoint=lambda p: checkpoint(p, tmp_path / 'backups'))
    copied = Path(result['checkpoint']) / 'experiment' / 'turn-recording' / 'turn-one'
    assert (copied / 'generated.wav').read_bytes() == (archive / 'turn-one' / 'generated.wav').read_bytes()
    verify_checkpoint(result['checkpoint'])


def test_mute_uses_idempotent_desired_state_request(monkeypatch):
    import io
    seen = []
    def fake_open(request, timeout):
        seen.append(request)
        return io.BytesIO(b'{"ok":true}')
    monkeypatch.setattr('src.synth.finish_recorded_test.urlopen', fake_open)
    _mute('http://127.0.0.1:8000')
    assert seen[0].full_url.endswith('/api/mic')
    assert json.loads(seen[0].data) == {'enabled': False}


@pytest.mark.parametrize('speaker_test', [False, True])
def test_empty_input_allowed_only_for_explicit_speaker_check(tmp_path, speaker_test):
    folder, audio = session(tmp_path)
    archive = make_archive(folder, speaker_test=speaker_test, empty_input=True)
    close_recording(audio)
    call = lambda: finish(folder, read_state=lambda: dict(idle(), archive_session=str(archive)),
                          make_checkpoint=lambda p: checkpoint(p, tmp_path / 'backups'))
    if speaker_test:
        assert call()['phase'] == 'verified-checkpoint'
    else:
        with pytest.raises((wave.Error, EOFError)):
            call()
        assert not (audio / 'STOP_AUDIO').exists()


@pytest.mark.parametrize('cancelled', [False, True])
def test_missing_request_allowed_only_for_cancelled_before_request(tmp_path, cancelled):
    folder, audio = session(tmp_path)
    archive = make_archive(folder, request=False, cancelled=cancelled)
    close_recording(audio)
    call = lambda: finish(folder, read_state=lambda: dict(idle(), archive_session=str(archive)),
                          make_checkpoint=lambda p: checkpoint(p, tmp_path / 'backups'))
    if cancelled:
        assert call()['phase'] == 'verified-checkpoint'
    else:
        with pytest.raises(ValueError, match='Request payload is missing'):
            call()
        assert not (audio / 'STOP_AUDIO').exists()


@pytest.mark.parametrize('change', [{'active_turn': True}, {'archive_error': 'late write failure'},
                                   {'archive_session': '/changed'}])
def test_app_state_is_revalidated_after_recorder_finishes(tmp_path, change):
    folder, audio = session(tmp_path)
    close_recording(audio)
    calls = []
    def state():
        calls.append(True)
        result = idle()
        if len(calls) >= 4:  # Extra read verifies the app's seal before STOP_AUDIO.
            result.update(change)
        return result
    with pytest.raises(RuntimeError):
        finish(folder, read_state=state,
               make_checkpoint=lambda _: pytest.fail('Changed app state checkpointed'))
    assert (audio / 'STOP_AUDIO').exists()


@pytest.mark.parametrize('change', ['activity', 'file'])
def test_late_changes_during_checkpoint_prevent_success_receipt(tmp_path, change):
    folder, audio = session(tmp_path)
    close_recording(audio)
    state = idle()
    destinations = []
    def save(source):
        destination = checkpoint(source, tmp_path / 'backups')
        destinations.append(destination)
        if change == 'activity':
            state['active_turn'] = True
        else:
            (audio / 'late-file.json').write_text('{}')
        return destination
    with pytest.raises(RuntimeError, match='during checkpoint'):
        finish(folder, read_state=lambda: state.copy(), make_checkpoint=save)
    assert not (destinations[0] / 'finalization-receipt.json').exists()
    verify_checkpoint(destinations[0])  # Historical copy remains valid.


@pytest.mark.parametrize('failure', ['missing', 'length', 'nonfinite'])
def test_exact_generated_float_stream_must_be_present_complete_and_finite(tmp_path, failure):
    folder, audio = session(tmp_path)
    archive = make_archive(folder)
    turn = archive / 'turn-one'
    path = turn / 'generated.f32'
    manifest_path = turn / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if failure == 'missing':
        path.unlink()
        del manifest['files']['generated.f32']
    else:
        data = b'\0' * 4 if failure == 'length' else b'\0\0\xc0\x7f' * 3
        path.write_bytes(data)
        manifest['files']['generated.f32'] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='float audio'):
        finish(folder, read_state=lambda: dict(idle(), archive_session=str(archive)),
               make_checkpoint=lambda _: pytest.fail('Invalid exact audio checkpointed'))
    assert not (audio / 'STOP_AUDIO').exists()
