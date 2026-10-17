"""Preservation fences using fabricated archives and inert app dependencies."""
import io
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from app import Assistant, Handler
from session_archive import SessionArchive
import session_archive as archive_module
from src.synth.finish_recorded_test import finish, _seal
from src.synth.checkpoint_experiment import checkpoint
from test_realtime import FakeMic, FakeFront, FakePlayer, FakeClient
from test_recorded_test_finish import session, close_recording, idle


def assistant(archive=None):
    a = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(),
                  client=FakeClient(), start_muted=True, session_archive=archive)
    a.status = 'listening'
    return a


def roster(folder):
    return {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob('*') if p.is_file()}


def test_sealed_archive_refuses_new_turn_and_restart(tmp_path):
    archive = SessionArchive(tmp_path / 'turns')
    archive.seal()
    before = roster(archive.folder)
    assert archive.sealed
    archive.seal()  # Idempotent and does not rewrite the marker.
    with pytest.raises(RuntimeError, match='finalized'):
        archive.begin_turn(1, b'neutral', True)
    with pytest.raises(FileExistsError):
        SessionArchive(archive.folder)
    assert roster(archive.folder) == before


def test_successful_finalization_fences_later_activity(tmp_path):
    folder, audio = session(tmp_path)
    close_recording(audio)
    archive = SessionArchive(folder / 'turns')
    a = assistant(archive)
    finish(folder, read_state=a.state,
           make_checkpoint=lambda p: checkpoint(p, tmp_path / 'backup'), seal_archive=a.seal_archive)
    before = roster(folder)
    with pytest.raises(RuntimeError, match='finalized'):
        a.set_mic_enabled(True)
    assert roster(folder) == before


@pytest.mark.parametrize('action', ['enable', 'submit', 'speaker'])
def test_app_fences_every_entrypoint_after_seal(tmp_path, action):
    archive = SessionArchive(tmp_path / 'turns')
    a = assistant(archive)
    assert a.seal_archive(str(archive.folder))['archive_sealed'] is True
    before = roster(archive.folder)
    stops = a.player.stops
    with pytest.raises(RuntimeError, match='finalized'):
        if action == 'enable': a.set_mic_enabled(True)
        elif action == 'speaker': a.test_speaker()
        else: a.submit_audio([.1] * 3200)
    assert not a.mic_on and not a.client.calls and a.player.stops == stops
    assert not a._pending_jobs and roster(archive.folder) == before


def test_physical_only_session_is_fenced_and_path_mismatch_rejected(tmp_path):
    a = assistant()
    with pytest.raises(ValueError): a.seal_archive(str(tmp_path / 'other'))
    assert a.seal_archive(None) == {'ok': True, 'archive_session': None, 'archive_sealed': True}
    with pytest.raises(RuntimeError): a.set_mic_enabled(True)


@pytest.mark.parametrize('field,value', [('mic_on', True), ('_active', object()),
    ('_pending_jobs', {1: object()}), ('_archive_pending', {1}), ('archive_error', 'failed')])
def test_busy_or_failed_app_cannot_seal(tmp_path, field, value):
    archive = SessionArchive(tmp_path / 'turns')
    a = assistant(archive)
    setattr(a, field, value)
    with pytest.raises(RuntimeError): a.seal_archive(str(archive.folder))
    assert not archive.sealed


def test_archive_seal_refuses_unfinished_writer(tmp_path):
    archive = SessionArchive(tmp_path / 'turns')
    turn = archive.begin_turn(1, b'neutral', True)
    with pytest.raises(RuntimeError): archive.seal()
    turn.finish_generation({'text': '', 'cancel': True})
    turn.finish_playback(0, True)
    archive.seal()
    assert archive.sealed


def test_finalizer_requires_ack_before_stopping_and_seal_survives_failure(tmp_path):
    folder, audio = session(tmp_path)
    archive = SessionArchive(folder / 'turns')
    a = assistant(archive)
    close_recording(audio)
    def fail_checkpoint(_): raise OSError('simulated checkpoint failure')
    with pytest.raises(OSError, match='checkpoint failure'):
        finish(folder, read_state=a.state, seal_archive=a.seal_archive, make_checkpoint=fail_checkpoint)
    assert archive.sealed and a.state()['archive_sealed']
    with pytest.raises(RuntimeError): a.set_mic_enabled(True)


@pytest.mark.parametrize('ack', [{}, {'ok': True},
    {'ok': True, 'archive_session': None, 'archive_sealed': False},
    {'ok': True, 'archive_session': '/other', 'archive_sealed': True}])
def test_bad_ack_never_stops_recorder(tmp_path, ack):
    folder, audio = session(tmp_path)
    with pytest.raises(RuntimeError, match='seal'):
        finish(folder, read_state=idle, seal_archive=lambda expected: ack,
               make_checkpoint=lambda _: pytest.fail('Premature checkpoint'))
    assert not (audio / 'STOP_AUDIO').exists()


def test_unsupported_seal_never_stops_recorder(tmp_path):
    folder, audio = session(tmp_path)
    def unsupported(_): raise OSError('old app has no endpoint')
    with pytest.raises(OSError): finish(folder, read_state=idle, seal_archive=unsupported)
    assert not (audio / 'STOP_AUDIO').exists()


@pytest.mark.parametrize('action', ['enable', 'submit', 'speaker'])
def test_seal_wins_barrier_race_before_new_activity(tmp_path, monkeypatch, action):
    archive = SessionArchive(tmp_path / 'turns')
    a = assistant(archive)
    entered, release = threading.Event(), threading.Event()
    original = archive.seal
    def delayed():
        entered.set()
        assert release.wait(2)
        return original()
    monkeypatch.setattr(archive, 'seal', delayed)
    errors = []
    def new_activity():
        try:
            if action == 'enable': a.set_mic_enabled(True)
            elif action == 'submit': a.submit_audio([.1] * 3200)
            else: a.test_speaker()
        except RuntimeError as exc: errors.append(str(exc))
    sealing = threading.Thread(target=lambda: a.seal_archive(str(archive.folder)))
    sealing.start()
    assert entered.wait(1)
    incoming = threading.Thread(target=new_activity)
    incoming.start()
    release.set()
    sealing.join(2); incoming.join(2)
    assert not sealing.is_alive() and not incoming.is_alive()
    assert errors and 'finalized' in errors[0]
    assert not a._pending_jobs and not a.client.calls and not a.mic_on
    assert not list(archive.folder.glob('turn-*'))


def test_submit_wins_barrier_and_prevents_seal(tmp_path, monkeypatch):
    archive = SessionArchive(tmp_path / 'turns')
    a = assistant(archive)
    entered, release = threading.Event(), threading.Event()
    original = a._require_recording_open
    def delayed():
        original(); entered.set()
        assert release.wait(2)
    monkeypatch.setattr(a, '_require_recording_open', delayed)
    monkeypatch.setattr(a, '_spawn', lambda *args: None)  # No generation/playback.
    incoming = threading.Thread(target=lambda: a.submit_audio([.1] * 3200))
    incoming.start(); assert entered.wait(1)
    errors = []
    def sealing():
        try: a.seal_archive(str(archive.folder))
        except RuntimeError as exc: errors.append(str(exc))
    closer = threading.Thread(target=sealing); closer.start()
    release.set(); incoming.join(2); closer.join(2)
    assert errors and not archive.sealed and not a.client.calls


def test_begin_turn_and_seal_share_lifecycle_lock(tmp_path, monkeypatch):
    archive = SessionArchive(tmp_path / 'turns')
    entered, release = threading.Event(), threading.Event()
    original = archive_module.TurnArchive
    def delayed(*args):
        entered.set(); assert release.wait(2)
        return original(*args)
    monkeypatch.setattr(archive_module, 'TurnArchive', delayed)
    creator = threading.Thread(target=lambda: archive.begin_turn(1, b'neutral', True))
    creator.start(); assert entered.wait(1)
    errors = []
    def seal():
        try: archive.seal()
        except RuntimeError as exc: errors.append(str(exc))
    closer = threading.Thread(target=seal); closer.start()
    release.set(); creator.join(2); closer.join(2)
    assert errors and not archive.sealed


def test_marker_survives_failed_seal_and_fences_future_writers(tmp_path, monkeypatch):
    archive = SessionArchive(tmp_path / 'turns')
    a = assistant(archive)
    original = archive_module._exclusive_json
    def failed(path, data):
        if path.name == 'sealed.json':
            path.write_bytes(b'{')
            raise OSError('simulated seal persistence failure')
        return original(path, data)
    monkeypatch.setattr(archive_module, '_exclusive_json', failed)
    with pytest.raises(OSError): a.seal_archive(str(archive.folder))
    assert a.archive_error and archive.sealed
    with pytest.raises(RuntimeError): archive.begin_turn(1, b'neutral', True)
    with pytest.raises(RuntimeError): a.set_mic_enabled(True)
    with pytest.raises(RuntimeError): a.seal_archive(str(archive.folder))


def test_marker_is_reloaded_when_memory_flag_is_clear(tmp_path):
    archive = SessionArchive(tmp_path / 'turns')
    archive.seal()
    archive._sealed = False
    assert archive.sealed
    with pytest.raises(RuntimeError): archive.begin_turn(1, b'', True)


@pytest.mark.parametrize('payload', [{}, [], {'expected_session': True},
    {'expected_session': 'relative'}, {'expected_session': None, 'extra': 1}])
def test_seal_route_validates_inputs_without_network(tmp_path, payload):
    a = assistant(SessionArchive(tmp_path / 'turns'))
    raw = json.dumps(payload).encode()
    handler = object.__new__(Handler)
    handler.path = '/api/archive/seal'
    handler.headers = {'Content-Length': str(len(raw))}
    handler.rfile = io.BytesIO(raw)
    handler.server = SimpleNamespace(assistant=a)
    handler._local = lambda: True
    seen = []
    handler._send = lambda body, status=200, **kwargs: seen.append(status)
    handler.do_POST()
    assert seen == [400] and not a.session_archive.sealed


def test_ack_without_sealed_state_is_refused(tmp_path):
    folder, audio = session(tmp_path)
    with pytest.raises(RuntimeError, match='unverified'):
        finish(folder, read_state=idle,
               seal_archive=lambda expected: {'ok': True, 'archive_session': expected, 'archive_sealed': True})
    assert not (audio / 'STOP_AUDIO').exists()


def test_seal_http_helper_sends_only_explicit_identity(monkeypatch, tmp_path):
    seen = []
    def fake_open(request, timeout):
        seen.append((request, timeout))
        return io.BytesIO(b'{"ok":true,"archive_session":null,"archive_sealed":true}')
    monkeypatch.setattr('src.synth.finish_recorded_test.urlopen', fake_open)
    assert _seal('http://127.0.0.1:8000', str(tmp_path))['archive_sealed']
    request, timeout = seen[0]
    assert request.full_url == 'http://127.0.0.1:8000/api/archive/seal'
    assert json.loads(request.data) == {'expected_session': str(tmp_path)}
    assert request.get_method() == 'POST' and timeout == 2
