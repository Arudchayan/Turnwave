"""Process-local microphone reporting, using fake catalogues and streams only."""
import copy
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import app
from test_realtime import FakeClient, FakeFront, FakeMic, FakePlayer


def descriptor(index=12, name='Neutral full microphone endpoint', host='Windows WASAPI'):
    return {'index': index, 'full_name': name, 'host_api': host,
            'host_api_index': 2, 'max_input_channels': 2, 'default_samplerate': 48000.0}


class CatalogMic(FakeMic):
    def __init__(self, device=12, identity=None, fail=False):
        super().__init__()
        self.device = device
        self.identity = copy.deepcopy(identity or descriptor(device))
        self.fail = fail
        self.opened = False
        self.latency_s = .02

    @property
    def selected_device_identity(self):
        return dict(self.identity, capture_samplerate=48000, capture_channels=2,
                    capture_dtype='float32', output_samplerate=16000)

    def start(self):
        if self.fail:
            raise OSError('Neutral fake open failure')
        self.opened = True

    def stop(self):
        self.opened = False


def make_app(monkeypatch, catalog=None, mic=None):
    snapshots = [copy.deepcopy(catalog if catalog is not None else [descriptor()])]
    calls = []
    def enumerate_devices():
        calls.append(True)
        return snapshots[-1]
    monkeypatch.setattr(app, 'list_input_devices', enumerate_devices, raising=False)
    if mic is None:
        mic = CatalogMic()
    monkeypatch.setattr(app, 'Mic', lambda **kwargs: mic)
    assistant = app.Assistant(player=FakePlayer(), front=FakeFront(), client=FakeClient(), start_muted=True)
    return assistant, mic, calls, snapshots


def start_fake_capture(assistant):
    assistant._loop = lambda: None
    assistant._startup()


def change_device(assistant, index):
    done, cancelled = threading.Event(), threading.Event()
    result = []
    assistant._device_requests.put((index, done, result, cancelled))
    assistant._apply_device_requests()
    assert done.is_set()
    return result


def test_catalog_reports_app_indices_and_full_metadata_before_stream_opens(monkeypatch):
    long_name = 'Neutral endpoint ' + 'x' * 70
    assistant, mic, calls, _ = make_app(monkeypatch, [descriptor(12, long_name)], CatalogMic(12, descriptor(12, long_name)))
    state = assistant.state()
    assert state['mic_devices'] == [descriptor(12, long_name)]
    assert state['mics'] == [(12, long_name[:48])]
    assert state['mic_index'] == 12  # An external process's index18 has no role.
    assert state['mic_identity'] is None  # Constructor metadata is not an opened stream.
    assert len(calls) == 1 and not mic.opened


def test_identity_is_published_after_successful_start_and_state_is_defensive(monkeypatch):
    assistant, mic, calls, snapshots = make_app(monkeypatch)
    start_fake_capture(assistant)
    state = assistant.state()
    assert state['mic_identity'] == mic.selected_device_identity
    state['mic_identity']['full_name'] = 'changed by consumer'
    state['mic_devices'][0]['full_name'] = 'changed by consumer'
    state['mics'].clear()
    snapshots[0][0]['full_name'] = 'changed by enumerator'
    assert assistant.state()['mic_identity']['full_name'] == descriptor()['full_name']
    assert assistant.state()['mic_devices'] == [descriptor()]
    assert assistant.state()['mics'] == [(12, descriptor()['full_name'])]
    assert len(calls) == 1


def test_successful_device_change_refreshes_catalog_and_selected_identity(monkeypatch):
    assistant, old, calls, snapshots = make_app(monkeypatch)
    start_fake_capture(assistant)
    selected = CatalogMic(4, descriptor(4, 'Neutral replacement endpoint'))
    snapshots.append([descriptor(4, 'Neutral replacement endpoint')])
    monkeypatch.setattr(app, 'Mic', lambda **kwargs: selected)
    assert change_device(assistant, 4) == [True]
    state = assistant.state()
    assert state['mic_index'] == 4 and state['mic_identity'] == selected.selected_device_identity
    assert state['mic_devices'] == snapshots[-1]
    assert not old.opened and selected.opened and len(calls) == 2


def test_failed_device_change_keeps_open_identity_and_catalog(monkeypatch):
    assistant, old, calls, snapshots = make_app(monkeypatch)
    start_fake_capture(assistant)
    before = assistant.state()
    monkeypatch.setattr(app, 'Mic', lambda **kwargs: CatalogMic(4, fail=True))
    assert change_device(assistant, 4) == [False]
    state = assistant.state()
    assert state['mic_identity'] == before['mic_identity']
    assert state['mic_devices'] == before['mic_devices']
    assert old.opened and len(calls) == 1


def test_resolved_open_identity_supplies_index_without_default_lookup(monkeypatch):
    selected = CatalogMic(device='Neutral configured name', identity=descriptor())
    assistant, mic, calls, snapshots = make_app(monkeypatch, mic=selected)
    assert assistant.state()['mic_index'] == -1
    start_fake_capture(assistant)
    assert assistant.state()['mic_index'] == 12
    assert assistant.state()['mic_identity']['index'] == 12


def test_cleanup_failure_does_not_attest_a_closed_replacement(monkeypatch):
    assistant, old, calls, snapshots = make_app(monkeypatch)
    start_fake_capture(assistant)
    replacement = CatalogMic(4, descriptor(4, 'Neutral replacement endpoint'))
    snapshots.append([descriptor(4, 'Neutral replacement endpoint')])
    monkeypatch.setattr(app, 'Mic', lambda **kwargs: replacement)
    def fail_stop():
        raise OSError('Neutral old stream cleanup failure')
    old.stop = fail_stop
    assert change_device(assistant, 4) == [False]
    assert not replacement.opened
    assert assistant.state()['mic_identity'] is None
    assert 'cleanup failure' in assistant.state()['error']


def test_cancellation_during_catalog_refresh_does_not_publish_replacement(monkeypatch):
    assistant, old, calls, snapshots = make_app(monkeypatch)
    start_fake_capture(assistant)
    before = assistant.state()
    replacement = CatalogMic(4, descriptor(4, 'Neutral replacement endpoint'))
    monkeypatch.setattr(app, 'Mic', lambda **kwargs: replacement)
    done, cancelled = threading.Event(), threading.Event()
    result = []
    def enumerate_after_timeout():
        cancelled.set()  # set_mic's timeout can win while enumeration is pending.
        return [descriptor(4, 'Neutral replacement endpoint')]
    monkeypatch.setattr(app, 'list_input_devices', enumerate_after_timeout)
    assistant._device_requests.put((4, done, result, cancelled))
    assistant._apply_device_requests()
    assert done.is_set() and result == []
    assert assistant.mic is old and old.opened and not replacement.opened
    assert assistant.state()['mic_identity'] == before['mic_identity']
    assert assistant.state()['mic_devices'] == before['mic_devices']


@pytest.mark.parametrize('catalog,identity', [
    ([descriptor(host=None)], descriptor(host=None)),
    ([descriptor()], dict(descriptor(), default_samplerate=None)),
    ([descriptor(), descriptor(18)], descriptor()),  # Same full identity, two process indices.
    ([descriptor()], descriptor(18)),  # Opened identity does not match this process catalogue.
])
def test_incomplete_or_ambiguous_metadata_does_not_claim_identity(monkeypatch, catalog, identity):
    assistant, mic, calls, snapshots = make_app(monkeypatch, catalog, CatalogMic(identity=identity))
    start_fake_capture(assistant)
    assert assistant.state()['mic_identity'] is None
    assert assistant.state()['mic_devices'] == catalog


def test_state_polls_do_not_query_devices_or_selected_property(monkeypatch):
    assistant, mic, calls, snapshots = make_app(monkeypatch)
    start_fake_capture(assistant)
    def forbidden(*args, **kwargs):
        raise AssertionError('Device query during state polling')
    monkeypatch.setattr(app, 'list_input_devices', forbidden)
    monkeypatch.setattr('audio_io.sd.query_devices', forbidden)
    monkeypatch.setattr('audio_io.sd.query_hostapis', forbidden)
    monkeypatch.setattr(CatalogMic, 'selected_device_identity', property(forbidden))
    for _ in range(8):
        assert assistant.state()['mic_identity']['index'] == 12
    assert len(calls) == 1


def test_injected_mic_without_identity_has_no_device_queries(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Injected fake must not enumerate physical devices')
    monkeypatch.setattr(app, 'list_input_devices', forbidden, raising=False)
    monkeypatch.setattr('audio_io.sd.query_devices', forbidden)
    fake = FakeMic()
    fake.device = None
    assistant = app.Assistant(mic=fake, player=FakePlayer(), front=FakeFront(), client=FakeClient(), start_muted=True)
    state = assistant.state()
    assert state['mic_index'] == -1 and state['mic_identity'] is None
    assert state['mic_devices'] == [] and state['mics'] == []
