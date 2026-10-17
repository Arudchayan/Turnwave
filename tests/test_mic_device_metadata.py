"""Process-local device metadata using fake PortAudio inventories only."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


class FakePortAudio:
    def __init__(self, devices, hosts=None, default=0):
        self.devices = devices
        self.hosts = hosts or ['Windows WASAPI', 'MME']
        self.default = SimpleNamespace(device=[default, 0])
        self.calls = []
        self.bad_hosts = set()
        self.unavailable = False
        self.opened = []

    def query_devices(self, device=None):
        self.calls.append(('devices', device))
        if self.unavailable:
            raise OSError('inventory unavailable')
        if device is None:
            return [dict(row, index=i) for i, row in enumerate(self.devices)]
        if isinstance(device, str):
            matches = [i for i, row in enumerate(self.devices)
                       if row['name'] == device]
            if len(matches) != 1:
                raise ValueError('ambiguous or unavailable device')
            device = matches[0]
        if not 0 <= device < len(self.devices):
            raise ValueError('unavailable device')
        return dict(self.devices[device], index=device)

    def query_hostapis(self, index):
        self.calls.append(('host', index))
        if index in self.bad_hosts:
            raise OSError('host unavailable')
        return {'name': self.hosts[index]}

    def InputStream(self, **kwargs):
        self.opened.append(kwargs)
        return SimpleNamespace(active=False, latency=.04,
                               start=lambda: None, close=lambda: None,
                               stop=lambda: None)


def device(name='Full microphone endpoint', host=0, channels=2, rate=48000):
    return dict(name=name, hostapi=host, max_input_channels=channels,
                default_samplerate=rate)


@pytest.fixture
def load_audio(monkeypatch):
    def load(fake):
        # Import under a private name so even sounddevice initialization is fake.
        monkeypatch.setitem(sys.modules, 'sounddevice', fake)
        path = Path(__file__).resolve().parents[1] / 'src' / 'audio_io.py'
        spec = importlib.util.spec_from_file_location('metadata_audio_io', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return load


def test_full_catalog_preserves_host_duplicates_and_legacy_labels(load_audio):
    prefix = 'Microphone endpoint ' + 'x' * 50
    fake = FakePortAudio([device(prefix + ' A'), device(prefix + ' B', host=1),
                          device('output only', channels=0), device(prefix + ' A')])
    audio = load_audio(fake)
    rows = audio.list_input_devices()
    assert [row['index'] for row in rows] == [0, 1, 3]
    assert [row['full_name'] for row in rows] == [prefix + ' A', prefix + ' B', prefix + ' A']
    assert [row['host_api'] for row in rows] == ['Windows WASAPI', 'MME', 'Windows WASAPI']
    assert rows[0]['max_input_channels'] == 2
    assert rows[0]['default_samplerate'] == 48000.0
    assert fake.calls.count(('host', 0)) == 1  # One lookup per host per roster.
    assert audio.list_inputs() == [(0, prefix[:48]), (1, prefix[:48]), (3, prefix[:48])]


def test_same_endpoint_uses_each_process_inventory_index(load_audio):
    full_name = 'Microphone Array (complete endpoint identifier)'
    def inventory(index):
        return [device('output ' + str(i), channels=0) for i in range(index)] + [device(full_name)]
    external = load_audio(FakePortAudio(inventory(18))).list_input_devices()
    app = load_audio(FakePortAudio(inventory(12))).list_input_devices()
    assert external[0]['index'] == 18 and app[0]['index'] == 12
    assert {k: v for k, v in external[0].items() if k != 'index'} == {
        k: v for k, v in app[0].items() if k != 'index'}


def test_selected_identity_is_bound_cached_defensive_metadata(load_audio):
    fake = FakePortAudio([device('chosen', channels=8)])
    audio = load_audio(fake)
    mic = audio.Mic(device=0)
    identity = mic.selected_device_identity
    assert identity == dict(index=0, full_name='chosen', host_api='Windows WASAPI',
                            host_api_index=0, max_input_channels=8,
                            default_samplerate=48000.0, capture_samplerate=48000,
                            capture_channels=4, capture_dtype='float32',
                            output_samplerate=16000)
    calls = list(fake.calls)
    identity['full_name'] = 'mutated'
    fake.devices[0]['name'] = 'later inventory'
    for _ in range(20):
        assert mic.selected_device_identity['full_name'] == 'chosen'
    mic._cb(np.zeros((960, 4), np.float32), 960, None, False)
    assert fake.calls == calls  # Neither property reads nor callbacks enumerate.
    with pytest.raises(AttributeError):
        mic.selected_device_identity = {}


@pytest.mark.parametrize('selection', [None, 'selected full endpoint'])
def test_default_or_unique_name_is_pinned_to_queried_numeric_index(load_audio, selection):
    fake = FakePortAudio([device('other'), device('selected full endpoint')], default=1)
    audio = load_audio(fake)
    mic = audio.Mic(device=selection)
    assert mic.device == 1 and mic.selected_device_identity['index'] == 1
    fake.default.device[0] = 0
    before = list(fake.calls)
    mic.start()  # A fake constructor only; never opens PortAudio.
    assert fake.opened[-1]['device'] == 1
    assert fake.calls == before


def test_ambiguous_name_does_not_invent_a_selected_identity(load_audio):
    fake = FakePortAudio([device('same full endpoint'), device('same full endpoint', host=1)])
    audio = load_audio(fake)
    assert len(audio.list_input_devices()) == 2
    assert audio.Mic(device='same full endpoint').selected_device_identity is None


def test_unavailable_or_output_only_selection_has_no_identity(load_audio):
    fake = FakePortAudio([device('output only', channels=0)])
    audio = load_audio(fake)
    assert audio.list_input_devices() == []
    assert audio.Mic(device=0).selected_device_identity is None
    assert audio.Mic(device=12).selected_device_identity is None
    fake.unavailable = True
    assert audio.list_input_devices() == [] and audio.list_inputs() == []
    assert audio.Mic(device=0).selected_device_identity is None


def test_failed_host_metadata_is_unknown_not_fabricated(load_audio):
    fake = FakePortAudio([device()])
    fake.bad_hosts.add(0)
    audio = load_audio(fake)
    row = audio.list_input_devices()[0]
    assert row['host_api'] is None and row['host_api_index'] == 0
    assert audio.list_inputs() == [(0, 'Full microphone endpoint')]
    assert audio.Mic(device=0).selected_device_identity is None


def test_inconsistent_query_index_or_missing_index_is_not_guessed(load_audio):
    fake = FakePortAudio([device()])
    original = fake.query_devices
    fake.query_devices = lambda selected=None: dict(original(selected), index=3)
    audio = load_audio(fake)
    assert audio.Mic(device=0).selected_device_identity is None
    fake.query_devices = lambda selected=None: {k: v for k, v in original(selected).items() if k != 'index'}
    assert audio.Mic(device='Full microphone endpoint').selected_device_identity is None
