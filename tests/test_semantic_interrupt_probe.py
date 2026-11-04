"""The optional ASR screen can use a local Whisper model without contacting a server."""
import json
import sys
from types import SimpleNamespace
import numpy as np
from src.synth import semantic_interrupt_probe as screen


def test_local_whisper_screen_uses_local_model_and_records_missing_captures(monkeypatch, tmp_path):
    fixtures = tmp_path / 'fixtures'
    fixtures.mkdir()
    for name in screen.FIXTURES:
        (fixtures / name).write_bytes(b'synthetic fixture')
    model_dir = tmp_path / 'local-snapshot'
    model_dir.mkdir()
    output = tmp_path / 'report.json'
    calls = []
    class FakeWhisper:
        def __init__(self, path, **kwargs):
            calls.append((path, kwargs))
        def transcribe(self, audio, **kwargs):
            assert isinstance(audio, np.ndarray)
            assert kwargs == {'language': 'en', 'beam_size': 5,
                              'condition_on_previous_text': False, 'vad_filter': False}
            return iter([SimpleNamespace(text=' test ')]), None
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=FakeWhisper))
    monkeypatch.setattr(screen.importlib.metadata, 'version', lambda name: 'test-version')
    monkeypatch.setattr(screen.probe, 'read_wav', lambda path: (np.ones(32000, np.float32), 16000))
    monkeypatch.setattr(screen.probe, 'resample', lambda audio, rate: audio)
    monkeypatch.setattr(screen.probe, 'run_case', lambda *a, **kw: {'cancel_ms': None, 'error': None, 'aec_error': None})
    monkeypatch.setattr(screen, 'backend_session', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('server should not be used')))
    monkeypatch.setattr(sys, 'argv', ['probe', '--fixtures', str(fixtures), '--whisper-model', str(model_dir), '--out', str(output)])
    screen.main()
    report = json.loads(output.read_text(encoding='utf-8'))
    assert calls[0][1]['local_files_only'] is True
    assert calls[0][1]['device'] == 'cpu' and calls[0][1]['compute_type'] == 'int8'
    assert report['model']['revision'] == 'local-snapshot'
    assert len(report['rows']) == 8
    assert all(row['clean_asr'] == 'test' and row['capture_asr'] is None for row in report['rows'])
