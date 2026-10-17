"""Local benchmark evidence boundaries, using neutral audio and injected clients."""
import base64
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import wave

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from client import ServerClient
from synth import content_benchmark as bench


def digest(data):
    return hashlib.sha256(data).hexdigest()


def wav_bytes(rate=16000, channels=1, width=2, silent=False):
    stream = io.BytesIO()
    with wave.open(stream, 'wb') as writer:
        writer.setparams((channels, width, rate, 0, 'NONE', 'not compressed'))
        writer.writeframes(bytes(320 * channels * width) if silent else b'\x01' * (320 * channels * width))
    return stream.getvalue()


def source(tmp_path, count=1, **audio_options):
    folder = tmp_path / 'source'
    folder.mkdir()
    audio = wav_bytes(**audio_options)
    (folder / 'input.wav').write_bytes(audio)
    metadata = {'prompt': 'Choose the first letter.', 'reference': 'A',
                'id': 'original-row-id', 'status': 'original-source-field'}
    manifest = {'version': 1, 'dataset': 'neutral/fixture', 'revision': 'a' * 40,
                'cases': [{'id': f'case-{i}', 'subset': 'openbookqa', 'audio': 'input.wav',
                           'sha256': digest(audio), 'metadata': dict(metadata, prompt=f'Question {i}')} for i in range(count)]}
    path = folder / 'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    return path, manifest, audio


class InjectedClient:
    def __init__(self, behavior=None):
        self.behavior = behavior
        self.calls = []

    def chat_turn(self, wav, turn, first, on_audio=None):
        self.calls.append((wav, turn, first))
        payload = {'reset_context': first, 'messages': [{'role': 'user', 'content': [
            {'type': 'input_audio', 'input_audio': {'data': base64.b64encode(wav).decode(), 'format': 'wav'}}]}]}
        turn.request_observer(payload)
        payload['reset_context'] = False  # Observer must retain an immutable snapshot.
        if self.behavior:
            self.behavior(turn, on_audio)
        else:
            turn.text_parts.append('A')
            on_audio(np.array([.125, -.25, .5], np.float32))
        turn.done.set()


def run(tmp_path, behavior=None, count=1):
    path, manifest, audio = source(tmp_path, count)
    prepared = bench.prepare(path, tmp_path / 'prepared')
    client = InjectedClient(behavior)
    folder = bench.infer_injected(prepared, tmp_path / 'run', client=client,
                                  mock=True, provenance={'fixture': 'neutral'})
    return Path(folder), client, manifest, audio


def test_exact_request_frames_and_fresh_context(tmp_path):
    def reuse(turn, receive):
        frame = np.array([.25, -.5], dtype=np.float32)
        receive(frame)
        frame[:] = [.75, 0]
        receive(frame)
        frame[:] = 9
        turn.text_parts.append('A')
    folder, client, manifest, audio = run(tmp_path, reuse, count=2)
    assert len({id(turn) for _, turn, _ in client.calls}) == 2
    assert all(first and wav == audio for wav, _, first in client.calls)
    case = folder / 'cases' / 'case-0'
    assert (case / 'input.wav').read_bytes() == audio
    assert json.loads((case / 'request.json').read_text())['reset_context'] is True
    np.testing.assert_array_equal(np.fromfile(case / 'generated.f32', '<f4'), [.25, -.5, .75, 0])
    with wave.open(str(case / 'generated.wav')) as reader:
        assert reader.getframerate() == 24000 and reader.getnframes() == 4
    result = json.loads((case / 'result.json').read_text())
    assert result['status'] == 'complete' and result['generated_samples'] == 4
    assert result['generated_sha256'] == digest((case / 'generated.f32').read_bytes())
    assert (case / 'model-text.txt').read_text() == 'A'
    assert len((folder / 'results.jsonl').read_text().splitlines()) == 2


@pytest.mark.parametrize('mode,expected', [('raise', 'error'), ('nan', 'error'),
    ('empty', 'empty'), ('silent', 'silent'), ('cancel', 'cancelled'), ('length', 'length')])
def test_failure_outcomes_are_durable_and_not_ordinary_answers(tmp_path, mode, expected):
    def behavior(turn, receive):
        turn.text_parts.append('A partial answer')
        if mode == 'empty':
            return
        receive(np.zeros(5, np.float32) if mode == 'silent' else np.ones(5, np.float32) * .25)
        if mode == 'raise':
            raise OSError('simulated stream failure')
        if mode == 'nan':
            receive(np.array([float('nan')], np.float32))
        if mode == 'cancel':
            turn.abort()
        if mode == 'length':
            turn.finish_reason = 'length'
    folder, _, _, _ = run(tmp_path, behavior)
    result = json.loads((folder / 'cases/case-0/result.json').read_text())
    assert result['status'] == expected
    assert (folder / 'cases/case-0/model-text.txt').read_text() == 'A partial answer'
    assert len((folder / 'results.jsonl').read_text().splitlines()) == 1
    exported = bench.export_grades(folder, tmp_path / 'grades')
    row = json.loads((exported / 'openbookqa-model-text.jsonl').read_text())
    assert row['response'] == '' and row['_localvoice']['eligible'] is False
    roster = json.loads((exported / 'roster.json').read_text())
    assert roster['expected_cases'] == 1 and roster['cases'][0]['status'] == expected


@pytest.mark.parametrize('audio_options', [{'rate': 24000}, {'channels': 2}, {'width': 1}])
def test_invalid_input_audio_rejected_before_output(tmp_path, audio_options):
    path, _, _ = source(tmp_path, **audio_options)
    with pytest.raises(ValueError):
        bench.prepare(path, tmp_path / 'prepared')
    assert not (tmp_path / 'prepared').exists()


@pytest.mark.parametrize('change', ['hash', 'truncated', 'path', 'duplicate', 'reserved', 'revision', 'metadata'])
def test_manifest_integrity_and_safe_names(tmp_path, change):
    path, manifest, audio = source(tmp_path)
    item = manifest['cases'][0]
    if change == 'hash':
        item['sha256'] = '0' * 64
    elif change == 'truncated':
        audio = audio[:-1]
        (path.parent / 'input.wav').write_bytes(audio)
        item['sha256'] = digest(audio)
    elif change == 'path':
        item['audio'] = '../escape.wav'
    elif change == 'duplicate':
        manifest['cases'].append(dict(item, id='CASE-0'))
    elif change == 'reserved':
        item['id'] = 'CON'
    elif change == 'revision':
        manifest['revision'] = 'main'
    elif change == 'metadata':
        item['metadata']['response'] = 'spoofed response'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        bench.prepare(path, tmp_path / 'prepared')
    assert not (tmp_path / 'prepared').exists()


def test_prepared_input_tamper_and_output_reuse_refused(tmp_path):
    path, _, _ = source(tmp_path)
    prepared = bench.prepare(path, tmp_path / 'prepared')
    with pytest.raises(FileExistsError):
        bench.prepare(path, prepared)
    (prepared / 'cases/case-0/input.wav').write_bytes(wav_bytes(silent=True))
    fake = InjectedClient()
    with pytest.raises(ValueError):
        bench.infer_injected(prepared, tmp_path / 'run', client=fake, mock=True, provenance={'fixture': True})
    assert not fake.calls and not (tmp_path / 'run').exists()


def test_missing_asr_never_uses_model_text_and_metadata_survives(tmp_path):
    folder, _, manifest, _ = run(tmp_path)
    exported = bench.export_grades(folder, tmp_path / 'grades')
    model = json.loads((exported / 'openbookqa-model-text.jsonl').read_text())
    spoken = json.loads((exported / 'openbookqa-spoken-asr.jsonl').read_text())
    for key, value in manifest['cases'][0]['metadata'].items():
        assert model[key] == spoken[key] == value
    assert model['response'] == 'A' and spoken['response'] == ''
    assert model['_localvoice']['eligible'] is True
    assert spoken['_localvoice']['eligible'] is False
    assert spoken['_localvoice']['reason'] == 'missing_asr'


@pytest.mark.parametrize('bad', [None, 'hash', 'duplicate', 'unknown', 'provenance'])
def test_asr_is_explicit_and_bound_to_audio(tmp_path, bad):
    folder, _, _, _ = run(tmp_path)
    result = json.loads((folder / 'cases/case-0/result.json').read_text())
    row = {'id': 'case-0', 'generated_sha256': result['generated_sha256'], 'response': 'B'}
    data = {'version': 1, 'provenance': {'engine': 'fixture', 'model': 'neutral', 'model_sha256': 'b' * 64}, 'cases': [row]}
    if bad == 'hash': row['generated_sha256'] = 'c' * 64
    if bad == 'duplicate': data['cases'].append(dict(row))
    if bad == 'unknown': row['id'] = 'missing'
    if bad == 'provenance': data['provenance'] = {}
    path = tmp_path / 'asr.json'
    path.write_text(json.dumps(data))
    if bad:
        with pytest.raises(ValueError): bench.export_grades(folder, tmp_path / 'grades', asr_path=path)
        assert not (tmp_path / 'grades').exists()
    else:
        exported = bench.export_grades(folder, tmp_path / 'grades', asr_path=path)
        row = json.loads((exported / 'openbookqa-spoken-asr.jsonl').read_text())
        assert row['response'] == 'B' and row['_localvoice']['eligible'] is True


def test_real_client_must_acquire_test_ownership_and_cannot_use_mock_mode(tmp_path, monkeypatch):
    path, _, _ = source(tmp_path)
    prepared = bench.prepare(path, tmp_path / 'prepared')
    held, calls = [], []
    @contextmanager
    def owned(base, *, test):
        assert test is True
        held.append(True)
        try: yield
        finally: held.pop()
    class LocalStub(ServerClient):
        def chat_turn(self, wav, turn, first, on_audio=None):
            assert held
            calls.append(first)
            return InjectedClient().chat_turn(wav, turn, first, on_audio)
    client = LocalStub()
    monkeypatch.setattr(bench, 'backend_session', owned)
    with pytest.raises(ValueError):
        bench.infer_injected(prepared, tmp_path / 'bad', client=client, mock=True, provenance={'fixture': True})
    assert not calls
    bench.infer_injected(prepared, tmp_path / 'run', client=client, provenance={'fixture': True})
    assert calls == [True] and not held


def test_result_survives_index_failure_and_export_marks_unindexed(tmp_path, monkeypatch):
    path, _, _ = source(tmp_path)
    prepared = bench.prepare(path, tmp_path / 'prepared')
    def fail(*args): raise OSError('simulated index failure')
    monkeypatch.setattr(bench, '_append_index', fail)
    with pytest.raises(OSError):
        bench.infer_injected(prepared, tmp_path / 'run', client=InjectedClient(), mock=True, provenance={'fixture': True})
    assert (tmp_path / 'run/cases/case-0/result.json').exists()
    exported = bench.export_grades(tmp_path / 'run', tmp_path / 'grades')
    row = json.loads((exported / 'openbookqa-model-text.jsonl').read_text())
    assert not row['_localvoice']['eligible'] and row['_localvoice']['reason'] == 'unindexed'


def test_direct_cli_prepare_and_export_from_an_unrelated_directory(tmp_path):
    path, _, _ = source(tmp_path)
    script = Path(bench.__file__).resolve()
    prepared = tmp_path / 'cli-prepared'
    subprocess.run([sys.executable, str(script), 'prepare', str(path), str(prepared)],
                   cwd=tmp_path, check=True, capture_output=True, text=True, timeout=10)
    folder = bench.infer_injected(prepared, tmp_path / 'run', client=InjectedClient(),
                                  mock=True, provenance={'fixture': True})
    exported = tmp_path / 'cli-grades'
    subprocess.run([sys.executable, str(script), 'export', str(folder), str(exported)],
                   cwd=tmp_path, check=True, capture_output=True, text=True, timeout=10)
    assert json.loads((exported / 'roster.json').read_text())['expected_cases'] == 1


def test_incomplete_index_tail_keeps_full_denominator_and_raw_evidence(tmp_path):
    folder, _, _, _ = run(tmp_path, count=2)
    index_path = folder / 'results.jsonl'
    first = index_path.read_bytes().splitlines(keepends=True)[0]
    torn = first + b'{"id":"case-1","status":"com'
    index_path.write_bytes(torn)
    exported = bench.export_grades(folder, tmp_path / 'grades')
    rows = [json.loads(line) for line in (exported / 'openbookqa-model-text.jsonl').read_text().splitlines()]
    assert len(rows) == 2 and rows[0]['response'] == 'A' and rows[1]['response'] == ''
    assert rows[1]['_localvoice']['reason'] == 'unindexed'
    assert (exported / 'result-index.raw').read_bytes() == torn
    assert json.loads((exported / 'roster.json').read_text())['index_incomplete_tail_bytes'] > 0


def test_malformed_committed_index_rejected_before_export(tmp_path):
    folder, _, _, _ = run(tmp_path)
    with (folder / 'results.jsonl').open('ab') as stream:
        stream.write(b'broken committed row\n')
    with pytest.raises(ValueError):
        bench.export_grades(folder, tmp_path / 'grades')
    assert not (tmp_path / 'grades').exists()


def test_run_metadata_drift_is_rejected_before_export(tmp_path):
    folder, _, _, _ = run(tmp_path)
    path = folder / 'run.json'
    metadata = json.loads(path.read_text())
    metadata['cases'][0]['metadata']['reference'] = 'spoofed answer'
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        bench.export_grades(folder, tmp_path / 'grades')
    assert not (tmp_path / 'grades').exists()


def test_untrusted_run_subset_cannot_escape_export(tmp_path):
    folder, _, _, _ = run(tmp_path)
    source_path, run_path = folder / 'source-manifest.json', folder / 'run.json'
    source_data, run_data = json.loads(source_path.read_text()), json.loads(run_path.read_text())
    source_data['cases'][0]['subset'] = '../escape'
    source_path.write_text(json.dumps(source_data))
    run_data.update(cases=source_data['cases'], source_manifest_sha256=digest(source_path.read_bytes()))
    run_path.write_text(json.dumps(run_data))
    with pytest.raises(ValueError): bench.export_grades(folder, tmp_path / 'grades')
    assert not (tmp_path / 'grades').exists() and not (tmp_path / 'escape-model-text.jsonl').exists()


def test_interrupt_retains_current_case_and_later_unfinished_denominator(tmp_path):
    path, _, _ = source(tmp_path, count=2)
    prepared = bench.prepare(path, tmp_path / 'prepared')
    def interrupt(turn, receive):
        receive(np.array([.25], np.float32))
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        bench.infer_injected(prepared, tmp_path / 'run', client=InjectedClient(interrupt),
                              mock=True, provenance={'fixture': True})
    folder = tmp_path / 'run'
    assert not (folder / 'finished.json').exists()
    assert (folder / 'cases/case-0/result.json').exists()
    exported = bench.export_grades(folder, tmp_path / 'grades')
    roster = json.loads((exported / 'roster.json').read_text())
    assert [row['status'] for row in roster['cases']] == ['error', 'unfinished']
    assert len((exported / 'openbookqa-model-text.jsonl').read_text().splitlines()) == 2


@pytest.mark.parametrize('kind', ['missing', 'reset', 'bytes', 'swallowed_nonfinite'])
def test_bad_client_boundary_cannot_become_complete(tmp_path, kind):
    path, _, _ = source(tmp_path)
    prepared = bench.prepare(path, tmp_path / 'prepared')
    class BrokenClient:
        def chat_turn(self, wav, turn, first, on_audio=None):
            if kind == 'missing':
                turn.text_parts.append('A')
                on_audio(np.ones(3, np.float32))
                return
            if kind == 'swallowed_nonfinite':
                helper = InjectedClient(lambda t, callback: callback(np.array([np.inf], np.float32)))
                try: helper.chat_turn(wav, turn, first, on_audio)
                except ValueError: pass
                return
            payload = {'reset_context': kind != 'reset', 'messages': [{'content': [
                {'type': 'input_audio', 'input_audio': {'format': 'wav', 'data':
                 base64.b64encode(b'wrong' if kind == 'bytes' else wav).decode()}}]}]}
            turn.request_observer(payload)
    folder = bench.infer_injected(prepared, tmp_path / 'run', client=BrokenClient(),
                                  mock=True, provenance={'fixture': True})
    assert json.loads((folder / 'cases/case-0/result.json').read_text())['status'] == 'error'


def test_current_ifeval_metadata_and_silent_input_are_preserved(tmp_path):
    path, manifest, _ = source(tmp_path, silent=True)
    item = manifest['cases'][0]
    item['subset'] = 'ifeval'
    item['metadata'] = {'prompt': 'Say one word.', 'key': 7,
                        'instruction_id_list': ['length_constraints:number_words'],
                        'kwargs': [{'num_words': 1, 'relation': 'at most'}]}
    path.write_text(json.dumps(manifest))
    prepared = bench.prepare(path, tmp_path / 'prepared')
    assert json.loads((prepared / 'cases/case-0/case.json').read_text())['input']['silent'] is True
    folder = bench.infer_injected(prepared, tmp_path / 'run', client=InjectedClient(),
                                  mock=True, provenance={'fixture': True})
    exported = bench.export_grades(folder, tmp_path / 'grades')
    row = json.loads((exported / 'ifeval-model-text.jsonl').read_text())
    assert {k: row[k] for k in item['metadata']} == item['metadata']


def test_real_ownership_denial_precedes_requests_and_output(tmp_path, monkeypatch):
    path, _, _ = source(tmp_path)
    prepared = bench.prepare(path, tmp_path / 'prepared')
    @contextmanager
    def denied(base, *, test):
        assert test is True
        raise RuntimeError('another owner')
        yield
    monkeypatch.setattr(bench, 'backend_session', denied)
    with pytest.raises(RuntimeError, match='another owner'):
        bench.infer_injected(prepared, tmp_path / 'run', client=ServerClient(), provenance={'fixture': True})
    assert not (tmp_path / 'run').exists()


def test_index_append_calls_fsync_with_complete_newline_record(tmp_path, monkeypatch):
    path = tmp_path / 'index.jsonl'
    observed = []
    original = bench.os.fsync
    def synced(fd):
        observed.append(path.read_bytes())
        original(fd)
    monkeypatch.setattr(bench.os, 'fsync', synced)
    bench._append_index(path, {'id': 'fixture', 'status': 'error'})
    assert len(observed) == 1 and observed[0].endswith(b'\n')
    assert json.loads(observed[0])['status'] == 'error'
