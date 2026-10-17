"""Device-free local content evidence; no datasets, models or judges downloaded.

CLI exposes prepare/export only. Inference requires an explicitly supplied client:
real ServerClient instances acquire backend_session(test=True); mock=True accepts
only injected non-ServerClient fixtures. Generated audio is not played audio.
Finite float32 samples are authoritative; PCM16 WAVs are clipped previews.
Scorer exports retain the full denominator, blanking ineligible responses and
recording their reasons separately. ASR is user supplied, never inferred here.
"""
import argparse
import base64
from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import wave

import numpy as np

# The app's shared modules use src as their import root. Bootstrap that layout
# for direct scripts, package CLI and the importable evaluation API alike.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend_session import backend_session
from client import ServerClient, Turn


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _write(path, data):
    with Path(path).open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _json(path, value):
    _write(path, _encoded(value))


def _load(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    _encoded(value)
    return value


def _hash(value, length=64):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{' + str(length) + '}', value) is not None


def _safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', value):
        raise ValueError('Case id must be 1–64 letters, digits, underscores or hyphens')
    if value.upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(10)), *(f'LPT{i}' for i in range(10))}:
        raise ValueError('Reserved case id')
    return value


def _wav(data):
    if len(data) < 44 or data[:4] != b'RIFF' or data[8:12] != b'WAVE' or int.from_bytes(data[4:8], 'little') + 8 != len(data):
        raise ValueError('Expected a complete RIFF WAV')
    try:
        with wave.open(io.BytesIO(data), 'rb') as reader:
            if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getcomptype()) != (1, 2, 16000, 'NONE'):
                raise ValueError('Input must be 16-kHz mono PCM16 WAV')
            frames = reader.getnframes()
            pcm = reader.readframes(frames)
            if not frames or len(pcm) != frames * 2:
                raise ValueError('Empty or truncated WAV samples')
    except (wave.Error, EOFError) as exc:
        raise ValueError('Invalid input WAV') from exc
    return {'samples': frames, 'silent': not np.any(np.frombuffer(pcm, '<i2'))}


def _records(manifest):
    if not isinstance(manifest, dict) or manifest.get('version') != 1 or not isinstance(manifest.get('dataset'), str) or not manifest['dataset'].strip() or not _hash(manifest.get('revision'), 40):
        raise ValueError('Manifest needs version=1, dataset and immutable 40-hex revision')
    cases = manifest.get('cases')
    if not isinstance(cases, list) or not 1 <= len(cases) <= 1024:
        raise ValueError('Manifest needs 1–1024 cases')
    seen, prompts = set(), set()
    for item in cases:
        if not isinstance(item, dict):
            raise ValueError('Each manifest case must be an object')
        case_id = _safe_id(item.get('id'))
        if case_id.casefold() in seen:
            raise ValueError('Duplicate case id')
        seen.add(case_id.casefold())
        subset, metadata = item.get('subset'), item.get('metadata')
        if subset not in {'ifeval', 'openbookqa'} or not isinstance(metadata, dict) or not isinstance(metadata.get('prompt'), str) or not metadata['prompt'].strip():
            raise ValueError('Case needs supported subset and original scorer metadata/prompt')
        if {'response', '_localvoice'} & metadata.keys():
            raise ValueError('Scorer metadata contains a reserved output field')
        if (subset, metadata['prompt']) in prompts:
            raise ValueError('Duplicate prompt would collide in the upstream scorer')
        prompts.add((subset, metadata['prompt']))
        if subset == 'ifeval':
            ids, kwargs = metadata.get('instruction_id_list'), metadata.get('kwargs')
            if 'key' not in metadata or not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids) or not isinstance(kwargs, list) or len(ids) != len(kwargs) or not all(isinstance(x, dict) for x in kwargs):
                raise ValueError('IFEval requires key, instruction_id_list and matching kwargs')
        elif not isinstance(metadata.get('reference'), str):
            raise ValueError('OpenBookQA requires its original reference')
    return cases


def _validate(manifest, base, *, prepared=False):
    validated, total = [], 0
    for item in _records(manifest):
        case_id = item['id']
        relative = item.get('audio')
        if not isinstance(relative, str):
            raise ValueError('Audio must name a local relative WAV')
        relative_path = Path(relative)
        if relative_path.is_absolute() or '..' in relative_path.parts or ':' in relative or '\\' in relative:
            raise ValueError('Audio must stay within the manifest folder')
        path = (base / ('cases/' + case_id + '/input.wav' if prepared else relative)).resolve()
        if not path.is_relative_to(base.resolve()):
            raise ValueError('Audio resolves outside the input folder')
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError('Input WAV exceeds 64 MiB')
        data = path.read_bytes()
        total += len(data)
        if total > 256 * 1024 * 1024:
            raise ValueError('Prepared inputs exceed 256 MiB')
        if not _hash(item.get('sha256')) or _digest(data) != item['sha256']:
            raise ValueError('Input SHA-256 mismatch')
        validated.append((item, data, _wav(data)))
    return validated


def prepare(manifest_path, output_dir):
    """Validate all local sources before creating a fresh immutable input tree."""
    manifest_path, output = Path(manifest_path).resolve(), Path(output_dir).resolve()
    source = manifest_path.read_bytes()
    manifest = json.loads(source)
    _encoded(manifest)
    validated = _validate(manifest, manifest_path.parent)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'source-manifest.json', source)
    _json(output / 'prepared.json', {'version': 1, 'source_manifest_sha256': _digest(source),
                                    'cases': len(validated), 'inference_run': False})
    (output / 'cases').mkdir()
    for item, data, details in validated:
        folder = output / 'cases' / item['id']
        folder.mkdir()
        _write(folder / 'input.wav', data)
        _json(folder / 'case.json', {'source': item, 'input': details})
    return output


def _prepared(folder):
    source = (folder / 'source-manifest.json').read_bytes()
    if _digest(source) != _load(folder / 'prepared.json').get('source_manifest_sha256'):
        raise ValueError('Prepared source manifest changed')
    manifest = json.loads(source)
    return manifest, _validate(manifest, folder, prepared=True)


def _append_index(path, record):
    with path.open('ab') as stream:
        stream.write((json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
        stream.flush()
        os.fsync(stream.fileno())


def _infer_case(folder, item, data, client):
    folder.mkdir()
    _write(folder / 'input.wav', data)
    _json(folder / 'started.json', {'id': item['id'], 'first': True, 'input_sha256': _digest(data)})
    turn, samples, nonzero, callback_error, requested = Turn(), 0, False, None, False
    started = time.perf_counter()

    def request(payload):
        nonlocal requested, callback_error
        try:
            _json(folder / 'request.json', payload)
            requested = True
            content = payload['messages'][-1]['content']
            audio = next(part['input_audio'] for part in content if part.get('type') == 'input_audio')
            if payload.get('reset_context') is not True or audio.get('format') != 'wav' or base64.b64decode(audio['data'], validate=True) != data:
                raise ValueError('Observed request does not preserve input/reset_context')
        except Exception as exc:
            callback_error = str(exc)
            raise

    turn.request_observer = request
    interrupted = None
    with (folder / 'generated.f32').open('xb') as raw, wave.open(str(folder / 'generated.wav'), 'wb') as preview:
        preview.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        def receive(audio):
            nonlocal samples, nonzero, callback_error
            try:
                frame = np.array(audio, dtype='<f4', copy=True)
                if frame.ndim != 1 or not np.isfinite(frame).all():
                    raise ValueError('Expected finite mono float32 generated audio')
                raw.write(frame.tobytes())
                raw.flush()
                os.fsync(raw.fileno())
                pcm = np.clip(np.rint(np.clip(frame, -1, 1) * 32768), -32768, 32767).astype('<i2')
                preview.writeframes(pcm.tobytes())
                samples += len(frame)
                nonzero = nonzero or bool(np.any(frame))
            except Exception as exc:
                callback_error = str(exc)
                raise
        try:
            client.chat_turn(data, turn, first=True, on_audio=receive)
        except BaseException as exc:
            turn.error = turn.error or str(exc) or type(exc).__name__
            if not isinstance(exc, Exception):
                interrupted = exc
        raw.flush()
        os.fsync(raw.fileno())
    # Refresh empty WAV header and sync the closed preview before final evidence.
    with (folder / 'generated.wav').open('r+b') as stream:
        stream.flush()
        os.fsync(stream.fileno())
    error = callback_error or turn.error or (None if requested else 'Client did not expose an actual request payload')
    status = ('error' if error else 'cancelled' if turn.cancel.is_set() else
              'length' if turn.truncated else 'empty' if not samples else
              'silent' if not nonzero else 'complete')
    _write(folder / 'model-text.txt', turn.text.encode('utf-8'))
    result = {'version': 1, 'id': item['id'], 'subset': item['subset'], 'status': status,
              'error': error, 'cancelled': turn.cancel.is_set(), 'finish_reason': turn.finish_reason,
              'first': True, 'request_observed': requested, 'generated_samples': samples,
              'generated_sha256': _digest((folder / 'generated.f32').read_bytes()),
              'preview_sha256': _digest((folder / 'generated.wav').read_bytes()),
              'model_text_sha256': _digest((folder / 'model-text.txt').read_bytes()),
              'input_sha256': _digest(data), 'duration_s': time.perf_counter() - started,
              'audio_format': {'dtype': '<f4', 'channels': 1, 'sample_rate': 24000},
              'text': turn.text, 'has_nonzero_samples': nonzero,
              'ttft': turn.ttft, 'client_audio_samples': turn.audio_samples}
    _json(folder / 'result.json', result)
    return result, interrupted


def infer_injected(prepared_dir, output_dir, *, client, mock=False, provenance):
    """Run explicit clients synchronously; never instantiate or select a model.

    Caller-supplied provenance is recorded, not independently verified. Real
    clients always acquire exclusive test ownership before creating outputs.
    No watchdog is added: transport timeouts/cancellation remain client duties.
    A hard process exit can leave an unfinished case, which exports as a miss.
    """
    real = isinstance(client, ServerClient)
    if mock and real or not mock and not real:
        raise ValueError('Use mock=True only for injected fixtures; real clients require test ownership')
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError('Explicit target or fixture provenance is required')
    _encoded(provenance)
    prepared, output = Path(prepared_dir).resolve(), Path(output_dir).resolve()
    manifest, validated = _prepared(prepared)
    source = (prepared / 'source-manifest.json').read_bytes()
    ownership = backend_session(client.base, test=True) if real else nullcontext()
    with ownership:
        output.mkdir(parents=True, exist_ok=False)
        _write(output / 'source-manifest.json', source)
        _json(output / 'run.json', {'version': 1, 'dataset': manifest['dataset'],
              'revision': manifest['revision'], 'cases': manifest['cases'],
              'source_manifest_sha256': _digest(source),
              'mock': mock, 'provenance': provenance, 'generated_audio_only': True})
        (output / 'cases').mkdir()
        _write(output / 'results.jsonl', b'')
        for item, data, _ in validated:
            folder = output / 'cases' / item['id']
            result, interrupted = _infer_case(folder, item, data, client)
            _append_index(output / 'results.jsonl', {'id': item['id'], 'status': result['status'],
                           'result_sha256': _digest((folder / 'result.json').read_bytes())})
            if interrupted is not None:
                raise interrupted
        _json(output / 'finished.json', {'attempted_cases': len(validated)})
    return output


def export_grades(run_dir, output_dir, *, asr_path=None):
    """Export complete denominators; no judge call or model-text ASR fallback."""
    run, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    manifest = _load(run / 'run.json')
    source = (run / 'source-manifest.json').read_bytes()
    original = json.loads(source)
    _records(original)
    if _digest(source) != manifest.get('source_manifest_sha256') or any(manifest.get(k) != original.get(k) for k in ('version', 'dataset', 'revision', 'cases')):
        raise ValueError('Run metadata differs from the frozen source manifest')
    expected = manifest['cases']
    ids = [_safe_id(item['id']) for item in expected]
    if len(set(x.casefold() for x in ids)) != len(ids):
        raise ValueError('Duplicate run case id')
    index = {}
    index_bytes = (run / 'results.jsonl').read_bytes()
    committed, _, incomplete_tail = index_bytes.rpartition(b'\n')
    # Newline is the append commit boundary. A hard exit may leave an arbitrary
    # trailing byte fragment, including a partial UTF-8 sequence. Keep it as
    # evidence, but never count it as a committed result.
    for line in committed.split(b'\n') if committed else []:
        record = json.loads(line)
        if record['id'] not in ids or record['id'] in index:
            raise ValueError('Invalid or duplicate result index entry')
        index[record['id']] = record
    results = {}
    for case_id in ids:
        folder = run / 'cases' / case_id
        path = folder / 'result.json'
        if not path.exists():
            if case_id in index:
                raise ValueError('Indexed result is missing')
            results[case_id] = {'status': 'unfinished'}
            continue
        data, result = path.read_bytes(), _load(path)
        if result['id'] != case_id or case_id in index and _digest(data) != index[case_id]['result_sha256']:
            raise ValueError('Result hash/id mismatch')
        for name, key in [('generated.f32', 'generated_sha256'), ('generated.wav', 'preview_sha256'),
                          ('model-text.txt', 'model_text_sha256'), ('input.wav', 'input_sha256')]:
            if _digest((folder / name).read_bytes()) != result[key]:
                raise ValueError('Result evidence hash mismatch: ' + name)
        results[case_id] = result
    asr, asr_provenance = {}, None
    if asr_path is not None:
        supplied = _load(asr_path)
        asr_provenance = supplied.get('provenance')
        if supplied.get('version') != 1 or not isinstance(asr_provenance, dict) or not all(isinstance(asr_provenance.get(k), str) and asr_provenance[k].strip() for k in ('engine', 'model')) or not _hash(asr_provenance.get('model_sha256')) or not isinstance(supplied.get('cases'), list):
            raise ValueError('ASR needs engine/model/weight-hash provenance and case list')
        for record in supplied['cases']:
            case_id = record.get('id')
            if case_id not in ids or case_id in asr or not isinstance(record.get('response'), str) or not _hash(record.get('generated_sha256')) or record['generated_sha256'] != results[case_id].get('generated_sha256'):
                raise ValueError('Invalid, duplicate, unknown or mismatched ASR case')
            asr[case_id] = record
    rows, roster = {}, []
    for item in expected:
        case_id = item['id']
        result = results[case_id]
        status = result['status']
        reason = status if status != 'complete' else None if case_id in index else 'unindexed'
        spoken_reason = reason or ('missing_asr' if case_id not in asr else 'empty_asr' if not asr[case_id]['response'].strip() else None)
        for source, failure in [('model-text', reason), ('spoken-asr', spoken_reason)]:
            row = dict(item['metadata'])
            row['response'] = '' if failure else result['text'] if source == 'model-text' else asr[case_id]['response']
            row['_localvoice'] = {'id': case_id, 'status': status, 'source': source,
                                 'eligible': failure is None, 'reason': failure}
            rows.setdefault(item['subset'] + '-' + source + '.jsonl', []).append(row)
        roster.append({'id': case_id, 'status': status, 'model_text_reason': reason,
                       'spoken_asr_reason': spoken_reason, 'asr': asr.get(case_id)})
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'result-index.raw', index_bytes)
    for name, records in rows.items():
        _write(output / name, ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in records).encode('utf-8'))
    _json(output / 'roster.json', {'version': 1, 'dataset': manifest['dataset'], 'revision': manifest['revision'],
          'expected_cases': len(expected), 'cases': roster, 'mock': manifest['mock'],
          'index_incomplete_tail_bytes': len(incomplete_tail),
          'asr_provenance': asr_provenance, 'policy': 'All rows retained; ineligible responses blank; no implicit ASR or judge.'})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('manifest', type=Path)
    prep.add_argument('output', type=Path)
    export = commands.add_parser('export')
    export.add_argument('run', type=Path)
    export.add_argument('output', type=Path)
    export.add_argument('--asr', type=Path)
    args = parser.parse_args(argv)
    print(prepare(args.manifest, args.output) if args.command == 'prepare' else
          export_grades(args.run, args.output, asr_path=args.asr))


if __name__ == '__main__':
    main()
