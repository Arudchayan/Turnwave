"""Focused tests for the narrow reviewed-research publication exception."""
import hashlib
import importlib.util
import io
import json
import struct
import wave
import zlib
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'publication_research_assets',
    Path(__file__).resolve().parents[1] / 'scripts/check_publication.py',
)
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


WAV_PATH = 'research/examples/pass-reply.wav'


def wav_bytes(rate=24000, seconds=1):
    out = io.BytesIO()
    with wave.open(out, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(bytes(rate * seconds * 2))
    return out.getvalue()


def manifest_entry(path, data, **overrides):
    suffix = Path(path).suffix
    entry = {
        'path': path,
        'sha256': hashlib.sha256(data).hexdigest(),
        'bytes': len(data),
        'kind': {'.wav': 'generated_audio', '.json': 'evidence_json',
                 '.svg': 'chart_svg', '.png': 'chart_png'}[suffix],
        'source_type': 'authored_synthetic_model_output',
        'contains_human_recording': False,
        'whole_reply': suffix == '.wav',
    }
    entry.update(overrides)
    return entry


def manifest_bytes(*entries):
    return json.dumps({'schema_version': 1, 'assets': list(entries)}).encode()


def png_chunk(kind, payload):
    body = kind + payload
    return struct.pack('>I', len(payload)) + body + struct.pack('>I', zlib.crc32(body) & 0xffffffff)


def png_bytes():
    return (publication.PNG_SIGNATURE
            + png_chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 6, 0, 0, 0))
            + png_chunk(b'IDAT', zlib.compress(b'\x00\x00\x00\x00\xff\xff'))
            + png_chunk(b'IEND', b''))


def test_exact_manifest_approval_allows_only_valid_whole_reply_wav():
    audio = wav_bytes()
    manifest = manifest_bytes(manifest_entry(WAV_PATH, audio))
    approved, errors = publication.validate_manifest(
        manifest, {WAV_PATH: audio, publication.PUBLICATION_MANIFEST: manifest})

    assert errors == []
    assert WAV_PATH in approved
    assert publication.inspect_file(WAV_PATH, audio, approved[WAV_PATH]) == []
    # Existing callers get no implicit exception: the explicit manifest entry is required.
    assert publication.inspect_file(WAV_PATH, audio)


def test_hash_and_size_mismatches_do_not_approve_audio():
    audio = wav_bytes()
    entry = manifest_entry(WAV_PATH, audio, sha256='0' * 64, bytes=len(audio) + 1)
    approved, errors = publication.validate_manifest(
        manifest_bytes(entry), {WAV_PATH: audio})
    assert not approved
    assert any('byte count' in error or 'SHA-256' in error for error in errors)


@pytest.mark.parametrize('path', [
    '../outputs/private.wav', 'outputs/private.wav', 'data/capture.wav',
    'research/examples/unknown.wav', 'research/../../outputs/private.wav',
])
def test_manifest_cannot_approve_unlisted_or_private_paths(path):
    audio = wav_bytes()
    entry = manifest_entry(path, audio)
    approved, errors = publication.validate_manifest(
        manifest_bytes(entry), {path: audio})
    assert not approved
    assert errors


def test_manifest_cannot_approve_duplicate_or_human_recording():
    audio = wav_bytes()
    good = manifest_entry(WAV_PATH, audio)
    duplicate = manifest_entry(WAV_PATH, audio)
    approved, errors = publication.validate_manifest(
        manifest_bytes(good, duplicate), {WAV_PATH: audio})
    assert not approved
    assert any('duplicates' in error for error in errors)

    human = manifest_entry(WAV_PATH, audio, contains_human_recording=True)
    approved, errors = publication.validate_manifest(
        manifest_bytes(human), {WAV_PATH: audio})
    assert not approved
    assert any('contains_human_recording=false' in error for error in errors)


def test_audio_must_be_unmodified_format_and_within_duration_limit():
    for invalid in (wav_bytes(rate=16000), wav_bytes(seconds=61)):
        approved, errors = publication.validate_manifest(
            manifest_bytes(manifest_entry(WAV_PATH, invalid)), {WAV_PATH: invalid})
        assert not approved
        assert errors


def test_evidence_json_is_parsed_and_secret_scanned():
    path = 'research/results.json'
    data = json.dumps({'credential': 'ghp_' + 'A' * 36}).encode()
    approved, errors = publication.validate_manifest(
        manifest_bytes(manifest_entry(path, data)), {path: data})
    assert not approved
    assert any('provider credential' in error for error in errors)
    assert 'ghp_' not in '\n'.join(errors)

    password_key = 'pass' + 'word'
    password_data = ('{"' + password_key + '": "long-not-a-real-value"}').encode()
    approved, errors = publication.validate_manifest(
        manifest_bytes(manifest_entry(path, password_data)), {path: password_data})
    assert not approved
    assert any('literal credential assignment' in error for error in errors)


@pytest.mark.parametrize('svg,expected', [
    (b'<svg xmlns="http://www.w3.org/2000/svg"></svg>', None),
    (b'<svg><script>alert(1)</script></svg>', 'forbidden script'),
    (b'<svg><foreignObject /></svg>', 'forbidden foreignObject'),
    (b'<svg><image href="https://example.test/x" /></svg>', 'external reference'),
])
def test_svg_requires_text_scan_and_rejects_active_or_external_content(svg, expected):
    path = 'docs/assets/completion-screen.svg'
    approved, errors = publication.validate_manifest(
        manifest_bytes(manifest_entry(path, svg)), {path: svg})
    if expected is None:
        assert errors == []
        assert path in approved
    else:
        assert not approved
        assert any(expected in error for error in errors)


def test_snapshot_rejects_unmanifested_unknown_research_files():
    assert any('unreviewed research artifact' in error for error in
               publication.inspect_snapshot({'research/raw-log.json': b'{}'}))


def test_snapshot_requires_manifest_when_allowlisted_assets_are_present():
    audio = wav_bytes()
    errors = publication.inspect_snapshot({WAV_PATH: audio})
    assert any(publication.PUBLICATION_MANIFEST in error for error in errors)


def test_exact_chart_asset_path_still_requires_manifest_hash_approval():
    path = 'docs/assets/completion-screen.svg'
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'
    assert any('lacks matching manifest approval' in error for error in
               publication.inspect_file(path, svg))
    manifest = manifest_bytes()
    errors = publication.inspect_snapshot({path: svg, publication.PUBLICATION_MANIFEST: manifest})
    assert any('lacks matching manifest approval' in error for error in errors)


def test_png_is_crc_checked_and_must_have_complete_chunk_sequence():
    path = 'docs/assets/completion-screen.png'
    image = png_bytes()
    approved, errors = publication.validate_manifest(
        manifest_bytes(manifest_entry(path, image)), {path: image})
    assert errors == []
    assert path in approved

    corrupted = image[:-1] + bytes([image[-1] ^ 1])
    approved, errors = publication.validate_manifest(
        manifest_bytes(manifest_entry(path, corrupted)), {path: corrupted})
    assert not approved
    assert any('CRC' in error or 'IEND' in error for error in errors)


def test_manifest_has_a_fixed_schema_version_and_no_unknown_top_level_fields():
    for manifest in (
        b'{"schema_version":2,"assets":[]}',
        b'{"schema_version":1,"assets":[],"approve_all":true}',
        b'{"schema_version":1,"assets":{}}',
    ):
        _, errors = publication.validate_manifest(manifest, {})
        assert errors


def test_explicit_worktree_snapshot_includes_nonignored_untracked_research_files(
        tmp_path, monkeypatch):
    evidence = b'{"scope":"aggregate synthetic evidence"}'
    manifest = manifest_bytes(manifest_entry('research/results.json', evidence))
    (tmp_path / 'research').mkdir()
    (tmp_path / 'research' / 'publication-manifest.json').write_bytes(manifest)
    (tmp_path / 'research' / 'README.md').write_text('Reviewed public research index.', encoding='utf-8')
    (tmp_path / 'research' / 'results.json').write_bytes(evidence)

    def git_output(command):
        if command == ['git', 'ls-files', '-z', '--cached']:
            return b'research/publication-manifest.json\0research/README.md\0'
        if command == ['git', 'ls-files', '-z', '--others', '--exclude-standard']:
            return b'research/results.json\0'
        raise AssertionError(f'unexpected subprocess command: {command!r}')

    monkeypatch.setattr(publication.subprocess, 'check_output', git_output)
    files, errors = publication._snapshot_files(worktree=True, root=tmp_path)
    assert errors == []
    assert set(files) == {
        'research/publication-manifest.json', 'research/README.md', 'research/results.json'}
    assert publication.inspect_snapshot(files) == []
