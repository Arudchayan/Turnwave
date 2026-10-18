"""Check publication hazards in the staged index or an explicit worktree scan.

The normal check reads Git index blobs. ``--worktree`` reads tracked files plus
non-ignored untracked files so a reviewed publication bundle can be checked
before it is staged. The research-media exception is deliberately limited to
the exact paths in PUBLIC_RESEARCH_ASSETS and requires a matching version-1
manifest entry. This is a heuristic guard, not a complete content classifier.
"""
import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys
import wave
import zlib
from io import BytesIO
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree


PUBLIC_DOCS = {
    'ARCHITECTURE.md', 'KNOWN_LIMITATIONS.md', 'PRIVACY.md',
    'PUBLICATION.md', 'RECORDED_SESSIONS.md', 'RESULTS.md',
    'EVALUATION.md', 'THIRD_PARTY.md',
}
PUBLIC_RESEARCH_DOCS = {
    'research/README.md', 'research/examples/README.md',
}
PUBLICATION_MANIFEST = 'research/publication-manifest.json'
PUBLIC_RESEARCH_ASSETS = {
    'docs/assets/completion-screen.svg',
    'docs/assets/completion-screen.png',
    'docs/assets/short-turn-delay.svg',
    'docs/assets/short-turn-delay.png',
    'research/results.json',
    'research/short-turn-cases.json',
    'research/candidate-screen-cases.json',
    'research/examples/examples.json',
    'research/examples/pass-reply.wav',
    'research/examples/failure-reply.wav',
}
ASSET_MANIFEST_FIELDS = {
    'path', 'sha256', 'bytes', 'kind', 'source_type',
    'contains_human_recording', 'whole_reply',
}
PRIVATE_ROOTS = {
    'outputs', 'data', 'models', '.slim', '.desloppify',
    '.venv', '.venv-eval', 'public-release',
}
PRIVATE_SUFFIXES = {
    '.wav', '.mp3', '.flac', '.ogg', '.m4a', '.f32', '.gguf',
    '.pem', '.key', '.p12', '.pfx', '.token', '.bundle', '.log',
}
PATTERNS = {
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'provider credential': re.compile(
        r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}'
        r'|sk-[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{25,})'),
    'literal credential assignment': re.compile(
        r'''(?i)(?:api[_-]?key|access[_-]?token|password|secret)["']?\s*[:=]\s*["'][^"'\r\n]{8,}["']'''),
    'personal absolute path': re.compile(
        r'(?:[A-Z]:[\\/](?:Users|Repos)[\\/][\w.-]+|/home/[a-zA-Z0-9_.-]+/)'),
}
SVG_FORBIDDEN = {
    'script': re.compile(r'<\s*script\b', re.IGNORECASE),
    'foreignObject': re.compile(r'<\s*foreignObject\b', re.IGNORECASE),
    'external reference': re.compile(
        r'''(?:href\s*=\s*["']\s*(?:https?:|//|data:)|@import\b|url\(\s*["']?\s*(?:https?:|//|data:))''',
        re.IGNORECASE),
}
PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_TEXT_BYTES = 1024 * 1024
MAX_PNG_BYTES = 1024 * 1024
MAX_WAV_BYTES = 4 * 1024 * 1024
MAX_WAV_SECONDS = 60


def _safe_relative_path(name):
    """Return a canonical POSIX relative path, or None for escape/alias forms."""
    if not isinstance(name, str) or not name or '\\' in name or '\x00' in name:
        return None
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {'', '.', '..'} for part in path.parts):
        return None
    normalized = path.as_posix()
    return normalized if normalized == name else None


def _text_findings(name, data):
    reasons = []
    try:
        text = data.decode('utf-8-sig')
    except UnicodeError:
        return None, ['non-text content requires explicit review']
    if '\0' in text:
        reasons.append('binary content requires explicit review')
    for label, pattern in PATTERNS.items():
        for number, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                reasons.append(f'{label} at line {number}')
    return text, reasons


def _asset_type_and_size(name, data):
    """Validate structural type/size constraints for an approved asset."""
    suffix = PurePosixPath(name).suffix.lower()
    if suffix == '.wav':
        if len(data) > MAX_WAV_BYTES:
            return ['generated audio exceeds 4 MiB']
        if len(data) < 44 or data[:4] != b'RIFF' or data[8:12] != b'WAVE':
            return ['generated audio is not a RIFF/WAVE file']
        if struct.unpack('<I', data[4:8])[0] != len(data) - 8:
            return ['WAV RIFF length does not match file length']
        try:
            with wave.open(BytesIO(data), 'rb') as wav:
                if wav.getcomptype() != 'NONE':
                    return ['generated audio is not uncompressed PCM']
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 24000):
                    return ['generated audio must be mono PCM16 at 24 kHz']
                if wav.getnframes() <= 0 or wav.getnframes() > 24000 * MAX_WAV_SECONDS:
                    return ['generated audio must be nonempty and no longer than 60 seconds']
        except (wave.Error, EOFError, OSError, struct.error):
            return ['generated audio WAV structure is invalid']
        return []
    if suffix == '.png':
        if len(data) > MAX_PNG_BYTES:
            return ['PNG exceeds 1 MiB']
        if len(data) < 33 or data[:8] != PNG_SIGNATURE or data[12:16] != b'IHDR':
            return ['PNG signature or IHDR header is invalid']
        position = 8
        saw_ihdr = saw_idat = saw_iend = False
        while position < len(data):
            if position + 12 > len(data):
                return ['PNG chunk is truncated']
            chunk_length = struct.unpack('>I', data[position:position + 4])[0]
            chunk_type = data[position + 4:position + 8]
            chunk_end = position + 12 + chunk_length
            if chunk_end > len(data):
                return ['PNG chunk length exceeds file bounds']
            chunk_data = data[position + 8:position + 8 + chunk_length]
            expected_crc = struct.unpack('>I', data[position + 8 + chunk_length:chunk_end])[0]
            if zlib.crc32(chunk_type + chunk_data) & 0xffffffff != expected_crc:
                return ['PNG chunk CRC is invalid']
            if not saw_ihdr:
                if chunk_type != b'IHDR' or chunk_length != 13:
                    return ['PNG must begin with one 13-byte IHDR chunk']
                width, height = struct.unpack('>II', chunk_data[:8])
                if width == 0 or height == 0:
                    return ['PNG dimensions must be positive']
                saw_ihdr = True
            elif chunk_type == b'IHDR':
                return ['PNG contains duplicate IHDR']
            if chunk_type == b'IDAT':
                saw_idat = True
            if chunk_type == b'IEND':
                if chunk_length != 0 or chunk_end != len(data):
                    return ['PNG IEND must be empty and terminate the file']
                saw_iend = True
                break
            position = chunk_end
        if not (saw_ihdr and saw_idat and saw_iend):
            return ['PNG must contain IHDR, IDAT and terminating IEND chunks']
        return []
    if suffix in {'.svg', '.json'}:
        if len(data) > MAX_TEXT_BYTES:
            return ['reviewed text asset exceeds 1 MiB']
        return []
    return ['asset extension is not an allowed reviewed type']


def _validate_svg(name, text):
    reasons = []
    for label, pattern in SVG_FORBIDDEN.items():
        if pattern.search(text):
            reasons.append(f'SVG contains forbidden {label}')
    try:
        root = ElementTree.fromstring(text)
        if root.tag.rsplit('}', 1)[-1].lower() != 'svg':
            reasons.append('SVG root element is not <svg>')
    except ElementTree.ParseError:
        reasons.append('SVG XML is malformed')
    return reasons


def validate_manifest(manifest_bytes, files):
    """Validate the one exact publication manifest against an in-memory file map.

    ``files`` maps canonical repo-relative paths to their bytes. It is deliberately
    supplied by the caller so index and worktree validation use identical checks.
    """
    errors = []
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        return {}, ['publication manifest exceeds 1 MiB']
    try:
        manifest = json.loads(manifest_bytes.decode('utf-8-sig'))
    except (UnicodeError, json.JSONDecodeError):
        return {}, ['publication manifest is not valid UTF-8 JSON']
    if not isinstance(manifest, dict) or set(manifest) != {'schema_version', 'assets'}:
        return {}, ['publication manifest must contain only schema_version and assets']
    if type(manifest['schema_version']) is not int or manifest['schema_version'] != 1:
        errors.append('publication manifest schema_version must be 1')
    entries = manifest['assets']
    if not isinstance(entries, list):
        return {}, errors + ['publication manifest assets must be an array']

    approved = {}
    for index, entry in enumerate(entries):
        prefix = f'publication manifest asset {index}'
        if not isinstance(entry, dict) or set(entry) != ASSET_MANIFEST_FIELDS:
            errors.append(f'{prefix} must use the exact version-1 asset fields')
            continue
        name = entry.get('path')
        canonical = _safe_relative_path(name)
        if canonical is None:
            errors.append(f'{prefix} path is not a normalized relative path')
            continue
        if canonical not in PUBLIC_RESEARCH_ASSETS:
            errors.append(f'{prefix} path is not on the exact reviewed asset allowlist')
            continue
        if canonical in approved:
            errors.append(f'{prefix} duplicates a manifest path')
            continue
        digest = entry.get('sha256')
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            errors.append(f'{prefix} sha256 must be 64 lowercase hexadecimal characters')
            continue
        size = entry.get('bytes')
        if type(size) is not int or size < 0:
            errors.append(f'{prefix} bytes must be a nonnegative integer')
            continue
        suffix = PurePosixPath(canonical).suffix.lower()
        expected_kind = {
            '.svg': 'chart_svg', '.png': 'chart_png', '.json': 'evidence_json',
            '.wav': 'generated_audio',
        }.get(suffix)
        if entry.get('kind') != expected_kind:
            errors.append(f'{prefix} kind does not match the asset type')
            continue
        source_type = entry.get('source_type')
        if not isinstance(source_type, str) or not source_type.strip() or len(source_type) > 80:
            errors.append(f'{prefix} source_type must be a short nonempty string')
            continue
        if entry.get('contains_human_recording') is not False:
            errors.append(f'{prefix} must declare contains_human_recording=false')
            continue
        is_audio = suffix == '.wav'
        if entry.get('whole_reply') is not is_audio:
            errors.append(f'{prefix} whole_reply must be true only for generated audio')
            continue
        if canonical not in files:
            errors.append(f'{prefix} file is missing from the selected snapshot')
            continue
        data = files[canonical]
        if size != len(data):
            errors.append(f'{prefix} byte count does not match asset')
            continue
        if digest != hashlib.sha256(data).hexdigest():
            errors.append(f'{prefix} SHA-256 does not match asset')
            continue
        type_errors = _asset_type_and_size(canonical, data)
        if type_errors:
            errors.extend(f'{prefix}: {reason}' for reason in type_errors)
            continue
        if suffix == '.svg':
            text, text_errors = _text_findings(canonical, data)
            if text_errors:
                errors.extend(f'{prefix}: {reason}' for reason in text_errors)
                continue
            svg_errors = _validate_svg(canonical, text)
            if svg_errors:
                errors.extend(f'{prefix}: {reason}' for reason in svg_errors)
                continue
        if suffix == '.json':
            try:
                json.loads(data.decode('utf-8-sig'))
            except (UnicodeError, json.JSONDecodeError):
                errors.append(f'{prefix} evidence JSON is invalid')
                continue
            _, text_errors = _text_findings(canonical, data)
            if text_errors:
                errors.extend(f'{prefix}: {reason}' for reason in text_errors)
                continue
        approved[canonical] = entry

    # Any malformed or mismatched entry invalidates the entire manifest. This
    # keeps a mixed good/bad roster from approving its valid subset by accident.
    return (approved if not errors else {}), errors


def inspect_file(name, data, approved_asset=None):
    """Inspect one file. Asset approval must be explicitly supplied by manifest validation."""
    canonical = _safe_relative_path(name)
    path = PurePosixPath(name) if canonical is None else PurePosixPath(canonical)
    reasons = []
    if canonical is None:
        reasons.append('path is not a normalized relative path')
    is_approved_asset = (
        canonical in PUBLIC_RESEARCH_ASSETS and isinstance(approved_asset, dict)
        and approved_asset.get('path') == canonical
    )
    is_private_root = bool(path.parts and path.parts[0] in PRIVATE_ROOTS)
    if (is_private_root or '__pycache__' in path.parts
            or path.name in {'AGENTS.md', '.ignore', 'scorecard.png'}
            or path.name == '.env' or path.name.startswith('.env.') and path.name != '.env.example'
            or path.suffix.lower() in PRIVATE_SUFFIXES and not is_approved_asset):
        reasons.append('private/generated file')
    if path.parts and path.parts[0] == 'docs':
        if (len(path.parts) == 2 and path.name not in PUBLIC_DOCS
                or len(path.parts) >= 3 and canonical not in PUBLIC_RESEARCH_ASSETS):
            reasons.append('internal document')
    if path.parts and path.parts[0] == 'research':
        allowed_research = (
            canonical in PUBLIC_RESEARCH_DOCS
            or canonical == PUBLICATION_MANIFEST
            or canonical in PUBLIC_RESEARCH_ASSETS
        )
        if not allowed_research:
            reasons.append('unreviewed research artifact')
    if canonical in PUBLIC_RESEARCH_ASSETS and not is_approved_asset:
        reasons.append('reviewed research asset lacks matching manifest approval')
    if canonical == PUBLICATION_MANIFEST:
        if len(data) > MAX_MANIFEST_BYTES:
            reasons.append('publication manifest exceeds 1 MiB')
        try:
            json.loads(data.decode('utf-8-sig'))
        except (UnicodeError, json.JSONDecodeError):
            reasons.append('publication manifest is not valid UTF-8 JSON')
    size_limit = MAX_WAV_BYTES if path.suffix.lower() == '.wav' else MAX_TEXT_BYTES
    if len(data) > size_limit:
        reasons.append(f'file exceeds {size_limit // (1024 * 1024)} MiB; review before publication')
    text, text_reasons = _text_findings(name, data)
    if text is None:
        if not is_approved_asset or path.suffix.lower() not in {'.wav', '.png'}:
            reasons.extend(text_reasons)
    else:
        reasons.extend(text_reasons)
        if path.suffix.lower() == '.svg' and is_approved_asset:
            reasons.extend(_validate_svg(name, text))
        if path.suffix.lower() == '.png' and is_approved_asset:
            reasons.extend(_asset_type_and_size(canonical, data))
    if is_approved_asset:
        reasons.extend(_asset_type_and_size(canonical, data))
    return list(dict.fromkeys(reasons))


def _git_names(args):
    raw = subprocess.check_output(args)
    return [name.decode('utf-8') for name in raw.split(b'\0') if name]


def _snapshot_files(worktree, root):
    tracked = _git_names(['git', 'ls-files', '-z', '--cached'])
    if worktree:
        untracked = _git_names(['git', 'ls-files', '-z', '--others', '--exclude-standard'])
        names = sorted(set(tracked) | set(untracked))
    else:
        names = tracked
    files = {}
    errors = []
    for name in names:
        canonical = _safe_relative_path(name)
        if canonical is None:
            errors.append(f'{name}: path is not normalized')
            continue
        if worktree:
            target = root / Path(*PurePosixPath(canonical).parts)
            try:
                components = [root]
                current = root
                for part in PurePosixPath(canonical).parts:
                    current = current / part
                    components.append(current)
                if any(component.is_symlink() for component in components[1:]):
                    errors.append(f'{canonical}: symlink publication path is not allowed')
                    continue
                resolved = target.resolve(strict=True)
                resolved.relative_to(root.resolve())
                files[canonical] = resolved.read_bytes()
            except (OSError, ValueError):
                errors.append(f'{canonical}: worktree file is missing or escapes repository')
        else:
            try:
                files[canonical] = subprocess.check_output(['git', 'show', ':' + canonical])
            except subprocess.CalledProcessError:
                errors.append(f'{canonical}: index blob could not be read')
    return files, errors


def inspect_snapshot(files):
    failures = []
    manifest_data = files.get(PUBLICATION_MANIFEST)
    asset_files_present = set(files) & PUBLIC_RESEARCH_ASSETS
    approved = {}
    if manifest_data is not None:
        approved, manifest_errors = validate_manifest(manifest_data, files)
        failures.extend(f'{PUBLICATION_MANIFEST}: {reason}' for reason in manifest_errors)
    elif asset_files_present:
        failures.append(f'{PUBLICATION_MANIFEST}: required when reviewed assets are present')
    for name, data in files.items():
        findings = inspect_file(name, data, approved.get(name))
        failures.extend(f'{name}: {reason}' for reason in findings)
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--worktree', action='store_true',
        help='inspect tracked working files plus non-ignored untracked files (default: Git index blobs)',
    )
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    files, snapshot_errors = _snapshot_files(args.worktree, root)
    failures = snapshot_errors + inspect_snapshot(files)
    if failures:
        print('\n'.join(failures))
        print(f'Publication check failed: {len(failures)} finding(s). Values redacted.')
        return 1
    mode = 'worktree' if args.worktree else 'index'
    print(f'Publication {mode} check passed for {len(files)} files.')
    print('Review history, metadata and content manually; this is a heuristic guard.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
