"""Check the Git index for selected publication/privacy hazards.

Print categories and locations only, never the matching sensitive value.
This intentionally checks staged/tracked bytes, not ignored local experiments.
It is a heuristic guard, not a complete secret or content classifier.
"""
import re
import subprocess
import sys
from pathlib import PurePosixPath


PUBLIC_DOCS = {
    'ARCHITECTURE.md', 'KNOWN_LIMITATIONS.md', 'PRIVACY.md',
    'PUBLICATION.md', 'RECORDED_SESSIONS.md',
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
        r'''(?i)(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*["'][^"'\r\n]{8,}["']'''),
    'personal absolute path': re.compile(
        r'(?:[A-Z]:[\\/](?:Users|Repos)[\\/][\w.-]+|/home/[a-zA-Z0-9_.-]+/)'),
}


def inspect_file(name, data):
    path = PurePosixPath(name)
    reasons = []
    if (path.parts[0] in PRIVATE_ROOTS or '__pycache__' in path.parts
            or path.name in {'AGENTS.md', '.ignore', 'scorecard.png'}
            or path.name == '.env' or path.name.startswith('.env.') and path.name != '.env.example'
            or path.suffix.lower() in PRIVATE_SUFFIXES):
        reasons.append('private/generated file')
    if path.parts[0] == 'docs' and (len(path.parts) != 2 or path.name not in PUBLIC_DOCS):
        reasons.append('internal document')
    if len(data) > 1024 * 1024:
        reasons.append('file exceeds 1 MiB; review before publication')
    try:
        text = data.decode('utf-8-sig')
    except UnicodeError:
        reasons.append('non-text content requires explicit review')
        return reasons
    if '\0' in text:
        reasons.append('binary content requires explicit review')
    for label, pattern in PATTERNS.items():
        for number, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                reasons.append(f'{label} at line {number}')
    return reasons


def main():
    names = subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0')
    failures = []
    for name in filter(None, names):
        data = subprocess.check_output(['git', 'show', ':' + name])
        failures.extend(f'{name}: {reason}' for reason in inspect_file(name, data))
    if failures:
        print('\n'.join(failures))
        print(f'Publication check failed: {len(failures)} finding(s). Values redacted.')
        return 1
    print(f'Publication index check passed for {len(list(filter(None, names)))} files.')
    print('Review history, metadata and content manually; this is a heuristic check.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
