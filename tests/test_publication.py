"""Publication guard contracts without using real credentials or recordings."""
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'publication', Path(__file__).resolve().parents[1] / 'scripts/check_publication.py')
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


def test_private_files_are_rejected_even_when_tracked():
    for name in ['outputs/result.json', 'data/example.json', 'AGENTS.md',
                 '.env', '.env.local', 'recording.wav', 'docs/AGENT_HANDOFF.md']:
        assert publication.inspect_file(name, b'private fixture')


def test_provider_pattern_flags_without_echoing_value():
    fixture = ('gh' + 'p_' + 'A' * 36).encode()
    reasons = publication.inspect_file('config.py', fixture)
    assert any('provider credential' in reason for reason in reasons)
    assert all(fixture.decode() not in reason for reason in reasons)


def test_public_source_and_docs_pass():
    for name in ['src/app.py', 'README.md', '.env.example', 'docs/PRIVACY.md',
                 'docs/RESULTS.md', 'docs/EVALUATION.md', 'docs/THIRD_PARTY.md',
                 'research/README.md', 'research/examples/README.md']:
        assert not publication.inspect_file(name, b'Public example with no credentials.')


def test_unapproved_research_audio_remains_blocked_by_default():
    reasons = publication.inspect_file('research/examples/pass-reply.wav', b'not a wave')
    assert any('private/generated file' in reason for reason in reasons)
    assert any('lacks matching manifest approval' in reason for reason in reasons)


@pytest.mark.parametrize('value,valid', [('/srv/test-user\n', True), ('relative', False), ('/srv/a\n/srv/b', False)])
def test_wsl_home_validates_without_contacting_a_real_runtime(monkeypatch, value, valid):
    import sys
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from synth import native_abort_probe
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=value)

    monkeypatch.setattr(native_abort_probe.subprocess, 'run', run)
    if valid:
        assert native_abort_probe.wsl_home() == value.strip()
    else:
        with pytest.raises(ValueError):
            native_abort_probe.wsl_home()
    assert calls == [['wsl.exe', '-e', 'printenv', 'HOME']]
