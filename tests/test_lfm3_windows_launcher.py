"""Check the Windows launcher's artifact selection without starting a server."""

from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'src/server_lfm3_windows.ps1'


@pytest.fixture
def powershell():
    executable = shutil.which('pwsh') or shutil.which('powershell')
    if executable is None:
        pytest.skip('PowerShell is not installed')
    return executable


def test_custom_research_model_requires_explicit_projector(powershell, tmp_path):
    model = tmp_path / 'model.gguf'
    model.touch()
    result = subprocess.run([powershell, '-NoProfile', '-NonInteractive', '-File', str(SCRIPT),
                             '-Variant', 'research', '-Model', str(model)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert 'Specify -Projector explicitly' in result.stderr


def test_research_variant_requires_explicit_model(powershell):
    result = subprocess.run([powershell, '-NoProfile', '-NonInteractive', '-File', str(SCRIPT),
                             '-Variant', 'research'], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert 'Specify -Model explicitly' in result.stderr


def test_missing_runner_fails_before_starting(powershell, tmp_path):
    missing = tmp_path / 'missing.exe'
    result = subprocess.run([powershell, '-NoProfile', '-NonInteractive', '-File', str(SCRIPT),
                             '-Runner', str(missing)], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert 'Missing LFM3 artifact' in result.stderr


@pytest.mark.skipif(__import__('sys').platform != 'win32', reason='Windows batch runner fixture')
def test_explicit_research_artifacts_reach_runner(powershell, tmp_path, monkeypatch):
    runner = tmp_path / 'fake-runner.cmd'
    runner.write_text('@echo off\r\necho %* > "%TURNWAVE_CAPTURE%"\r\n', encoding='ascii')
    model = tmp_path / 'research.gguf'
    projector = tmp_path / 'projector.gguf'
    model.touch()
    projector.touch()
    capture = tmp_path / 'args.txt'
    monkeypatch.setenv('TURNWAVE_CAPTURE', str(capture))
    result = subprocess.run([powershell, '-NoProfile', '-NonInteractive', '-File', str(SCRIPT),
                             '-Variant', 'research', '-Runner', str(runner), '-Model', str(model),
                             '-Projector', str(projector), '-Port', '19081'],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    args = capture.read_text(encoding='utf-8')
    assert str(model) in args and str(projector) in args
    assert '--port 19081' in args
