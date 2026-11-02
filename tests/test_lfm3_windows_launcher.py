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


def test_missing_runner_fails_before_starting(powershell, tmp_path):
    missing = tmp_path / 'missing.exe'
    result = subprocess.run([powershell, '-NoProfile', '-NonInteractive', '-File', str(SCRIPT),
                             '-Runner', str(missing)], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert 'Missing LFM3 artifact' in result.stderr
