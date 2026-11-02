"""Exercise variant selection without loading a model or occupying a GPU."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


def _bash():
    if os.name == 'nt':
        path = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Git/bin/bash.exe'
        return str(path) if path.exists() else None
    return shutil.which('bash')


def _posix(path):
    value = str(path).replace('\\', '/')
    return '/' + value[0].lower() + value[2:] if len(value) > 1 and value[1] == ':' else value


@pytest.fixture
def launcher(tmp_path):
    bash = _bash()
    if not bash:
        pytest.skip('bash is required to exercise the WSL launcher')
    runner = tmp_path / 'runner'
    runner.mkdir()
    executable = runner / 'llama-server'
    executable.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n', encoding='utf-8')
    executable.chmod(0o755)
    models = tmp_path / 'standard'
    models.mkdir()
    for name in ('LFM3-Omni-2B-Q4_K_M.gguf', 'mmproj-LFM3-Omni-2B-Q4_K_M.gguf'):
        (models / name).touch()
    script = Path(__file__).resolve().parents[1] / 'src/server_lfm3_wsl.sh'
    env = os.environ.copy()
    for key in ('LOCALVOICE_LFM3_VARIANT', 'LOCALVOICE_LFM3_RESEARCH_MODEL',
                'LOCALVOICE_LFM3_RESEARCH_PROJECTOR'):
        env.pop(key, None)
    env.update(LOCALVOICE_LFM3_RUNNER=_posix(runner), LOCALVOICE_LFM3_MODEL_DIR=_posix(models))

    def run(**updates):
        return subprocess.run([bash, _posix(script)], env={**env, **updates},
                              capture_output=True, text=True, timeout=15)
    return run, tmp_path


def test_standard_variant_is_default(launcher):
    run, _ = launcher
    result = run()
    assert result.returncode == 0, result.stderr
    assert 'LFM3-Omni-2B-Q4_K_M.gguf' in result.stdout
    assert 'mmproj-LFM3-Omni-2B-Q4_K_M.gguf' in result.stdout


def test_research_variant_requires_both_artifacts_and_selects_them(launcher):
    run, root = launcher
    missing = run(LOCALVOICE_LFM3_VARIANT='research')
    assert missing.returncode != 0
    assert 'LOCALVOICE_LFM3_RESEARCH_MODEL' in missing.stderr
    model, projector = root / 'research.gguf', root / 'projector.gguf'
    model.touch()
    projector.touch()
    result = run(LOCALVOICE_LFM3_VARIANT='research', LOCALVOICE_LFM3_RESEARCH_MODEL=_posix(model),
                 LOCALVOICE_LFM3_RESEARCH_PROJECTOR=_posix(projector))
    assert result.returncode == 0, result.stderr
    assert _posix(model) in result.stdout
    assert _posix(projector) in result.stdout


def test_unknown_variant_fails_closed(launcher):
    run, _ = launcher
    result = run(LOCALVOICE_LFM3_VARIANT='unknown')
    assert result.returncode != 0
    assert 'Unknown LFM3 variant' in result.stderr
