"""The research runner must not execute arbitrary checkpoint code by default."""

import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/heretic_lfm3.py'


def test_checkpoint_code_requires_explicit_trust(tmp_path):
    (tmp_path / 'config.json').write_text('{"model_type":"lfm3_omni"}', encoding='utf-8')
    result = subprocess.run([sys.executable, str(SCRIPT), '--model-dir', str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert '--trust-model-code' in result.stderr


def test_other_architectures_are_rejected(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'other'}), encoding='utf-8')
    result = subprocess.run([sys.executable, str(SCRIPT), '--model-dir', str(tmp_path),
                             '--trust-model-code'], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'expected an LFM3 Omni checkpoint' in result.stderr
