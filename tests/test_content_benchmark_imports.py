"""Public entry points must work in fresh interpreters without PYTHONPATH."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('arguments', [
    ['-m', 'src.synth.content_benchmark', '--help'],
    ['-c', 'from src.synth.content_benchmark import prepare, infer_injected, export_grades'],
])
def test_package_entry_points_do_not_need_pythonpath(arguments):
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, *arguments], cwd=ROOT,
                            env=environment, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
