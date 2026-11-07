from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from synth import interruption_sweep


def test_sweep_refuses_to_overwrite_existing_evidence(tmp_path):
    output = tmp_path / 'result.json'
    output.write_text('preserve me', encoding='utf-8')
    with pytest.raises(FileExistsError):
        interruption_sweep.run(tmp_path, output)
    assert output.read_text(encoding='utf-8') == 'preserve me'


def test_sweep_requires_all_named_fixtures_before_running(tmp_path):
    with pytest.raises(ValueError, match='Missing fixtures'):
        interruption_sweep.run(tmp_path, tmp_path / 'new.json')
    assert not (tmp_path / 'new.json').exists()


@pytest.mark.parametrize('gate', [0, -1, 1.1])
def test_sweep_rejects_invalid_echo_gates(tmp_path, gate):
    with pytest.raises(ValueError, match='Invalid experiment setting'):
        interruption_sweep.run(tmp_path, tmp_path / 'new.json', active_echo_gate=gate)
