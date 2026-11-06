"""The WER helper must not hide punctuation, case or insertion errors."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from synth.text_metrics import edit_distance, normalize_text, word_error_rate


def test_normalization_is_case_and_punctuation_insensitive():
    assert normalize_text('Mr. Quilter, M.A.!') == ['MR', 'QUILTER', 'M', 'A']


def test_edit_distance_handles_insertion_deletion_and_substitution():
    assert edit_distance(['A', 'B'], ['A', 'B']) == 0
    assert edit_distance(['A', 'B'], ['A', 'C']) == 1
    assert edit_distance(['A', 'B'], ['A']) == 1
    assert edit_distance(['A'], ['A', 'B']) == 1


def test_wer_is_normalized_and_has_the_expected_denominator():
    assert word_error_rate('MR QUILTER', 'Mr. Quilter') == 0
    assert word_error_rate('A B', 'A C') == pytest.approx(.5)
    assert word_error_rate('', '') == 0
    assert word_error_rate('', 'unexpected') == 1
