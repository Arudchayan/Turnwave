"""Contracts for the private-free ASR screen's metrics and ownership guard."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('lfm3_realtime_asr_bench', ROOT / 'scripts' / 'lfm3_realtime_asr_bench.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_score_normalizes_case_and_punctuation_without_returning_text():
    row = bench.score('Hello, world!', 'hello there world')
    assert row == {'words': 2, 'edits': 1, 'empty': False}
    assert bench.score('hello', '') == {'words': 1, 'edits': 1, 'empty': True}


def test_summary_preserves_failed_cases_and_reference_denominator():
    result = bench.summary([{'words': 2, 'edits': 1, 'empty': False},
                            {'words': 3, 'edits': 0, 'empty': False},
                            {'error': 'TimeoutError'}])
    assert result == {'clips': 3, 'completed': 2, 'errors': 1, 'reference_words': 5,
                      'word_edits': 1, 'corpus_wer': .2, 'sentences_with_errors': 1,
                      'empty': 0}


def test_existing_evidence_is_never_overwritten(tmp_path):
    path = tmp_path / 'evidence.json'
    path.write_text('keep')
    try:
        bench.run('http://127.0.0.1:18122', tmp_path / 'model', '/model', path, 1)
    except FileExistsError:
        pass
    else:
        assert False, 'must refuse to overwrite evidence'
    assert path.read_text() == 'keep'
