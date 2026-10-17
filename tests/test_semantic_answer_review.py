import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from synth.review_semantic_answers import review


def test_exact_format_rejects_keyword_in_sentence_without_overwriting_old_grade():
    row=dict(run='x',fact='INST-1',framing='FA',text='The user picked seven.',passed=True)
    item=review(row,dict(directed=dict(FA=True)))
    assert item['historical_passed'] is True and item['strict_format_pass'] is False
    assert review(dict(row,text='Seven.'),dict(directed=dict(FA=True)))['strict_format_pass'] is True


def test_negation_is_review_flag_not_automatic_truth_or_new_grade():
    row=dict(run='x',fact='EPI-3',framing='FB',text='The sister does not live in Oxford.',passed=True)
    item=review(row,dict(directed=dict(FB=False)))
    assert item['negation_review'] and item['strict_format_pass'] is None
    assert item['historical_passed'] is True and 'passed' not in item
