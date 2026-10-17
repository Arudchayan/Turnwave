from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from synth.verify_reply_audio import correspondence
from synth.verify_reply_audio import validate_roster
import pytest


def test_number_and_punctuation_transcriptions_are_not_false_corruption_flags():
    assert not correspondence('Three.','3')['needs_review']
    assert not correspondence('Tuesday at 3 PM.','Tuesday at three p.m.')['needs_review']


def test_lost_negation_is_flagged_even_when_global_similarity_is_high():
    source='I do not know the exact time of the meeting, but you can check the calendar for more details.'
    heard=source.replace('not ','')
    report=correspondence(source,heard)
    assert report['similarity']>.9 and report['polarity_mismatch'] and report['needs_review']


def test_partial_audio_review_is_explicit_and_never_accepts_duplicate_or_unknown_ids():
    with pytest.raises(ValueError,match='complete'):validate_roster(['a'],{'a','b'})
    assert validate_roster(['a'],{'a','b'},True)==['b']
    with pytest.raises(ValueError,match='Duplicate'):validate_roster(['a','a'],{'a','b'},True)
    with pytest.raises(ValueError,match='unknown'):validate_roster(['z'],{'a','b'},True)


def test_one_lost_negation_is_caught_when_another_remains_in_a_different_clause():
    source="I don't have access to your calendar, so I can't tell you the exact time."
    heard=source.replace("can't tell",'can tell')
    assert correspondence(source,heard)['polarity_mismatch']
    assert not correspondence(source,source.replace("can't",'cannot'))['needs_review']
    assert correspondence('I cannot see but can hear.','I can see but cannot hear.')['polarity_mismatch']
