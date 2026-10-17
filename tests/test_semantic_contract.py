import io
from pathlib import Path
import sys
import wave
import hashlib

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from synth.semantic_contract import contract_from_setup,evaluate,audio_evidence
from synth.adjudicate_semantic_review import adjudicate


def contract(statement):
    return contract_from_setup('For this test, '+statement+'. Please remember this and acknowledge briefly.')


@pytest.mark.parametrize('statement,answer',[
    ('my name is Nora','Nora.'),
    ('my lucky number is seven','7.'),
    ('my appointment is Friday at noon','Friday at 12 PM.'),
    ('my meeting is Tuesday at 3 PM','Tuesday at 3 p.m.'),
    ('my sister lives in York','Your sister lives in York.'),
    ('I bought an orange mug yesterday','You bought an orange mug yesterday.'),
    ('I bought an orange mug yesterday','Orange mug.'),
    ('I walk my dog every afternoon','You walk your dog every afternoon.'),
])
def test_grounded_positive_controls_are_not_all_sent_to_manual(statement,answer):
    assert evaluate(answer,contract(statement),False,True)['outcome']=='pass'


@pytest.mark.parametrize('statement,answer',[
    ('my name is Nora',"The user's name is not Nora."),
    ('my name is Nora',"The assistant's name is Nora."),
    ('my name is Nora','Nora?'),
    ('my name is Nora','No ra.'),
    ('my name is Nora','Nora or Maya.'),
    ('my name is Nora','Nora 雪'),
    ('my name is Nora','Your name is Maya.'),
    ('my password is copper meadow','The password is not copper meadow.'),
    ('my sister lives in York','The sister does not live in York.'),
    ('my sister lives in York','The sister lives in York. Actually, she lives in Oxford.'),
    ('I walk my dog every evening','Most people walk their dog in the morning and evening.'),
    ('I walk my dog every evening','When the user walks their dog.'),
    ('my meeting is Thursday at 4 PM','I cannot see your calendar.'),
    ('my meeting is Tuesday at 3 PM','Tuesday at 3 a.m.'),
])
def test_negation_wrong_owner_questions_and_generic_advice_never_auto_pass(statement,answer):
    assert evaluate(answer,contract(statement),False,True)['outcome']!='pass'


def test_first_person_claim_and_directed_format_are_separate_failures():
    c=contract('my name is Nora')
    own=evaluate('My name is Nora.',c,False,True)
    assert own['content_pass'] and own['attribution_pass'] is False and own['outcome']=='fail'
    verbose=evaluate("The user's name is Nora.",c,True,True)
    assert verbose['content_pass'] and verbose['attribution_pass'] and not verbose['instructions_pass']
    assert verbose['outcome']=='fail'


def test_contract_comes_from_source_value_and_unfamiliar_paraphrase_stays_manual():
    c=contract('my sister lives in Exeter')
    assert evaluate('Your sister lives in York.',c,False,True)['outcome']=='manual'
    unknown=evaluate('She has her home in Exeter.',c,False,True)
    assert unknown['semantic_review_required'] and unknown['outcome']=='manual'
    assert evaluate('Exeter.',c,False,False)['outcome']=='fail'
    with pytest.raises(ValueError):contract_from_setup('Unsupported context')


def wav(samples):
    data=io.BytesIO()
    with wave.open(data,'wb') as writer:
        writer.setnchannels(1);writer.setsampwidth(2);writer.setframerate(24000)
        writer.writeframes(np.asarray(samples,dtype='<i2').tobytes())
    return data.getvalue()


def test_saved_audio_integrity_rejects_silence_truncation_and_wrong_duration():
    data=wav([1000,-1000]*1200)
    assert audio_evidence(data,.1)['valid']
    assert not audio_evidence(data[:-100],.1)['valid']
    assert not audio_evidence(data,.2)['valid']
    assert not audio_evidence(wav([0]*2400),.1)['valid']
    assert not audio_evidence(b'not a WAV',.1)['valid']


def test_adjudication_is_bound_to_exact_answer_and_cannot_override_transport():
    original=evaluate('Unknown wording',contract('my name is Nora'),False,None)
    original.update(run='r',fact='f',text='Unknown wording')
    label=dict(run='r',fact='f',text_sha256=hashlib.sha256(b'Unknown wording').hexdigest(),
               content_pass=True,attribution_pass=True,instructions_pass=True,reason='Explicit contextual review')
    result=adjudicate(original,label)
    assert result['outcome']=='manual' and result['transport_pass'] is None
    assert result['automatic']['outcome']=='manual' and original['content_pass'] is None
    with pytest.raises(ValueError,match='text differs'):adjudicate(dict(original,text='Edited'),label)
    with pytest.raises(ValueError,match='Only unresolved'):adjudicate(dict(original,outcome='pass'),label)
    with pytest.raises(ValueError,match='boolean'):adjudicate(original,dict(label,content_pass='true'))
