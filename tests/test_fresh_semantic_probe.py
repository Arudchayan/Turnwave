import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from synth import fresh_semantic_probe as probe


def test_preregistered_screen_balances_profile_order_and_shares_input_facts(tmp_path,monkeypatch):
    reference=tmp_path/'reference';reference.mkdir()
    (reference/'EPI-1-FA-david-r0-turn0-SETUP-request.json').write_text(json.dumps(dict(messages=[dict(content='Historical system')])) )
    (reference/'server-process.json').write_text('{}')
    monkeypatch.setattr(probe,'REFERENCE',reference)
    experiment=tmp_path/'new';probe.prepare(experiment)
    protocol=json.loads((experiment/'protocol.json').read_text())
    assert len(protocol['facts'])==6 and len(protocol['cases'])==12 and len(protocol['runs'])==24
    assert protocol['profiles']['ownership']==protocol['profiles']['baseline']+probe.OWNERSHIP
    pairs=[protocol['runs'][i:i+2] for i in range(0,24,2)]
    assert all(a['fact']==b['fact'] and a['voice']==b['voice'] and a['arm']!=b['arm'] for a,b in pairs)
    assert sum(pair[0]['arm']=='baseline' for pair in pairs)==6
    assert json.loads((experiment/'preregistration.json').read_text())['protocol_sha256']==probe.digest(experiment/'protocol.json')
    with pytest.raises(FileExistsError):probe.prepare(experiment)


def test_fixture_gate_requires_correct_value_and_uncut_instruction_tail():
    setup='For this test, my name is Nora. Please remember this and acknowledge briefly.'
    protocol=dict(facts=[dict(id='n',contract=probe.contract_from_setup(setup))])
    fixture=dict(id='n-SETUP',spoken_text=setup)
    assert probe.check_heard(fixture,setup,protocol)['passed']
    corrupted=probe.check_heard(fixture,setup.replace('Nora','Laura'),protocol)
    assert corrupted['similarity']>.9 and not corrupted['critical_detail_pass'] and not corrupted['passed']
    question=dict(id='n-RECALL',spoken_text='What is my name? Just the name.')
    assert not probe.check_heard(question,'What is my name? Just the',protocol)['passed']


def test_fixture_gate_accepts_equivalent_pm_punctuation_but_not_am():
    setup='For this test, my meeting is Tuesday at 3 PM. Please remember this and acknowledge briefly.'
    protocol=dict(facts=[dict(id='m',contract=probe.contract_from_setup(setup))])
    fixture=dict(id='m-SETUP',spoken_text=setup)
    assert probe.check_heard(fixture,setup.replace('3 PM','3 p.m.'),protocol)['passed']
    assert not probe.check_heard(fixture,setup.replace('3 PM','3 a.m.'),protocol)['passed']
