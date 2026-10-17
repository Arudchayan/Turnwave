import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from synth.text_input_diagnostic import validate_origin,prepare
from synth.semantic_contract import VERSION
from client import ServerClient
from synth.checkpoint_experiment import digest


def test_text_control_waits_for_exact_completed_audio_roster_not_summary_claim(tmp_path):
    (tmp_path/'protocol.json').write_text(json.dumps(dict(runs=[dict(id='a'),dict(id='b')])))
    sha=digest(tmp_path/'protocol.json')
    with pytest.raises(ValueError,match='Wait'):validate_origin(tmp_path,sha)
    (tmp_path/'summary.json').write_text('{"complete":true}')
    (tmp_path/'server-exit.json').write_text('{"exit_code":0}')
    result=tmp_path/'results.jsonl'
    result.write_text(json.dumps(dict(run='a',recall=True))+'\n')
    with pytest.raises(ValueError,match='exact roster'):validate_origin(tmp_path,sha)
    result.write_text(''.join(json.dumps(dict(run=k,recall=True))+'\n' for k in ['a','b']))
    validate_origin(tmp_path,sha)
    result.write_text(''.join(json.dumps(dict(run=k,recall=True))+'\n' for k in ['a','a','b']))
    with pytest.raises(ValueError,match='exact roster'):validate_origin(tmp_path,sha)
    (tmp_path/'protocol.json').write_text('{}')
    with pytest.raises(ValueError,match='protocol changed'):validate_origin(tmp_path,sha)


def test_replay_requires_exact_payloads_and_never_silently_changes_a_saved_question(tmp_path):
    origin=tmp_path/'audio';origin.mkdir();saved=tmp_path/'saved';saved.mkdir()
    fact=dict(id='n',setup_text='setup',question='question',directed=True)
    protocol=dict(evaluator=VERSION,facts=[fact],profiles=dict(baseline='system'),reference_server={})
    (origin/'protocol.json').write_text(json.dumps(protocol))
    (origin/'preregistration.json').write_text(json.dumps(dict(protocol_sha256=digest(origin/'protocol.json'))))
    for pos,(kind,text) in enumerate([('SETUP','setup'),('RECALL','question')]):
        (saved/f'n-text-turn{pos}-{kind}-request.json').write_text(json.dumps(ServerClient._payload('system',text,pos==0,512)))
    (saved/'protocol.json').write_text('{}')
    output=tmp_path/'replay';prepare(origin,output,['n'],saved)
    replay=json.loads((output/'protocol.json').read_text())
    assert len(replay['facts'])==1 and len(replay['expected_payloads'])==2
    path=saved/'n-text-turn1-RECALL-request.json'
    payload=json.loads(path.read_text());payload['max_tokens']=100
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='differs'):prepare(origin,tmp_path/'bad',['n'],saved)
    assert not (tmp_path/'bad').exists()
