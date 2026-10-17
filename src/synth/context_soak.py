"""Sustained native-context check using previously ASR-verified fixtures.

Requires the UI to be stopped. No capture or playback. Does not recreate
fixtures, because ASR would reset the exact context under test.
"""
import argparse
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from client import ServerClient, Turn
from conversation_suite import CASES, ensure_idle, normalize


def run(turns):
    ensure_idle();client=ServerClient();folder=ROOT/'data/synth/quality'
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    for name,script,_ in CASES[:3]:
        rows=[r for r in manifest if r['name']==name]
        if not rows or rows[0]['similarity']<.7 or normalize(rows[0]['script'])!=normalize(script):
            raise RuntimeError('Verify fixture first: '+name)
    rows=[]
    for index in range(turns):
        ensure_idle()
        name,script,rubric=CASES[index%3]
        turn=Turn();client.chat_turn((folder/(name+'.wav')).read_bytes(),turn,index==0,max_tokens=256)
        row=dict(turn=index+1,case=name,text=turn.text,error=turn.error,
                 first_audio_s=round(turn.ttft,3) if turn.ttft is not None else None,audio_s=round(turn.audio_samples/24000,3))
        row['passed']=bool(not turn.error and turn.audio_samples and re.search(rubric,turn.text,re.I) and turn.ttft is not None and turn.ttft<1.2)
        rows.append(row);print(json.dumps(row),flush=True)
        if turn.error:break
    report=dict(requested_turns=turns,completed_turns=len(rows),passed=len(rows)==turns and all(r['passed'] for r in rows),checks=rows,
                scope='Native cached context; repeated simple prompts. Does not prove context-dependent recall or physical audio quality.')
    (ROOT/'outputs/context_soak.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report['passed']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--turns',type=int,default=24);args=parser.parse_args()
    if not 3<=args.turns<=60:parser.error('Use 3-60 turns')
    from backend_session import backend_session
    with backend_session(test=True):
        raise SystemExit(0 if run(args.turns) else 1)
