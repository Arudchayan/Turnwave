"""Opt-in real-device echo/playout check against a running local app.

Plays three speaker checks while the microphone processes room sound. Saves
only timing/cancellation metrics; no raw mic audio. Do not talk during it.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request

BASE='http://127.0.0.1:8000'
ROOT=Path(__file__).resolve().parents[2]


def get(path):
    with urllib.request.urlopen(BASE+path,timeout=3) as response:return json.load(response)


def post(path):
    with urllib.request.urlopen(urllib.request.Request(BASE+path,data=b''),timeout=3) as response:return json.load(response)


def run(cycles=3):
    initial=get('/api/state')
    if initial['status']!='listening' or initial['error'] or initial['audio_diagnostics']['input_latency_s'] is None:
        raise RuntimeError('Microphone must be ready and error-free; a speaker-only test cannot establish echo handling.')
    if initial['mic_on']:
        raise RuntimeError('Mute the app before this opt-in test so it cannot interrupt a live conversation.')
    rows=[]
    try:
        post('/api/mic')
        for cycle in range(cycles):
            start=time.time();job_id=post('/api/test')['turn_id']
            deadline=time.time()+15
            while time.time()<deadline:
                time.sleep(.1)
                events=get('/api/diagnostics')
                ends=[e for e in events if e['event']=='playback_end' and e['turn_id']==job_id]
                if ends:break
            else:raise RuntimeError('Speaker check timed out')
            end=ends[-1]
            interruptions=[e for e in events if e['event']=='interrupt' and e['ts']>=start and e['reason']=='barge-in']
            state=get('/api/state')
            row=dict(cycle=cycle+1,playback=end,unexpected_speech_interruptions=len(interruptions),input=state['audio_diagnostics'])
            row['passed']=bool(not interruptions and not end['cancelled'] and not end['error'] and end['played_s']>1 and state['audio_diagnostics']['input_latency_s'] is not None and not state['error'] and end['playback'].get('underflows',0)==0)
            rows.append(row);print(json.dumps(row),flush=True)
            if not row['passed']:
                break
            time.sleep(.3)
    finally:
        if get('/api/state')['mic_on']:post('/api/mic')
        post('/api/interrupt')
    report=dict(passed=all(row['passed'] for row in rows),cases=rows,
                note='Real input/output with active microphone and AEC. Unexpected speech can be human speech or echo. No raw mic audio saved.')
    path=ROOT/'outputs/hardware_playback.json';path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report['passed']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',help='Play audio and briefly unmute capture for testing')
    args=parser.parse_args()
    if not args.run:parser.error('Use --run for the real-device check; stay quiet during its three spoken clips.')
    raise SystemExit(0 if run() else 1)
