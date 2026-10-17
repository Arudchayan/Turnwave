"""Repeatable semantic and state-recovery tests against the real GPU runner.

Unlike presence-only probes, every reply must satisfy an explicit rubric.
Run while the UI is stopped. This never records or plays device audio.
"""
import argparse
import difflib
import json
import pathlib
import re
import sys
import threading
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from client import ServerClient, Turn, TTS_VOICE, wav16k_bytes

CASES = [
    ('arithmetic', 'What is two plus two?', r'\b(four|4)\b'),
    ('banana', 'Say only the word banana.', r'^\W*banana\W*$'),
    ('sky', 'What color is a clear daytime sky?', r'\bblue\b'),
    ('name', 'My name is Alex. Please remember my name.', r'\balex\b'),
    ('recall', 'What is my name?', r'\balex\b'),
]


def normalize(text):
    return re.findall(r'[a-z0-9]+', text.lower())


def ensure_idle():
    import urllib.request, urllib.error
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/api/state', timeout=1) as response:
            state = json.load(response)
        if state:
            raise RuntimeError('Stop the UI before sharing its backend with tests; muted conversations still own context.')
    except urllib.error.URLError:
        pass  # UI need not be running.


def fixtures(client):
    folder = ROOT / 'data' / 'synth' / 'quality'
    folder.mkdir(exist_ok=True)
    rows = []
    for name, text, rubric in CASES:
        path = folder / (name + '.wav')
        if not path.exists():
            accepted = None
            for attempt in range(3):
                parts = []; turn = Turn()
                client.tts(text, turn, max_tokens=192, on_audio=parts.append)
                if turn.error or not parts:
                    continue
                audio = np.concatenate(parts)
                if not .4 <= len(audio)/24000 <= 2 + .65 * len(text.split()):
                    continue
                audio = np.interp(np.arange(round(len(audio)*2/3))*1.5, np.arange(len(audio)), audio).astype(np.float32)
                data = wav16k_bytes(audio)
                transcription = client.asr_text(data)
                expected, actual = normalize(text), normalize(transcription)
                similarity = difflib.SequenceMatcher(a=expected,b=actual).ratio()
                if similarity >= .7:
                    accepted = data
                    break
            if accepted is None:
                raise RuntimeError('Unable to generate a verified fixture for ' + name)
            path.write_bytes(accepted)
        # Validate cached fixtures too: corrupted TTS is never a passing input.
        transcript = client.asr_text(path.read_bytes())
        match = difflib.SequenceMatcher(a=normalize(text), b=normalize(transcript)).ratio()
        if match < .7:
            raise RuntimeError(f'Fixture {name} does not match its script: {transcript!r}')
        rows.append(dict(name=name, script=text, asr=transcript, similarity=round(match,3), rubric=rubric, file=path.name))
    (folder / 'manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    return {row['name']: (folder/row['file']).read_bytes() for row in rows}


def reply(client, data, first=True):
    turn = Turn()
    client.chat_turn(data, turn, first)
    return turn


def run(rounds, output):
    ensure_idle()
    client = ServerClient()
    assert client.health(), 'Start the GPU server first'
    inputs = fixtures(client)
    poison_input = (ROOT/'data/synth/user_fact.wav').read_bytes()
    rows = []
    baselines = {}
    for cycle in range(rounds):
        for name, script, rubric in CASES[:3]:
            # Alternate clean reset, token-limited audio, and socket abort.
            phase = ['fresh','token_limit','cancel'][cycle % 3]
            if phase == 'token_limit':
                short = Turn(); client.chat_turn(poison_input,short,True,max_tokens=10)
            elif phase == 'cancel':
                cut = Turn()
                client.chat_turn(poison_input,cut,True,on_audio=lambda audio:cut.abort())
            turn = reply(client, inputs[name])
            substantive = bool(re.search(rubric,turn.text,re.I))
            baseline = baselines.setdefault(name, normalize(turn.text))
            consistent = normalize(turn.text) == baseline
            row = dict(round=cycle+1,case=name,phase=phase,text=turn.text,
                       first_audio_s=round(turn.ttft,3) if turn.ttft is not None else None,
                       audio_s=round(turn.audio_samples/24000,3),error=turn.error,
                       rubric_pass=substantive,reset_consistent=consistent)
            row['passed'] = bool(substantive and consistent and not turn.error and turn.audio_samples>0 and turn.ttft is not None and turn.ttft<1.2)
            rows.append(row); print(json.dumps(row),flush=True)
        introduction = reply(client,inputs['name'])
        recall = reply(client,inputs['recall'],False)
        row = dict(round=cycle+1,case='memory',phase='continuation',text=recall.text,
                   error=recall.error,first_audio_s=round(recall.ttft,3) if recall.ttft is not None else None,
                   passed=bool(re.search(r'\balex\b',recall.text,re.I) and not re.search(r"\b(i(?:'|’)?m|my name is|i am)\s+alex\b", recall.text, re.I) and not recall.error and recall.audio_samples>0))
        rows.append(row);print(json.dumps(row),flush=True)
    report = dict(rounds=rounds, passed=all(row['passed'] for row in rows), checks=rows,
                  chat_system=__import__('client').CHAT_SYS, description='Real GPU; verified synthetic input; semantic, reset, interruption recovery, and memory gates.')
    pathlib.Path(output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PASS' if report['passed'] else 'FAIL',output,flush=True)
    return report['passed']


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--output',default=str(ROOT/'outputs'/'conversation_suite.json'))
    args=parser.parse_args()
    if not 1 <= args.rounds <= 30: parser.error('Use 1-30 rounds per bounded run')
    from backend_session import backend_session
    with backend_session(test=True):
        raise SystemExit(0 if run(args.rounds,args.output) else 1)
