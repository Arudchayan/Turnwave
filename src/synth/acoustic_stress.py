"""Seeded acoustic stress tests: delayed/reverberant echo, noise and double talk."""
import json
import pathlib
import sys
import wave
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from duplex import DuplexFrontend


def read(name):
    with wave.open(str(ROOT/'data/synth'/name),'rb') as reader:
        rate=reader.getframerate()
        audio=np.frombuffer(reader.readframes(reader.getnframes()),dtype='<i2').astype(np.float32)/32768
    if rate!=16000:
        audio=np.interp(np.arange(round(len(audio)*16000/rate))*rate/16000,np.arange(len(audio)),audio).astype(np.float32)
    return audio


def run(output=None):
    far=read('_assistant.wav')
    user=read('user_barge.wav')
    voiced=np.flatnonzero(abs(user)>.02);user=user[max(0,voiced[0]-160):]
    # Enough far-end audio for adaptation before speech begins.
    far=np.tile(far,2)[:8*16000]
    rng=np.random.default_rng(20261016)
    rows=[]
    for delay_ms in (20,80,180):
        delay=round(delay_ms*16)
        for gain in (.2,1.0):
            for overlap in (False,True):
                echo=np.zeros(len(far),np.float32)
                for tail_ms,scale in [(0,1),(25,.3),(70,.15)]:
                    offset=delay+round(tail_ms*16)
                    echo[offset:]+=far[:len(far)-offset]*gain*scale
                mic=echo+rng.normal(0,.0005,len(far)).astype(np.float32)
                onset=16000
                if overlap:
                    end=min(len(mic),onset+len(user));mic[onset:end]+=.8*user[:end-onset]
                mic=np.clip(mic,-1,1)
                front=DuplexFrontend(stream_delay_ms=80);front.warmup()
                starts=[]
                for i in range(0,len(far)-512,512):
                    front.push_far(far[i:i+512])
                    _,started,_,_=front.update(mic[i:i+512])
                    if started:starts.append((i+512)/16000)
                false_starts=[t for t in starts if not overlap or t<1.0]
                detection=next((t-1 for t in starts if t>=1),None) if overlap else None
                row=dict(delay_ms=delay_ms,echo_gain=gain,overlap=overlap,starts=starts,
                         false_starts=len(false_starts),barge_s=round(detection,3) if detection is not None else None,
                         aec_error=front.aec_error)
                row['passed']=not false_starts and not front.aec_error and (not overlap or detection is not None and detection<.5)
                rows.append(row);print(json.dumps(row),flush=True)
    # Quiet user speech against louder echo: protect barge-in when adding
    # the H3 low-energy residual rejection gate.
    for delay_ms in (80,180):
        for user_gain in (.15,.3):
            delay=delay_ms*16;echo=np.zeros(len(far),np.float32)
            for tail_ms,scale in [(0,1),(25,.3),(70,.15)]:
                offset=delay+tail_ms*16;echo[offset:]+=far[:-offset]*scale
            mic=echo+rng.normal(0,.0005,len(far)).astype(np.float32);onset=16000
            end=min(len(mic),onset+len(user));mic[onset:end]+=user_gain*user[:end-onset]
            front=DuplexFrontend(stream_delay_ms=80);front.warmup();starts=[]
            for i in range(0,len(far)-512,512):
                front.push_far(far[i:i+512]);_,started,_,_=front.update(mic[i:i+512])
                if started:starts.append((i+512)/16000)
            false=[t for t in starts if t<1];detection=next((t-1 for t in starts if t>=1),None)
            row=dict(scenario='quiet_double_talk',delay_ms=delay_ms,user_gain=user_gain,starts=starts,
                     false_starts=len(false),barge_s=round(detection,3) if detection is not None else None,aec_error=front.aec_error)
            row['passed']=not false and not front.aec_error and detection is not None and detection<.5
            rows.append(row);print(json.dumps(row),flush=True)
    # Render stops abruptly; the room still returns delayed/reverberant audio.
    for delay_ms in (20,80,180):
        for overlap in (False,True):
            render=far[:4*16000].copy();cut=2*16000;render[cut:]=0
            mic=np.zeros(len(render),np.float32)
            for tail_ms,scale in [(0,1),(25,.3),(70,.15)]:
                offset=(delay_ms+tail_ms)*16
                mic[offset:]+=render[:-offset]*scale
            if overlap:
                end=min(len(mic),cut+len(user));mic[cut:end]+=.8*user[:end-cut]
            mic+=rng.normal(0,.0005,len(mic)).astype(np.float32)
            front=DuplexFrontend(stream_delay_ms=80);front.warmup();starts=[];cleared=False
            for i in range(0,len(render)-512,512):
                if i<cut:front.push_far(render[i:min(i+512,cut)])
                elif not cleared:front.clear_playback();cleared=True
                _,started,_,_=front.update(mic[i:i+512])
                if started:starts.append((i+512)/16000)
            false=[t for t in starts if not overlap or t<2]
            detection=next((t-2 for t in starts if t>=2),None) if overlap else None
            row=dict(scenario='cancel_tail',delay_ms=delay_ms,overlap=overlap,starts=starts,
                     false_starts=len(false),barge_s=round(detection,3) if detection is not None else None,aec_error=front.aec_error)
            row['passed']=not false and not front.aec_error and (not overlap or detection is not None and detection<.5)
            rows.append(row);print(json.dumps(row),flush=True)
    report=dict(seed=20261016,passed=all(row['passed'] for row in rows),cases=rows,
                description='Real AEC + Silero; 22 seeded delayed/multipath echo, double-talk and post-cancellation tail cases, including mismatch from configured delay.')
    path=pathlib.Path(output) if output else ROOT/'outputs/acoustic_stress.json'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report['passed']


if __name__=='__main__':raise SystemExit(0 if run() else 1)
