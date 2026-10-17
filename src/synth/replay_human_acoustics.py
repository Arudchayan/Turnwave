"""Replay recorded H3 acoustics offline; compare old/new residual onset gates."""
import argparse
import json
from pathlib import Path
import sys
import wave
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from duplex import DuplexFrontend


def read(path):
    with wave.open(str(path),'rb') as reader:
        rate=reader.getframerate();channels=reader.getnchannels()
        samples=np.frombuffer(reader.readframes(reader.getnframes()),dtype='<i2').astype(np.float32).reshape(-1,channels)/32768
    channel=int(np.argmax(np.mean(samples*samples,axis=0)))
    samples=samples[:,channel]
    return np.interp(np.arange(round(len(samples)*16000/rate))*rate/16000,np.arange(len(samples)),samples).astype(np.float32)


def main(folder,floor=.005,filename="h3-acoustic-replay-floor005.json",*,output_folder=None,frontend=None):
    if output_folder is None:raise ValueError('Provide a separate experiment output folder; historical recordings are immutable')
    source_folder=Path(folder).resolve();destination=Path(output_folder).resolve()
    if destination==source_folder or source_folder in destination.parents:
        raise ValueError('Replay output must be outside the historical experiment')
    if Path(filename).name!=filename:raise ValueError('Replay filename must stay inside its output folder')
    output=destination/filename
    factory=frontend or DuplexFrontend
    if output.exists():raise FileExistsError('Preserve earlier replay; use a new output version')
    audio=folder/'audio-from-E';session=json.loads((audio/'audio-session.json').read_text(encoding='utf-8'));origin=session['started_ts']
    events=[json.loads(line) for line in (folder/'after/outputs/logs/app.jsonl').read_text(encoding='utf-8').splitlines()]
    events=[e for e in events if e.get('pid')==22204]
    starts={e['turn_id']:e['ts'] for e in events if e['event']=='playback_start'}
    ends={e['turn_id']:e['ts'] for e in events if e['event']=='playback_end'}
    target_start=starts[76];target_end=ends[76];begin=max(origin,target_start-30);finish=target_end+.08
    near=read(audio/'microphone-aligned.wav');far=read(audio/'speaker-aligned.wav')
    length=max(len(near),len(far));near=np.pad(near,(0,length-len(near)));far=np.pad(far,(0,length-len(far)))
    spans=[(start,ends[job]) for job,start in starts.items() if job in ends and start<finish and ends[job]>begin]
    rows=[]
    for delay in [40,80]:
        for floor in [0,floor]:
            front=factory(stream_delay_ms=delay,residual_floor=floor);front.warmup();onsets=[];active=False
            start_index=round((begin-origin)*16000);end_index=min(length,round((finish-origin)*16000))
            for i in range(start_index,end_index-512,512):
                ts=origin+(i+256)/16000;playing=any(a<=ts<=b for a,b in spans)
                if playing:front.push_far(far[i:i+512])
                elif active:front.clear_playback()
                active=playing
                _,started,ended,_=front.update(near[i:i+512])
                if started:onsets.append(dict(ts=round(origin+(i+512)/16000,3),probability=front.probability,echo_score=front.echo_score,clean_rms=front.level))
                if ended:front.reset()
            false=[e for e in onsets if target_start+.35<=e['ts']<=target_end+.05]
            row=dict(configured_delay_ms=delay,residual_floor=floor,quiet_h3_onsets=false,total_onsets=len(onsets),residual_rejections=front.residual_rejections,aec_error=front.aec_error)
            rows.append(row);print(json.dumps(row),flush=True)
    report=dict(cases=rows,target_turn=76,user_confirmed_quiet=True,window=dict(start=target_start,end=target_end),
                note='Offline raw microphone + timestamp-aligned speaker loopback, 30 s warm history. Reference gain/timing differ from device-bound app reference; a successful replay does not prove all-room hardware behavior.')
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);parser.add_argument('--output-folder',type=Path,required=True);args=parser.parse_args()
    args.output_folder.mkdir(parents=True,exist_ok=False);main(args.folder,output_folder=args.output_folder)
