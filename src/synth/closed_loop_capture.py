"""Real Assistant capture loop with synthetic room/playback and independent ASR.

Capture and ASR run in separate processes. No microphone, output device, or
model backend is opened. The speaker stops when Assistant.interrupt calls it.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import re
import shutil
import sys
import threading
import time
import wave

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from app import Assistant,Job
from diagnostics import Diagnostics
from duplex import DuplexFrontend
from synth.acoustic_stress import read
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint
from synth.dual_aec_experiment import make_candidate

WHISPER_REVISION='3d3d5dee26484f91867d81cb899cfcf72b96be6c'


class StepQueue(queue.Queue):
    """Acknowledge a frame only when the consumer asks for its next frame."""
    def __init__(self):
        super().__init__(maxsize=1);self.finished=threading.Condition();self.completed=0;self.previous=False
    def get(self,*args,**kwargs):
        with self.finished:
            if self.previous:
                self.completed+=1;self.previous=False;self.finished.notify_all()
        result=super().get(*args,**kwargs)
        with self.finished:self.previous=True
        return result
    def wait_completed(self,count):
        with self.finished:
            if not self.finished.wait_for(lambda:self.completed>=count,timeout=10):
                raise TimeoutError('Capture worker did not complete its frame')


class VirtualMic:
    device=-1
    def __init__(self):self.q=StepQueue()
    def start(self):pass
    def stop(self):pass


class VirtualSpeaker:
    def __init__(self):self.playing=True;self.sample_clock=0;self.stop_sample=None
    def stop(self):
        if self.playing and self.stop_sample is None:self.stop_sample=self.sample_clock
        self.playing=False


class NoBackend:
    base='offline://capture-only'
    def health(self):raise AssertionError('Capture experiment must not access a backend')


class CaptureAssistant(Assistant):
    def __init__(self,**kwargs):super().__init__(**kwargs);self.captured=[];self.capture_end_samples=[]
    def submit_audio(self,audio):
        self.captured.append(audio.copy());self.capture_end_samples.append(self.player.sample_clock);self.status='listening'
    def interrupt(self,why='button',announce=True):
        if why=='barge-in' and self._active is not None and hasattr(self.front,'use_barge_endpoint'):
            self.front.use_barge_endpoint()
        return super().interrupt(why,announce)


class CaptureDiagnostics(Diagnostics):
    def __init__(self,path,player):super().__init__(path);self.player=player
    def event(self,event,**fields):
        return super().event(event,audio_clock_s=self.player.sample_clock/16000,**fields)


def contextual_frontend(base,milliseconds):
    """Research-only longer endpoint for a turn that interrupted an active reply."""
    class ContextualFrontend(base):
        def __init__(self,**kwargs):
            super().__init__(**kwargs);self.base_end_chunks=self.end_chunks
        def use_barge_endpoint(self):self.end_chunks=max(1,round(milliseconds/32))
        def reset(self):
            super().reset();self.end_chunks=self.base_end_chunks
    return ContextualFrontend


def write_wav(path,audio):
    with wave.open(str(path),'wb') as writer:
        writer.setnchannels(1);writer.setsampwidth(2);writer.setframerate(16000)
        writer.writeframes(np.rint(np.clip(audio,-1,32767/32768)*32768).astype('<i2').tobytes())


def words(text):
    numbers=dict(zip(['one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve'],map(str,range(1,13))))
    return [numbers.get(token,token) for token in re.findall(r'[a-z0-9]+',text.lower())]


def word_errors(reference,hypothesis):
    a,b=words(reference),words(hypothesis)
    d=np.zeros((len(a)+1,len(b)+1),dtype=int);d[:,0]=np.arange(len(a)+1);d[0,:]=np.arange(len(b)+1)
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(a[i-1]!=b[j-1]))
    i,j=len(a),len(b);sub=delete=insert=0
    while i or j:
        if i and j and a[i-1]==b[j-1] and d[i,j]==d[i-1,j-1]:i-=1;j-=1
        elif i and j and d[i,j]==d[i-1,j-1]+1:sub+=1;i-=1;j-=1
        elif i and d[i,j]==d[i-1,j]+1:delete+=1;i-=1
        else:insert+=1;j-=1
    return dict(wer=(sub+delete+insert)/max(1,len(a)),substitutions=sub,deletions=delete,insertions=insert,reference_words=len(a))


def no_new_negation(reference,hypothesis):
    def counts(text):
        tokens=re.findall(r"\b(?:no|not|never|without|cannot|can't|don't|doesn't|didn't|won't|isn't|aren't|wasn't|weren't)\b",text.lower().replace('’',"'"))
        return collections.Counter('not' if t=='cannot' or "n't" in t else t for t in tokens)
    a,b=counts(reference),counts(hypothesis)
    return all(value<=a[key] for key,value in b.items())


def compare_capture(evaluated):
    """Do not hide a newly missed utterance behind unchanged pass totals."""
    rows={version:{r['case']:r for r in evaluated if r['version']==version} for version in ['baseline','candidate']}
    if rows['baseline'].keys()!=rows['candidate'].keys():raise RuntimeError('Unpaired capture comparison')
    lost=[key for key,a in rows['baseline'].items() if a['passed'] and not rows['candidate'][key]['passed']]
    missing=[key for key,a in rows['baseline'].items() if a['clips']>0 and rows['candidate'][key]['clips']==0]
    def onset(row):return row.get('detection_s') if row.get('mode')=='fresh' else row['barge_s']
    onset_changes=[dict(case=key,before=onset(a),after=onset(rows['candidate'][key]))
        for key,a in rows['baseline'].items() if onset(a)!=onset(rows['candidate'][key])]
    return dict(lost_baseline_passes=lost,new_missing_clips=missing,onset_changes=onset_changes,
        full_gate_passed=bool(rows['candidate']) and all(r['passed'] for r in rows['candidate'].values()) and not missing)


def silent_control_passed(row,events):
    return bool(not row['false_cutoff'] and row['clips']==0 and not row['error'] and not row['aec_error']
                and not any(event['event']=='speech_start' for event in events))


def capture(folder,extension,capture_postfiltered=False,endpoint_ms=480,minimum_guard_correlation=None,correlation_window_ms=100,validation_voices=False,baseline_source=None,production_candidate=False,candidate_source=None,pre_roll_ms=250,barge_endpoint_ms=None,fresh_listening=False):
    folder.mkdir(parents=True,exist_ok=False);capture_files(folder/'source')
    if candidate_source:
        shutil.copy2(candidate_source,folder/'candidate-duplex.py')
        spec=importlib.util.spec_from_file_location('candidate_duplex',candidate_source.resolve())
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);Candidate,instances=module.DuplexFrontend,[]
    elif production_candidate:Candidate,instances=DuplexFrontend,[]
    else:Candidate,instances=make_candidate(folder,extension,True,capture_postfiltered,minimum_guard_correlation,correlation_window_ms)
    if barge_endpoint_ms is not None:Candidate=contextual_frontend(Candidate,barge_endpoint_ms)
    Baseline=DuplexFrontend
    if baseline_source:
        shutil.copy2(baseline_source,folder/'baseline-duplex.py')
        spec=importlib.util.spec_from_file_location('baseline_duplex',baseline_source.resolve())
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);Baseline=module.DuplexFrontend
    origin=ROOT/'outputs/experiments/synthetic-ad-v1';protocol=json.loads((origin/'protocol.json').read_text())
    scripts={r['id']:r['script'] for r in protocol['cases']}
    fixtures={}
    definitions=[
        ('original',ROOT/'data/synth/user_barge.wav','Wait, stop, let me think for a second.',['stop','think']),
        ('david-C2',origin/'inputs/david-C2.wav',scripts['C2'],['12','5','apples']),
        ('zira-A4',origin/'inputs/zira-A4.wav',scripts['A4'],['banana'])]
    if validation_voices:
        definitions=[('david-C4',origin/'inputs/david-C4.wav',scripts['C4'],['coffee','small','tea']),
                     ('zira-A6',origin/'inputs/zira-A6.wav',scripts['A6'],['river','window','orange'])]
    for key,path,script,required in definitions:
        with wave.open(str(path),'rb') as reader:
            assert reader.getframerate()==16000 and reader.getnchannels()==1
            audio=np.frombuffer(reader.readframes(reader.getnframes()),dtype='<i2').astype(np.float32)/32768
        first=np.flatnonzero(abs(audio)>.02)[0];audio=audio[max(0,first-160):]
        factor=1.
        if key!='original':factor=.12/float(np.sqrt(np.mean(audio[abs(audio)>.01]**2)));audio=audio*factor
        fixtures[key]=dict(audio=audio,script=script,required=required,normalization_factor=factor)
        shutil.copy2(path,folder/(key+'-source.wav'))
    definitions=[]
    for key in fixtures:
        for delay in [80,180]:
            for gain in [.15,.3]:definitions.append(dict(voice=key,delay_ms=delay,user_gain=gain,echo_gain=1.))
    for key in [key for key in fixtures if key!='original']:
        definitions.append(dict(voice=key,delay_ms=140,user_gain=.15,echo_gain=.8))
        definitions.append(dict(voice=key,delay_ms=80,user_gain=.15,echo_gain=0.))
    for delay in [80,180]:definitions.append(dict(voice=None,delay_ms=delay,user_gain=0.,echo_gain=1.))
    (folder/'protocol.json').write_text(json.dumps(dict(cases=definitions,frame_samples=320,onset_s=1.,seed=20261018,
        scripts={key:{k:v for k,v in value.items() if k!='audio'} for key,value in fixtures.items()},
        extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest() if not production_candidate and not candidate_source else None,
        baseline_sha256=hashlib.sha256((baseline_source or ROOT/'src/duplex.py').read_bytes()).hexdigest(),
        production_candidate=production_candidate,
        candidate_source_sha256=hashlib.sha256(candidate_source.read_bytes()).hexdigest() if candidate_source else None,
        baseline_pre_roll_ms=250,candidate_pre_roll_ms=pre_roll_ms,
        candidate_barge_endpoint_ms=barge_endpoint_ms,mode='fresh' if fresh_listening else 'interruption',
        candidate_capture_signal='source-defined' if candidate_source else 'production-postfiltered' if production_candidate else 'postfiltered' if capture_postfiltered else 'linear',candidate_endpoint_ms=endpoint_ms,
        minimum_guard_correlation=minimum_guard_correlation,
        correlation_window_ms=correlation_window_ms,
        validation_voices=validation_voices,
        scope='Real app capture, endpointing, RMS submission gate and interruption; virtual devices, no generation. Audio-time stepping waits for each capture iteration.'),indent=2))
    base=read('_assistant.wav');shutil.copy2(ROOT/'data/synth/_assistant.wav',folder/'far-source.wav')
    with (folder/'capture-results.jsonl').open('x') as output:
        for index,definition in enumerate(definitions):
            case=folder/f'case-{index:02d}';case.mkdir();voice=definition['voice'];onset=16000
            user=fixtures[voice]['audio']*definition['user_gain'] if voice else np.zeros(0,np.float32)
            size=round(max(8*16000,onset+len(user)+2*16000)/320)*320
            noise=np.random.default_rng(20261018+index).normal(0,.0005,size).astype(np.float32)
            far_signal=np.resize(base,size)
            near=np.zeros(size,np.float32);near[onset:onset+len(user)]=user
            if voice:write_wav(case/'reference.wav',np.concatenate((np.zeros(4800),user,np.zeros(16000))))
            for version,frontend in [('baseline',Baseline),('candidate',Candidate)]:
                destination=case/version;destination.mkdir();front=frontend(stream_delay_ms=80,end_silence_ms=endpoint_ms if version=='candidate' else 480);front.warmup()
                mic=VirtualMic();speaker=VirtualSpeaker();speaker.playing=not fresh_listening;diag=CaptureDiagnostics(destination/'events.jsonl',speaker)
                app=CaptureAssistant(mic=mic,player=speaker,front=front,client=NoBackend(),diagnostics=diag,pre_roll_ms=pre_roll_ms if version=='candidate' else 250)
                app.status='listening' if fresh_listening else 'speaking';job=Job(id=1);app._active=None if fresh_listening else job;app._spawn(app._loop)
                render=np.zeros(size,np.float32);raw=np.zeros(size,np.float32)
                try:
                    for offset in range(0,size,320):
                        speaker.sample_clock=offset+320
                        if speaker.playing:
                            render[offset:offset+320]=far_signal[offset:offset+320]
                            front.push_far(render[offset:offset+320]);job.played_samples+=480
                        block=near[offset:offset+320].copy()+noise[offset:offset+320]
                        for tail,scale in [(0,1),(25,.3),(70,.15)]:
                            delay=(definition['delay_ms']+tail)*16;start=offset-delay;end=offset+320-delay
                            if end>0:block[max(0,-start):]+=render[max(0,start):max(0,end)]*definition['echo_gain']*scale
                        raw[offset:offset+320]=block;mic.q.put(block,timeout=10);mic.q.wait_completed(offset//320+1)
                    stop=speaker.stop_sample
                    first_speech=next((e['audio_clock_s'] for e in diag.recent(200) if e['event']=='speech_start'),None)
                    for clip_index,audio in enumerate(app.captured):write_wav(destination/f'clip-{clip_index:02d}.wav',audio)
                    np.savez_compressed(destination/'signals.npz',near=near,raw=raw,render=render)
                    row=dict(case=index,version=version,definition=definition,stop_s=stop/16000 if stop is not None else None,
                        mode='fresh' if fresh_listening else 'interruption',detection_s=round(first_speech-onset/16000,3) if first_speech is not None and voice else None,
                        first_speech_s=first_speech,
                        false_cutoff=stop is not None and (not voice or stop<onset),
                        barge_s=round((stop-onset)/16000,3) if stop is not None and voice else None,
                        clips=len(app.captured),clip_samples=[len(a) for a in app.captured],
                        clip_submission_s=[sample/16000 for sample in app.capture_end_samples],
                        clip_rms=[float(np.sqrt(np.mean(a**2))) for a in app.captured],error=app.error,aec_error=front.aec_error)
                    output.write(json.dumps(row)+'\n');output.flush();os.fsync(output.fileno());print(json.dumps(row),flush=True)
                finally:app.close();instances.clear()
    with (folder/'capture-complete.json').open('x') as complete:
        complete.write(json.dumps(dict(cases=len(definitions),rows=2*len(definitions)),indent=2))


def transcribe(folder):
    from faster_whisper import WhisperModel
    from importlib.metadata import version
    revision=WHISPER_REVISION
    model_path=Path.home()/'.cache/huggingface/hub/models--Systran--faster-whisper-base.en/snapshots'/revision
    model=WhisperModel(str(model_path),device='cpu',compute_type='int8',cpu_threads=2,num_workers=1,local_files_only=True)
    protocol=json.loads((folder/'protocol.json').read_text());rows=[json.loads(line) for line in (folder/'capture-results.jsonl').read_text().splitlines()]
    completion=json.loads((folder/'capture-complete.json').read_text())
    if len(rows)!=completion['rows'] or len(rows)!=2*len(protocol['cases']):
        raise RuntimeError('Capture results are incomplete')
    target=folder/'asr-results.jsonl';existing={r['key']:r for r in map(json.loads,target.read_text().splitlines())} if target.exists() else {}
    tasks=[]
    for row in rows:
        case=folder/f"case-{row['case']:02d}"
        if row['definition']['voice'] and row['version']=='baseline':tasks.append((f"{row['case']}-reference",case/'reference.wav'))
        for clip in sorted((case/row['version']).glob('clip-*.wav')):tasks.append((f"{row['case']}-{row['version']}-{clip.stem}",clip))
    with target.open('a') as stream:
        for key,path in tasks:
            if key in existing:
                if existing[key]['sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():
                    raise RuntimeError('Audio changed after its saved transcription: '+key)
                continue
            segments,info=model.transcribe(str(path),language='en',beam_size=5,temperature=0,
                condition_on_previous_text=False,vad_filter=False)
            segments=list(segments);result=dict(key=key,path=str(path.relative_to(folder)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                text=' '.join(s.text.strip() for s in segments),segments=[dict(start=s.start,end=s.end,text=s.text,avg_logprob=s.avg_logprob,no_speech_prob=s.no_speech_prob) for s in segments])
            stream.write(json.dumps(result)+'\n');stream.flush();os.fsync(stream.fileno());existing[key]=result;print(json.dumps(result),flush=True)
    score_capture(folder,existing)


def score_capture(folder,existing,output_name='summary.json'):
    from importlib.metadata import version
    protocol=json.loads((folder/'protocol.json').read_text());rows=[json.loads(line) for line in (folder/'capture-results.jsonl').read_text().splitlines()]
    if len(rows)!=2*len(protocol['cases']):raise RuntimeError('Cannot score incomplete capture results')
    evaluated=[]
    for row in rows:
        result=dict(row);voice=row['definition']['voice']
        if voice:
            script=protocol['scripts'][voice];reference=existing[f"{row['case']}-reference"]['text']
            result['source_control']=word_errors(script['script'],reference)
            result['source_control_valid']=bool(result['source_control']['wer']<=.1 and all(t in words(reference) for t in script['required']) and no_new_negation(script['script'],reference))
            clips=[existing[f"{row['case']}-{row['version']}-clip-{i:02d}"]['text'] for i in range(row['clips'])]
            result['transcripts']=clips;result['capture_words']=word_errors(script['script'],' '.join(clips));result['critical_words_present']=all(token in words(' '.join(clips)) for token in script['required'])
            result['no_new_negation']=no_new_negation(script['script'],' '.join(clips))
            timing=row.get('detection_s') if row.get('mode')=='fresh' else row['barge_s']
            result['passed']=bool(result['source_control_valid'] and row['clips']==1 and not row['false_cutoff'] and timing is not None and 0<=timing<.5 and result['capture_words']['wer']<=.2 and result['critical_words_present'] and result['no_new_negation'] and not row['error'] and not row['aec_error'])
        else:
            events=[json.loads(line) for line in (folder/f"case-{row['case']:02d}"/row['version']/'events.jsonl').read_text().splitlines()]
            result['false_speech_start']=any(event['event']=='speech_start' for event in events)
            result['passed']=silent_control_passed(row,events)
        evaluated.append(result)
    report=dict(cases=evaluated,summary={v:dict(passed=sum(r['passed'] for r in evaluated if r['version']==v),total=sum(r['version']==v for r in evaluated),missing_clips=sum(r['definition']['voice'] is not None and not r['clips'] for r in evaluated if r['version']==v)) for v in ['baseline','candidate']},
        comparison=compare_capture(evaluated),comparison_version=1,
        scoring_version=4,invalid_source_cases=sorted({r['case'] for r in evaluated if r['definition']['voice'] and not r['source_control_valid']}),
        model=dict(name='Systran/faster-whisper-base.en',revision=WHISPER_REVISION,packages={p:version(p) for p in ['faster-whisper','ctranslate2','av']}),deployed=False,
        scope='Capture-quality/latency gate in synthetic closed-loop acoustics. ASR is an independent consistency check, not human ground truth or a physical-room validation.')
    with (folder/output_name).open('x') as output:output.write(json.dumps(report,indent=2))
    print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);parser.add_argument('--extension',type=Path);parser.add_argument('--capture-postfiltered',action='store_true');parser.add_argument('--endpoint-ms',type=int,choices=[480,800,1120],default=480);parser.add_argument('--minimum-guard-correlation',type=float);parser.add_argument('--correlation-window-ms',type=int,default=100);parser.add_argument('--validation-voices',action='store_true');parser.add_argument('--baseline-source',type=Path);parser.add_argument('--production-candidate',action='store_true');parser.add_argument('--candidate-source',type=Path);parser.add_argument('--pre-roll-ms',type=int,default=250);parser.add_argument('--barge-endpoint-ms',type=int,choices=[480,800,1120]);parser.add_argument('--fresh-listening',action='store_true');group=parser.add_mutually_exclusive_group();group.add_argument('--asr-only',action='store_true');group.add_argument('--rescore-to');args=parser.parse_args()
    if args.candidate_source and args.production_candidate:parser.error('Choose one candidate source')
    if not args.asr_only and not args.rescore_to and (args.folder.exists() or not args.production_candidate and not args.candidate_source and args.extension is None):parser.error('Use a new experiment folder and provide an extension or candidate source')
    if args.asr_only and (not (args.folder/'capture-complete.json').exists() or (args.folder/'summary.json').exists()):parser.error('Need completed capture results without a final ASR summary')
    if args.rescore_to and (Path(args.rescore_to).name!=args.rescore_to or (args.folder/args.rescore_to).exists()):parser.error('Choose a new summary filename inside the experiment')
    try:
        if args.rescore_to:score_capture(args.folder,{r['key']:r for r in map(json.loads,(args.folder/'asr-results.jsonl').read_text().splitlines())},args.rescore_to)
        elif args.asr_only:transcribe(args.folder)
        else:capture(args.folder,args.extension,args.capture_postfiltered,args.endpoint_ms,args.minimum_guard_correlation,args.correlation_window_ms,args.validation_voices,args.baseline_source,args.production_candidate,args.candidate_source,args.pre_roll_ms,args.barge_endpoint_ms,args.fresh_listening)
    except Exception:
        if args.folder.exists():
            import traceback
            (args.folder/f'failure-{time.time_ns()}.txt').write_text(traceback.format_exc())
        raise
    finally:
        if args.folder.exists():print('Verified backup:',checkpoint(args.folder),flush=True)
