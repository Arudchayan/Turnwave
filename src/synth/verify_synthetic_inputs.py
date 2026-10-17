"""Independent CPU transcription of saved synthetic inputs; never calls chat."""
import argparse
from datetime import datetime,timezone
import difflib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from read_aloud_rubrics import normalize


def main(folder):
    from faster_whisper import WhisperModel
    target=folder/'independent-asr.jsonl'
    if target.exists():raise FileExistsError('Independent transcription already exists; preserve it')
    model=WhisperModel('base.en',device='cpu',compute_type='int8',cpu_threads=2,num_workers=1)
    manifest=dict(started_utc=datetime.now(timezone.utc).isoformat(),model='base.en',backend='faster-whisper',version=importlib.metadata.version('faster-whisper'),device='cpu',compute_type='int8',cpu_threads=2,
                  note='Independent recognizer output is evidence, not human ground truth. No audio uploaded; model weights downloaded for local use.')
    (folder/'independent-asr-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'));rows=[]
    for voice in protocol['voices']:
        for case in protocol['cases']:
            key=voice['id']+'-'+case['id'];path=folder/'inputs'/(key+'.wav')
            segments,info=model.transcribe(str(path),language='en',beam_size=5,condition_on_previous_text=False,vad_filter=False)
            segments=list(segments);text=' '.join(segment.text.strip() for segment in segments)
            ratio=difflib.SequenceMatcher(a=normalize(case['spoken_text']),b=normalize(text)).ratio()
            row=dict(key=key,script=case['spoken_text'],transcript=text,similarity=round(ratio,3),input_check_pass=ratio>=.9,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     segments=[dict(start=s.start,end=s.end,text=s.text,avg_logprob=s.avg_logprob,no_speech_prob=s.no_speech_prob) for s in segments])
            with target.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
            rows.append(row);print(key,round(ratio,3),text,flush=True)
    (folder/'independent-asr-summary.json').write_text(json.dumps(dict(checked=len(rows),matched=sum(r['input_check_pass'] for r in rows),flagged=sum(not r['input_check_pass'] for r in rows)),indent=2),encoding='utf-8')
    from checkpoint_experiment import checkpoint
    print('Verified backup:',checkpoint(folder),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);args=parser.parse_args();main(args.folder)
