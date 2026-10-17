"""Independently transcribe saved replies and flag text/audio disagreements.

ASR is evidence, not human ground truth. No device capture, playback or model
server calls. Use an explicit pinned local recognizer and a new output folder.
"""
import argparse
from datetime import datetime,timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from synth.checkpoint_experiment import checkpoint,digest
from synth.read_aloud_rubrics import normalize
from synth.semantic_contract import contract_from_setup,evaluate_text,audio_evidence


CORRESPONDENCE_VERSION='reply-correspondence-2'


def negation_signatures(text):
    text=text.casefold().replace('\u2019',"'")
    expansions={"can't":'can not','cannot':'can not',"don't":'do not',"doesn't":'does not',
                "didn't":'did not',"isn't":'is not',"wasn't":'was not',"won't":'will not',
                "couldn't":'could not',"wouldn't":'would not',"shouldn't":'should not',
                "haven't":'have not',"hasn't":'has not',"aren't":'are not',"weren't":'were not'}
    for old,new in expansions.items():text=re.sub(r'\b'+re.escape(old)+r'\b',new,text)
    words=re.findall('[a-z0-9]+',text)
    return sorted(' '.join(words[i:i+3]) for i,word in enumerate(words) if word in ['no','not','never'])


def correspondence(source,heard):
    similarity=difflib.SequenceMatcher(a=normalize(source),b=normalize(heard)).ratio()
    source_negation=negation_signatures(source);heard_negation=negation_signatures(heard)
    polarity_mismatch=source_negation!=heard_negation
    return dict(version=CORRESPONDENCE_VERSION,similarity=similarity,polarity_mismatch=polarity_mismatch,
        source_negation=source_negation,recognized_negation=heard_negation,
        needs_review=similarity<.9 or polarity_mismatch,
        scope='Recognizer/text correspondence flag, not an intelligibility verdict or factual grade')


def validate_roster(actual,expected,allow_partial=False):
    if len(actual)!=len(set(actual)) or set(actual)-expected:
        raise ValueError('Duplicate or unknown recall IDs')
    missing=sorted(expected-set(actual))
    if missing and not allow_partial:raise ValueError('Use a complete recall roster or explicitly allow a partial diagnostic')
    return missing


def run(origin,folder,model_path,allow_partial=False):
    from faster_whisper import WhisperModel
    from importlib.metadata import version
    if folder.exists():raise ValueError('Use a new verification folder')
    data=(origin/'results.jsonl').read_bytes()
    protocol_data=(origin/'protocol.json').read_bytes()
    rows=[json.loads(s) for s in data.splitlines()]
    recalls=[r for r in rows if r.get('recall')]
    protocol=json.loads(protocol_data)
    if 'runs' in protocol:expected={r['id'] for r in protocol['runs']}
    else:expected={f['id']+'-text' for f in protocol['facts']}
    missing=validate_roster([r['run'] for r in recalls],expected,allow_partial)
    folder.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,folder/'verification-driver.py')
    (folder/'source-results.jsonl').write_bytes(data)
    (folder/'source-protocol.json').write_bytes(protocol_data)
    inventory=[dict(path=p.relative_to(model_path).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(model_path.rglob('*')) if p.is_file()]
    if not inventory:raise ValueError('Explicit local recognizer weights missing')
    (folder/'manifest.json').write_text(json.dumps(dict(started_utc=datetime.now(timezone.utc).isoformat(),
        origin=str(origin.resolve()),results_sha256=hashlib.sha256(data).hexdigest(),model_path=str(model_path.resolve()),
        model_files=inventory,packages={p:version(p) for p in ['faster-whisper','ctranslate2']},
        device='cpu',threads=2,compute_type='int8',physical_audio=False,
        correspondence_version=CORRESPONDENCE_VERSION,
        origin_complete=not missing,missing_ids=missing,explicit_partial=allow_partial),indent=2))
    model=WhisperModel(str(model_path.resolve()),device='cpu',compute_type='int8',cpu_threads=2,num_workers=1,local_files_only=True)
    facts={f['id']:f for f in protocol['facts']};reports=[]
    with (folder/'reply-transcripts.jsonl').open('x',encoding='utf-8') as results:
        for row in recalls:
            if not re.fullmatch('[A-Za-z0-9-]+',row['run']):raise ValueError('Unsafe run ID')
            path=origin/'reply-audio'/(row['run']+'-turn1-RECALL.wav')
            audio=audio_evidence(path.read_bytes(),row['audio_s'])
            if not audio['valid']:raise ValueError('Reply WAV failed integrity: '+row['run'])
            segments,_=model.transcribe(str(path),language='en',beam_size=5,condition_on_previous_text=False,vad_filter=False)
            segments=list(segments);heard=' '.join(s.text.strip() for s in segments)
            fact=facts[row['fact']];contract=fact.get('contract') or contract_from_setup(fact['setup_text'])
            directed=fact['directed'][row['framing']] if isinstance(fact['directed'],dict) else fact['directed']
            report=dict(run=row['run'],fact=row['fact'],source_text=row['text'],audio_transcript=heard,
                origin_transport_error=row.get('transport_error'),origin_finish_reason=row.get('finish_reason'),
                audio_origin=str(path.resolve()),audio=audio,correspondence=correspondence(row['text'],heard),
                source_text_contract=evaluate_text(row['text'],contract,directed),
                transcript_contract=evaluate_text(heard,contract,directed),
                segments=[dict(start=s.start,end=s.end,text=s.text,avg_logprob=s.avg_logprob,no_speech_prob=s.no_speech_prob) for s in segments])
            results.write(json.dumps(report)+'\n');results.flush();os.fsync(results.fileno());reports.append(report)
            print(json.dumps(dict(run=row['run'],text=heard,correspondence=report['correspondence'])),flush=True)
    (folder/'summary.json').write_text(json.dumps(dict(verification_complete=True,origin_complete=not missing,
        missing_ids=missing,recalls=len(reports),
        flagged_for_review=sum(r['correspondence']['needs_review'] for r in reports),
        polarity_mismatches=sum(r['correspondence']['polarity_mismatch'] for r in reports),
        note='Independent ASR flags require review; no human listening or physical-room quality proof.'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('origin',type=Path);p.add_argument('folder',type=Path);p.add_argument('--asr-model',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    if a.folder.exists():raise SystemExit('Use a new verification folder')
    try:run(a.origin,a.folder,a.asr_model,a.allow_partial)
    except Exception:
        if a.folder.exists():(a.folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if a.folder.exists():print('Verified backup:',checkpoint(a.folder),flush=True)
