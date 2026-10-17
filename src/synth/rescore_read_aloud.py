"""Regrade saved replies without rerunning inference or overwriting old scores."""
import argparse
import hashlib
import json
from pathlib import Path
from read_aloud_rubrics import grade, VERSION


def rescore(folder):
    target=folder/('rescored-v'+VERSION+'.jsonl')
    if target.exists():raise FileExistsError('Scores already exist; retain them and use a new rubric version')
    source=folder/'results.jsonl';rows=[]
    validation_path=folder/'independent-asr.jsonl'
    independent={row['key']:row for row in map(json.loads,validation_path.read_text(encoding='utf-8').splitlines())} if validation_path.exists() else None
    with target.open('x',encoding='utf-8') as stream:
        for line in source.read_text(encoding='utf-8').splitlines():
            original=json.loads(line);row=dict(original);row['rubric']=grade(row['case'],row['text'])
            if independent is not None:
                check=independent.get(row['key'],{});audio=folder/'inputs'/(row['key']+'.wav')
                row['independent_fixture_check_pass']=bool(check.get('input_check_pass') and audio.is_file()
                    and hashlib.sha256(audio.read_bytes()).hexdigest()==check.get('sha256'))
            value=row['rubric']['passed'];row['passed']=bool(value and not row['error'] and row['audio_s']>0) if value is not None else None
            rows.append(row);stream.write(json.dumps(row)+'\n')
    report=dict(rubric_version=VERSION,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                passed=sum(row['passed'] is True for row in rows),failed=sum(row['passed'] is False for row in rows),manual_review=sum(row['passed'] is None for row in rows),
                setup_only=sum(row['rubric'].get('setup_only',False) for row in rows),
                input_validation_pending=sum(not row.get('independent_fixture_check_pass',row['fixture_check_pass']) for row in rows),
                independent_validation_used=independent is not None,
                transport_errors=sum(bool(row['error']) or row['audio_s']<=0 for row in rows),
                correction='V3 rejects incorrect speaker ownership and echoed user instructions in setup responses. Setup presence is not proof of recall. Raw outputs and earlier scores remain intact.')
    (folder/('summary-v'+VERSION+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);args=parser.parse_args();rescore(args.folder)
