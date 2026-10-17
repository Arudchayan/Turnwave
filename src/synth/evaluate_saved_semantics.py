"""Apply a new contextual contract equally to complete saved arms, without inference.

Automatic outcomes are bounded by the grammar. Manual rows remain unresolved;
reported confirmed passes are a lower bound, not a complete accuracy estimate.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from synth.audit_semantic_matrix import recalls,validate_protocols
from synth.checkpoint_experiment import checkpoint
from synth.semantic_contract import VERSION,contract_from_setup,evaluate,audio_evidence


def run(folder,arms):
    if folder.exists():raise ValueError('Use a new evaluation folder')
    snapshots=[]
    for arm in arms:
        data=(arm/'results.jsonl').read_bytes()
        rows,partial=recalls(data)
        if partial:raise ValueError('Saved arm has a partial final line')
        protocol=json.loads((arm/'protocol.json').read_text())
        validate_protocols(rows,rows,protocol,protocol)
        if len(rows)!=len(protocol['runs']):raise ValueError('Use complete saved arms')
        contracts={f['id']:dict(contract=contract_from_setup(f['setup_text']),directed=f['directed']) for f in protocol['facts']}
        if snapshots:
            validate_protocols(snapshots[0]['rows'],rows,snapshots[0]['protocol'],protocol)
            if contracts!=snapshots[0]['contracts']:raise ValueError('Source facts or directed contracts differ across arms')
        snapshots.append(dict(origin=arm,data=data,rows=rows,protocol=protocol,contracts=contracts))
    folder.mkdir(parents=True,exist_ok=False)
    for name in ['semantic_contract.py','evaluate_saved_semantics.py']:
        shutil.copy2(ROOT/'src/synth'/name,folder/name)
    summary=dict(version=VERSION,scope='Offline bounded-grammar contextual review. No inference or physical audio. Manual means unresolved; no general accuracy claim.',arms={})
    for index,snapshot in enumerate(snapshots):
        label=f'arm{index}';origin=snapshot['origin'];rows=snapshot['rows'];reports=[]
        (folder/(label+'-results.jsonl')).write_bytes(snapshot['data'])
        (folder/(label+'-protocol.json')).write_text(json.dumps(snapshot['protocol'],indent=2))
        (folder/(label+'-contracts.json')).write_text(json.dumps(snapshot['contracts'],indent=2))
        for run_id,row in rows.items():
            if not re.fullmatch(r'[A-Za-z0-9-]+',run_id):raise ValueError('Unsafe run ID')
            audio_path=origin/'reply-audio'/(run_id+'-turn1-RECALL.wav')
            try:
                audio=audio_evidence(audio_path.read_bytes(),row['audio_s'])
            except OSError as exc:
                audio=dict(valid=False,error=str(exc))
            transport=bool(audio['valid'] and not row.get('error') and not row.get('truncated')
                           and not row.get('transport_error') and row.get('finish_reason')=='stop')
            fact=snapshot['contracts'][row['fact']]
            report=evaluate(row['text'],fact['contract'],fact['directed'][row['framing']],transport)
            report.update(run=run_id,fact=row['fact'],framing=row['framing'],voice=row['voice'],rep=row['rep'],
                          text=row['text'],historical_passed=row['passed'],audio=audio,audio_origin=str(audio_path.resolve()))
            reports.append(report)
        (folder/(label+'-review.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in reports))
        queue=[r for r in reports if r['semantic_review_required'] or r['instructions_pass'] is None]
        (folder/(label+'-manual-queue.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in queue))
        summary['arms'][label]=dict(origin=str(origin.resolve()),results_sha256=hashlib.sha256(snapshot['data']).hexdigest(),
            recalls=len(rows),outcomes=dict(Counter(r['outcome'] for r in reports)),manual_semantic_rows=len(queue),
            historical_passed=sum(r['passed'] is True for r in rows.values()),
            historical_pass_to_manual=sum(r['historical_passed'] is True and r['outcome']=='manual' for r in reports),
            invalid_audio_rows=sum(not r['audio']['valid'] for r in reports),
            attribution_failures=sum(r['attribution_pass'] is False for r in reports),
            known_format_failures=sum(r['instructions_pass'] is False for r in reports))
    (folder/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('arms',type=Path,nargs='+');a=p.parse_args()
    if a.folder.exists():raise SystemExit('Use a new evaluation folder')
    try:run(a.folder,a.arms)
    except Exception:
        if a.folder.exists():(a.folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if a.folder.exists():print('Verified backup:',checkpoint(a.folder),flush=True)
