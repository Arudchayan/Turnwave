import json
from pathlib import Path
import sys

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from synth.audit_semantic_matrix import recalls,compare,validate_protocols
from synth.snapshot_live_matrix import completed_files


def row(fact='OWN-1',rep=0,passed=False):
    return dict(run=f'{fact}-FA-david-r{rep}',recall=True,fact=fact,voice='david',framing='FA',rep=rep,passed=passed)


def test_duplicate_recall_is_rejected_instead_of_overwriting():
    data=(json.dumps(row())+'\n')*2
    with pytest.raises(ValueError,match='Duplicate'):recalls(data.encode())


def test_partial_matrix_cannot_produce_final_statistics():
    base={r['run']:r for r in [row(rep=0),row(rep=1)]};item=row(passed=True)
    report=compare(base,{item['run']:item})
    assert not report['complete'] and len(report['missing_ids'])==1
    assert 'fact_cluster_sensitivity' not in report and 'row_mcnemar_descriptive' not in report
    assert report['replication_groups']['groups_with_multiple_reps']==0
    assert report['replication_groups']['candidate_consistent_multiple_reps']==0


def test_repeated_successes_remain_one_fact_for_cluster_sensitivity():
    before=[row(fact=f,rep=r) for f in ['OWN-1','OWN-2'] for r in [0,1]]
    after=[dict(r,passed=True) for r in before]
    report=compare({r['run']:r for r in before},{r['run']:r for r in after})
    assert report['row_mcnemar_descriptive']['exact_p']==.125
    assert report['fact_cluster_sensitivity']['facts']==2
    assert report['fact_cluster_sensitivity']['exact_sign_flip_p']==.5


def test_mismatched_pair_metadata_is_rejected():
    a=row();b=dict(a,voice='zira')
    with pytest.raises(ValueError,match='Mismatched'):compare({a['run']:a},{b['run']:b})


def test_partial_final_json_line_is_preserved_but_excluded():
    data=(json.dumps(row())+'\n'+ '{"run":').encode()
    rows,partial=recalls(data)
    assert partial and len(rows)==1


def test_matching_incomplete_files_do_not_masquerade_as_complete_protocol():
    a=row();b=row(rep=1)
    protocol=dict(runs=[dict(a,id=a['run']),dict(b,id=b['run'])],grading_sha256='same',rubric_version='same')
    with pytest.raises(ValueError,match='Baseline matrix is incomplete'):
        validate_protocols({a['run']:a},{a['run']:a},protocol,protocol)


def test_live_snapshot_includes_only_outputs_referenced_by_completed_rows():
    setup=dict(row(),recall=False,case='SETUP');files=completed_files([setup])
    assert files==['OWN-1-FA-david-r0-turn0-SETUP-request.json','reply-audio/OWN-1-FA-david-r0-turn0-SETUP.wav']
    assert not any('turn1' in name for name in files)
    with pytest.raises(ValueError,match='Unsafe'):completed_files([dict(setup,run='../other')])
