import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from synth.resume_semantic_matrix import missing_runs
from synth.combine_semantic_continuation import complete_pairs


def test_continuation_restarts_orphaned_setup_from_fresh_context():
    runs=[dict(id='finished'),dict(id='orphan'),dict(id='new')]
    rows=[dict(run='finished',case='SETUP',recall=False),
          dict(run='finished',case='R-x',recall=True),
          dict(run='orphan',case='SETUP',recall=False)]
    assert missing_runs(runs,rows)==runs[1:]


@pytest.mark.parametrize('rows',[
    [dict(run='a',recall=True)],
    [dict(run='a',case='SETUP'),dict(run='a',recall=True),dict(run='a',recall=True)],
    [dict(run='unknown',recall=True)],
])
def test_continuation_rejects_ambiguous_completed_context(rows):
    with pytest.raises(ValueError):missing_runs([dict(id='a')],rows)


def test_projection_excludes_old_orphan_setup_and_retains_pair_origin():
    metadata=dict(fact='OWN-1',fact_type='OWN',framing='FA',voice='david',rep=0)
    roster=[dict(id=k,**metadata) for k in ['a','b']]
    def setup(k,text):return dict(run=k,case='SETUP',recall=False,text=text,**metadata)
    def recall(k):return dict(run=k,case='R-x',recall=True,**metadata)
    original=[setup('a','old'),recall('a'),setup('b','unfinished')]
    continuation=[setup('b','fresh'),recall('b')]
    selected,origins=complete_pairs(roster,original,continuation)
    assert origins==dict(a='original',b='continuation')
    assert [r.get('text') for r in selected if not r['recall']]==['old','fresh']
    with pytest.raises(ValueError):complete_pairs(roster,original,[])
    with pytest.raises(ValueError):complete_pairs(roster,original,original[:2])
