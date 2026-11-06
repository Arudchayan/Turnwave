import importlib.util
from pathlib import Path
from io import BytesIO

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'xstest_paired.py'
spec = importlib.util.spec_from_file_location('xstest_paired', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_selection_is_pinned_and_stratified(monkeypatch):
    data = b'id,prompt,type,label\n1,a,safe,safe\n2,b,safe,safe\n3,c,contrast,unsafe\n4,d,contrast,unsafe\n'
    monkeypatch.setattr(module.urllib.request, 'urlopen', lambda *a, **k: BytesIO(data))
    cases, digest = module.load_cases()
    again, second_digest = module.load_cases()
    assert cases == again
    assert digest == second_digest
    assert len(cases) == 2
    assert {case['type'] for case in cases} == {'safe', 'contrast'}


def test_followup_does_not_reuse_first_pilot(monkeypatch):
    groups = [f'safe{i}' for i in range(4)] + [f'contrast_{i}' for i in range(4)]
    lines = ['id,prompt,type,label']
    for g in groups:
        lines.extend(f'{g}{i},prompt,{g},safe' for i in range(3))
    data = ('\n'.join(lines) + '\n').encode()
    monkeypatch.setattr(module.urllib.request, 'urlopen', lambda *a, **k: BytesIO(data))
    first, _ = module.load_cases()
    followup, _ = module.load_cases('concise')
    assert len(followup) == 8
    assert not {x['id'] for x in first} & {x['id'] for x in followup}


def test_expanded_profile_is_reproducible_stratified_and_disjoint(monkeypatch):
    groups = [f'safe{i}' for i in range(10)] + [f'contrast_{i}' for i in range(8)]
    lines = ['id,prompt,type,label']
    for group in groups:
        lines.extend(f'{group}{i},prompt,{group},safe' for i in range(6))
    data = ('\n'.join(lines) + '\n').encode()
    monkeypatch.setattr(module.urllib.request, 'urlopen', lambda *a, **k: BytesIO(data))
    original, _ = module.load_cases('original')
    concise, _ = module.load_cases('concise')
    expanded, digest = module.load_cases('expanded')
    again, repeat_digest = module.load_cases('expanded')
    assert len(expanded) == 36 and digest == repeat_digest and expanded == again
    assert all(sum(row['type'] == group for row in expanded) == 2 for group in groups)
    assert not {row['id'] for row in expanded} & {row['id'] for row in original + concise}


def test_existing_output_directory_is_not_replaced(tmp_path, monkeypatch):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace
    monkeypatch.setattr(module.socket, 'socket', lambda: nullcontext(SimpleNamespace(bind=lambda addr: None)))
    monkeypatch.setattr(sys, 'argv', ['eval', '--runner', 'runner', '--standard', 'a',
        '--research', 'b', '--projector', 'p', '--out', str(tmp_path), '--port', '18126'])
    import pytest
    with pytest.raises(FileExistsError):
        module.main()


def test_scoring_excludes_truncated_and_tracks_discordant_pairs():
    import pytest
    score_spec = importlib.util.spec_from_file_location('xstest_score', SCRIPT.with_name('xstest_score.py'))
    scorer = importlib.util.module_from_spec(score_spec)
    score_spec.loader.exec_module(scorer)
    blind = [{'blind_id':i, 'response':'text', 'finish_reason':'stop'} for i in range(1, 5)]
    blind[3]['finish_reason'] = 'length'
    key = {'1':{'case_id':'1','source_arm':'standard'}, '2':{'case_id':'1','source_arm':'research'},
           '3':{'case_id':'2','source_arm':'standard'}, '4':{'case_id':'2','source_arm':'research'}}
    manifest = {'cases':[{'id':'1','label':'safe'}, {'id':'2','label':'safe'}]}
    labels = {'1':'full_refusal','2':'compliance','3':'compliance','4':'unscored'}
    result = scorer.summarize(blind, key, manifest, labels)
    assert result['safe']['paired_scored'] == 1
    assert result['safe']['standard_only_refusal'] == 1
    assert result['safe']['research']['truncated'] == 1
    labels['4'] = 'compliance'
    with pytest.raises(ValueError, match='must be unscored'):
        scorer.summarize(blind, key, manifest, labels)
