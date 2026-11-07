import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'compare_lfm_variants.py'
spec = importlib.util.spec_from_file_location('lfm_paired_eval', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_keyword_proxy_is_not_a_semantic_judge():
    assert module.refusal_proxy("I can't help with that.")
    assert module.refusal_proxy('This may be illegal.')
    assert not module.refusal_proxy('A response without a listed marker.')


def test_control_scoring_is_strict_and_normalized():
    assert module.exact_answer('  Mars.  ', 'mars')
    assert not module.exact_answer('The answer is Mars.', 'mars')
    assert len(module.CONTROLS) == 16
    assert len({case[0] for case in module.CONTROLS}) == 16


def test_only_local_servers_are_accepted():
    with pytest.raises(ValueError, match='local HTTP'):
        module.validate_server('https://remote.example', '/model.gguf')
    with pytest.raises(ValueError, match='local HTTP'):
        module.validate_server('http://127.0.0.1@remote.example', '/model.gguf')


def make_arm(variant):
    row = lambda group, ident, marker=False, exact=False: {
        'group':group, 'id':ident, 'ok':True, 'empty':False,
        'refusal_marker':marker, 'exact':exact, 'finish_reason':'stop', 'duration_s':1.0}
    return dict(variant=variant, manifest_sha256='same', harm_revision='h', harm_split='s',
                benign_revision='b', benign_split='s', temperature=0, seed=42, max_tokens=64,
                model_sha256=variant, completed_utc='2026-01-01T00:00:00Z', limits='limits',
                rows=[row('harmful','h',True), row('benign','b'), row('control','c',exact=True)])


def test_comparison_keeps_failed_and_truncated_cases_visible():
    a, b = make_arm('standard'), make_arm('research')
    b['rows'][0]['ok'] = False
    a['rows'][1]['finish_reason'] = 'length'
    result = module.compare(a,b)
    assert result['groups']['harmful']['research']['errors'] == 1
    assert result['groups']['harmful']['paired_ok'] == 0
    assert result['groups']['benign']['standard']['truncated'] == 1
    assert result['groups']['control']['standard']['exact'] == 1


def test_comparison_rejects_mismatched_or_incomplete_runs():
    a, b = make_arm('standard'), make_arm('research')
    b['manifest_sha256'] = 'different'
    with pytest.raises(ValueError, match='Incompatible arms'):
        module.compare(a,b)
    b = make_arm('research'); b.pop('completed_utc')
    with pytest.raises(ValueError, match='complete'):
        module.compare(a,b)
    b = make_arm('research'); b['rows'].append(dict(b['rows'][0]))
    with pytest.raises(ValueError, match='unique'):
        module.compare(a,b)


def test_run_does_not_save_dataset_prompts_or_generated_responses(tmp_path, monkeypatch):
    from types import SimpleNamespace
    secrets = iter(([{'text':'PRIVATE HARM PROMPT'}], [{'text':'PRIVATE BENIGN PROMPT'}]))
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=lambda *a, **k: next(secrets)))
    monkeypatch.setattr(module, 'validate_server', lambda *a: '/model')
    monkeypatch.setattr(module, 'run_completion', lambda *a: ('PRIVATE GENERATED RESPONSE', 'stop', .1, 2))
    model = tmp_path / 'weights.gguf'; model.write_bytes(b'fixture')
    out = tmp_path / 'evidence.json'
    module.run_arm('standard', 'http://127.0.0.1:1', model, out, '/model')
    text = out.read_text(encoding='utf-8')
    assert 'PRIVATE HARM PROMPT' not in text
    assert 'PRIVATE BENIGN PROMPT' not in text
    assert 'PRIVATE GENERATED RESPONSE' not in text
    report = json.loads(text)
    assert len(report['rows']) == 18 and 'completed_utc' in report
    with pytest.raises(FileExistsError):
        module.run_arm('standard', 'http://127.0.0.1:1', model, out, '/model')


def test_extended_profile_uses_preselected_subsets_and_larger_cap(tmp_path, monkeypatch):
    from types import SimpleNamespace
    calls = []
    def load(repo, revision, split):
        calls.append((repo, revision, split))
        return [{'text': 'PRIVATE TEST PROMPT'}]
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=load))
    monkeypatch.setattr(module, 'validate_server', lambda *a: '/model')
    def completion(url, prompt, cap):
        assert cap == 192
        return 'PRIVATE OUTPUT', 'stop', .1, 1
    monkeypatch.setattr(module, 'run_completion', completion)
    model = tmp_path / 'weights.gguf'; model.write_bytes(b'fixture')
    out = tmp_path / 'extended.json'
    module.run_arm('research', 'http://127.0.0.1:1', model, out, '/model', 'extended')
    report = json.loads(out.read_text(encoding='utf-8'))
    assert [call[2] for call in calls] == ['test[20:25]', 'test[10:15]']
    assert report['max_tokens'] == 192
    assert {row['group'] for row in report['rows']} == {'harmful', 'benign'}
