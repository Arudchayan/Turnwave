import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import self_improve


def repo_fixture(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    (repo / '.gitignore').write_text('outputs/\n', encoding='utf-8')
    (repo / 'src').mkdir()
    (repo / 'tests').mkdir()
    (repo / 'src' / 'value.py').write_text('VALUE = 1\n', encoding='utf-8')
    (repo / 'tests' / 'test_value.py').write_text('def test_value(): assert True\n', encoding='utf-8')
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.org', 'commit', '-qm', 'base'], check=True)
    return repo


def test_development_run_preserves_worktree_diff_and_test_log(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCALVOICE_FLOOR_MODE', 'reversible')
    monkeypatch.setenv('OPENCODE_GO_API_KEY', 'secret')
    repo = repo_fixture(tmp_path)
    phases = []
    def runner(argv, **kwargs):
        if argv[0] == 'git':
            return subprocess.run(argv, **kwargs)
        if argv[0] == 'fake-opencode':
            phases.append('edit')
            tree = Path(kwargs['cwd'])
            assert json.loads((tree / 'opencode.json').read_text()) == self_improve.POLICY
            (tree / 'src' / 'value.py').write_text('VALUE = 2\n', encoding='utf-8')
            (tree / 'src' / 'new_tool.py').write_text('VALUE = 3\n', encoding='utf-8')
            kwargs['stdout'].write('{"type":"text"}\n')
        else:
            phases.append('test')
            assert 'LOCALVOICE_FLOOR_MODE' not in kwargs['env']
            assert 'OPENCODE_GO_API_KEY' not in kwargs['env']
            kwargs['stdout'].write('tests passed\n')
        return SimpleNamespace(returncode=0)
    worker = self_improve.SelfImprover(repo, runner=runner)
    worker._run('test-run', 'Improve the value', self_improve.MODELS[0], 'fake-opencode', {'status': 'error'})
    result = worker.state()
    assert result['status'] == 'ready'
    assert phases == ['edit', 'test']
    assert (repo / 'src' / 'value.py').read_text() == 'VALUE = 1\n'
    run_dir = Path(result['run_dir'])
    assert 'VALUE = 2' in (run_dir / 'review.diff').read_text(encoding='utf-8')
    assert 'new_tool.py' in (run_dir / 'review.diff').read_text(encoding='utf-8')
    assert (run_dir / 'pytest-1.log').read_text() == 'tests passed\n'
    assert (run_dir / 'opencode-1.jsonl').exists()
    assert json.loads((run_dir / 'result.json').read_text())['status'] == 'ready'
    assert not (Path(result['worktree']) / 'opencode.json').exists()


def test_disallowed_changes_are_preserved_without_running_tests(tmp_path):
    repo = repo_fixture(tmp_path)
    def runner(argv, **kwargs):
        if argv[0] == 'git':
            return subprocess.run(argv, **kwargs)
        assert argv[0] == 'fake-opencode', 'Tests must not run for rejected changes'
        (Path(kwargs['cwd']) / 'outside.txt').write_text('preserved', encoding='utf-8')
        return SimpleNamespace(returncode=0)
    worker = self_improve.SelfImprover(repo, runner=runner)
    worker._run('rejected', 'Change a file', self_improve.MODELS[0], 'fake-opencode', {})
    assert worker.state()['status'] == 'error'
    assert 'permitted paths' in worker.state()['message']
    assert (Path(worker.state()['worktree']) / 'outside.txt').exists()


def test_development_run_validates_user_input(tmp_path, monkeypatch):
    monkeypatch.setattr(self_improve, 'find_opencode', lambda: 'fake-opencode')
    worker = self_improve.SelfImprover(tmp_path)
    with pytest.raises(ValueError):
        worker.start('', self_improve.MODELS[0])
    with pytest.raises(ValueError):
        worker.start('Improve tests', 'unknown/model')


def test_development_http_endpoint_forwards_only_summary():
    import json
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    from app import Handler
    class Assistant:
        def state(self):
            return {'last_trace': {'status': 'error', 'checks': {'reply_non_empty': False}},
                    'transcript': ['private conversation']}
    class Improver:
        def state(self): return {'status': 'idle'}
        def start(self, task, model, trace):
            assert task == 'Fix the failure'
            assert model == self_improve.MODELS[0]
            assert trace == {'status': 'error', 'checks': {'reply_non_empty': False}}
            return {'status': 'preparing'}
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.assistant, server.improver = Assistant(), Improver()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f'http://127.0.0.1:{server.server_port}/api/improve'
        request = urllib.request.Request(url, data=json.dumps({'task': 'Fix the failure', 'model': self_improve.MODELS[0]}).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=3) as response:
            assert json.load(response)['status'] == 'preparing'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_failing_tests_trigger_one_repair_with_separate_logs(tmp_path):
    repo = repo_fixture(tmp_path)
    phases = []
    def runner(argv, **kwargs):
        if argv[0] == 'git':
            return subprocess.run(argv, **kwargs)
        tree = Path(kwargs['cwd'])
        if argv[0] == 'fake-opencode':
            phases.append('edit')
            if phases.count('edit') == 1:
                (tree / 'src' / 'value.py').write_text('VALUE = 2\n', encoding='utf-8')
            else:
                assert 'example failure' in argv[-1]
                (tree / 'src' / 'value.py').write_text('VALUE = 3\n', encoding='utf-8')
            return SimpleNamespace(returncode=0)
        phases.append('test')
        if phases.count('test') == 1:
            kwargs['stdout'].write('example failure')
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)
    worker = self_improve.SelfImprover(repo, runner=runner)
    worker._run('retry', 'Improve the value', self_improve.MODELS[0], 'fake-opencode', {})
    assert phases == ['edit', 'test', 'edit', 'test']
    assert worker.state()['status'] == 'ready'
    run_dir = Path(worker.state()['run_dir'])
    assert (run_dir / 'pytest-1.log').read_text() == 'example failure'
    assert (run_dir / 'pytest-2.log').exists()
    assert 'VALUE = 3' in (run_dir / 'review.diff').read_text(encoding='utf-8')
