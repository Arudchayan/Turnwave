"""The field-test console stays self-contained and parseable."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from test_ui import TEST_PAGE
from ui import PAGE


def test_console_has_the_protocol_and_no_external_assets():
    for step in ('normal-1', 'backchannel', 'stop', 'pause', 'long',
                 'memory', 'echo', 'normal-2'):
        assert step in TEST_PAGE
    for control in ('id="arm"', 'id="step-list"', 'id="export-json"',
                    'id="export-csv"', 'id="copy-summary"', '/api/state',
                    '/api/diagnostics'):
        assert control in TEST_PAGE
    assert '<script src' not in TEST_PAGE
    assert 'https://' not in TEST_PAGE
    assert 'http://' not in TEST_PAGE


def test_main_console_links_to_the_test_bench():
    assert 'href="/testing"' in PAGE


def test_handler_serves_the_test_bench():
    from app import Handler
    handler = object.__new__(Handler)
    handler.path = '/testing'
    handler._local = lambda: True
    sent = []
    handler._send = lambda body, status=200, ctype='application/json': sent.append(
        (body, status, ctype))
    handler.do_GET()
    assert sent == [(TEST_PAGE, 200, 'text/html')]


def test_console_script_parses():
    if not shutil.which('node'):
        pytest.skip('Node required for script parse check')
    script = TEST_PAGE.split('<script>', 1)[1].split('</script>', 1)[0]
    result = subprocess.run(
        ['node', '-e', 'new Function(require("fs").readFileSync(0,"utf8"))'],
        input=script, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_comparison_runs_are_saved_separately():
    if not shutil.which('node'):
        pytest.skip('Node required for browser state check')
    script = TEST_PAGE.split('<script>', 1)[1].split('</script>', 1)[0]
    storage_code = script[script.index('const KEY ='):script.index('function showBanner')]
    harness = '''
const assert = require('assert');
const ARMS = {baseline: {}, A: {}};
const values = new Map();
const localStorage = {getItem: k => values.get(k) || null,
  setItem: (k, v) => values.set(k, v)};
const document = {getElementById: () => ({textContent: ''})};
''' + storage_code + '''
run.steps.pause = {outcome: 'pass'}; save();
arm = 'A'; run = load(arm);
assert.strictEqual(run.steps.pause, undefined);
run.steps.pause = {outcome: 'fail'}; save();
assert.strictEqual(load('baseline').steps.pause.outcome, 'pass');
assert.strictEqual(load('A').steps.pause.outcome, 'fail');
'''
    result = subprocess.run(['node', '-e', harness], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
