"""Compile the actual native abort callback with controlled transport doubles.

This is a source-level regression probe, not a loaded-model integration test.
It neither modifies the native source checkout nor contacts a model server.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint

PREFIX=r'''
#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#define LOG_INF(...) ((void)0)
struct OutputBuffer { std::atomic<bool> aborted{false}; };
struct Request { std::function<bool()> is_connection_closed; };
struct Runner { bool stopped=false; void stop() { stopped=true; } };
bool exercise(const char *name, bool pre_aborted, bool closed, bool running, bool expected) {
    Request req{[closed] { return closed; }};
    auto output=std::make_shared<OutputBuffer>(); output->aborted=pre_aborted;
    Runner runner; std::atomic<bool> is_server_running{running};
'''
SUFFIX=r'''
    check_abort();
    const bool passed=runner.stopped==expected;
    std::cout << "{\"case\":\"" << name << "\",\"stopped\":" << (runner.stopped?"true":"false")
              << ",\"passed\":" << (passed?"true":"false") << "}\n";
    return passed;
}
int main() {
    int failed=0;
    failed+=!exercise("connected",false,false,true,false);
    failed+=!exercise("closed_connection",false,true,true,true);
    failed+=!exercise("sink_already_marked_aborted",true,false,true,true);
    failed+=!exercise("server_stopping",false,false,false,true);
    return failed?1:0;
}
'''


def wsl_path(path):
    path=path.resolve();return '/mnt/'+path.drive[0].lower()+'/'+path.as_posix()[3:]


def wsl_home():
    result=subprocess.run(['wsl.exe','-e','printenv','HOME'],capture_output=True,text=True,check=True)
    home=result.stdout.strip()
    if not home.startswith('/') or '\n' in home or '\r' in home:
        raise ValueError('Invalid WSL home directory')
    return home


def run(folder):
    folder.mkdir(parents=True,exist_ok=False);capture_files(folder/'source')
    result=subprocess.run(['wsl.exe','-e','cat',wsl_home()+'/liquid-build/tools/liquid-audio/server.cpp'],capture_output=True,check=True)
    source=result.stdout.decode();(folder/'source-read-stderr.log').write_bytes(result.stderr)
    (folder/'server-before.cpp').write_bytes(result.stdout)
    marker='auto check_abort = [&req, output, &runner, &is_server_running]()'
    if source.count(marker)!=1:raise RuntimeError('Native callback signature changed')
    begin=source.index(marker);opening=source.index('{',begin);depth=1;end=opening+1
    while depth:
        if source[end]=='{':depth+=1
        if source[end]=='}':depth-=1
        end+=1
    if source[end]!=';':raise RuntimeError('Unexpected callback ending')
    original=source[begin:end+1]
    patched,count=re.subn(r'(if \(output->aborted\.load\(\)\) \{\n)(\s*)(return;)',r'\1\2runner.stop();\n\2\3',original,count=1)
    if count!=1:raise RuntimeError('Expected early-aborted return was not found')
    after=source[:begin]+patched+source[end+1:];(folder/'server-after.cpp').write_bytes(after.encode())
    (folder/'candidate.patch').write_bytes(''.join(difflib.unified_diff(source.splitlines(True),after.splitlines(True),fromfile='a/tools/liquid-audio/server.cpp',tofile='b/tools/liquid-audio/server.cpp')).encode())
    versions={}
    for label,callback in [('before',original),('after',patched)]:
        path=folder/(label+'.cpp');path.write_text(PREFIX+callback+SUFFIX)
        binary=folder/(label+'.bin')
        compiled=subprocess.run(['wsl.exe','-e','g++','-std=c++17','-O0','-Wall','-Wextra',wsl_path(path),'-o',wsl_path(binary)],capture_output=True)
        (folder/(label+'-build.log')).write_bytes(compiled.stdout+compiled.stderr)
        compiled.check_returncode()
        executed=subprocess.run(['wsl.exe','-e',wsl_path(binary)],capture_output=True)
        (folder/(label+'-run.log')).write_bytes(executed.stdout+executed.stderr)
        rows=[json.loads(line) for line in executed.stdout.decode().splitlines()]
        versions[label]=dict(exit_code=executed.returncode,cases=rows)
    before=versions['before'];after_result=versions['after']
    if before['exit_code']!=1 or [r['case'] for r in before['cases'] if not r['passed']]!=['sink_already_marked_aborted']:
        raise RuntimeError('The hypothesized failure was not reproduced precisely')
    if after_result['exit_code']!=0 or not all(r['passed'] for r in after_result['cases']):raise RuntimeError('Candidate callback failed a control')
    report=dict(versions=versions,native_source_sha256=hashlib.sha256(result.stdout).hexdigest(),deployed=False,
        scope='Actual callback extracted from current native source; controlled request/output/runner doubles. Proves stop notification logic only. Loaded-model abort/recovery timing and running-binary equivalence remain unverified.')
    (folder/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(versions),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);args=parser.parse_args()
    if args.folder.exists():parser.error('Use a new experiment folder')
    try:run(args.folder)
    except Exception:
        if args.folder.exists():
            import traceback
            (args.folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if args.folder.exists():print('Verified backup:',checkpoint(args.folder),flush=True)
