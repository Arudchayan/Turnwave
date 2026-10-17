"""Bounded GPU cancellation comparison, restoring the idle muted app in finally.

Never opens an audio output or enables microphone capture. Requires the exact
expected idle app and native process; saves a verified rollback before stopping.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from urllib.request import urlopen
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from backend_session import backend_session
from client import ServerClient,Turn,TTS_VOICE
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint
from synth.native_abort_integration import exercise,recovery_timing,gpu_inventory,save_audio,stop_owned
from synth.native_abort_probe import wsl_path,wsl_home


def linux(code,*args):
    result=subprocess.run(['wsl.exe','-e','python3','-c',code,*map(str,args)],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr+result.stdout)
    return result.stdout


def state():
    with urlopen('http://127.0.0.1:8000/api/state',timeout=3) as response:return json.load(response)


def idle(value):
    return (value['status']=='listening' and value['mic_on'] is False and not value['transcript']
        and value['transcript_revision']==0 and not value['assistant_partial'] and not value['error'])


def wait_health(client,process,timeout=90):
    deadline=time.monotonic()+timeout
    while not client.health():
        if process.poll() is not None:raise RuntimeError('Restored server exited')
        if time.monotonic()>deadline:raise TimeoutError('Restored backend health timeout')
        time.sleep(.2)


def launch(record,folder):
    folder.mkdir(exist_ok=False)
    (folder/'launch.json').write_text(json.dumps(record,indent=2))
    code='''import pathlib,json,os,sys,hashlib
r=json.loads(pathlib.Path(sys.argv[1]).read_text());binary=pathlib.Path(r['args'][0])
if hashlib.sha256(binary.read_bytes()).hexdigest()!=r['binary_sha256']:raise RuntimeError('Runner binary changed')
out=pathlib.Path(sys.argv[2]);(out/'server-process.json').write_text(json.dumps(dict(pid=os.getpid(),command=r['args'])))
os.environ.update(r['env']);os.execv(r['args'][0],r['args'])
'''
    log=(folder/'server.log').open('wb')
    process=subprocess.Popen(['wsl.exe','-e','python3','-c',code,wsl_path(folder/'launch.json'),wsl_path(folder)],stdout=log,stderr=subprocess.STDOUT)
    (folder/'proxy.json').write_text(json.dumps(dict(windows_pid=process.pid)))
    return process


def live_smoke(folder):
    client=ServerClient('http://127.0.0.1:8080',timeout=30);rows=[]
    for case in ['healthy_control','natural_cancel','natural_recovery']:
        turn=Turn();chunks=[];text=('One, two, three, four, five, six, seven, eight, nine, ten. Keep reading clearly.' if case=='natural_cancel' else 'Ready.')
        payload=client._payload(TTS_VOICE,text,True,512 if case=='natural_cancel' else 48)
        (folder/(case+'-request.json')).write_text(json.dumps(payload,indent=2))
        def audio(chunk):
            chunks.append(chunk.copy())
            if case=='natural_cancel' and turn.audio_samples>=4800 and float(np.sqrt(np.mean(chunk**2)))>=.005:turn.abort()
        client._stream(payload,turn,audio)
        rms=save_audio(folder/(case+'.wav'),chunks)
        passed=bool(turn.cancel.is_set() and chunks) if case=='natural_cancel' else bool(not turn.error and chunks and rms>.00001)
        rows.append(dict(version='deployed',case=case,first_audio_s=turn.ttft,passed=passed,error=turn.error,audio_samples=turn.audio_samples,audio_rms=rms))
        (folder/'smoke-results.json').write_text(json.dumps(rows,indent=2))
    timing=recovery_timing(rows,.5);passed=all(r['passed'] for r in rows) and timing['by_version']['deployed']['passed']
    (folder/'smoke-summary.json').write_text(json.dumps(dict(passed=passed,timing=timing),indent=2))
    if not passed:raise RuntimeError('Deployment cancellation/recovery smoke failed')


def run(folder,ui_pid,native_pid,reverse=False,activate_from=None):
    folder.mkdir(parents=True,exist_ok=False);capture_files(folder/'source')
    before=state()
    if not idle(before):raise RuntimeError('App is no longer muted and idle; leave it alone')
    (folder/'before-state.json').write_text(json.dumps(before,indent=2))
    shutil.copy2(ROOT/'outputs/logs/app.jsonl',folder/'before-app.jsonl')
    snapshot='''import pathlib,sys,json,hashlib,shutil
pid=sys.argv[1];out=pathlib.Path(sys.argv[2]);proc=pathlib.Path('/proc')/pid
binary=str(pathlib.Path.home()/'liquid-build/build/bin/llama-liquid-audio-server')
args=proc.joinpath('cmdline').read_bytes().rstrip(b'\\0').decode().split('\\0')
if args[0]!=binary:
 sys.path.insert(0,sys.argv[3]);from runner_selection import selected_runner
 selected=selected_runner(pathlib.Path.home()/'.config/localvoice/runner.json')
 if not selected or args[0]!=str(pathlib.Path(selected[0])/'llama-liquid-audio-server'):raise RuntimeError('Unexpected native owner')
if args[-2:]!=['--port','8080']:raise RuntimeError('Unexpected native endpoint')
env=dict(item.split('=',1) for item in proc.joinpath('environ').read_bytes().decode().split('\\0') if '=' in item)
saved={k:v for k,v in env.items() if k in ['LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','PATH','HOME']}
shutil.copy2(proc/'exe',out/'rollback-native-server')
record=dict(pid=int(pid),args=args,env=saved,binary_sha256=hashlib.sha256((out/'rollback-native-server').read_bytes()).hexdigest())
(out/'native-before.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
'''
    original=json.loads(linux(snapshot,native_pid,wsl_path(folder),wsl_path(ROOT/'src')))
    candidate=None
    if activate_from:
        prepared=subprocess.run(['wsl.exe','-e','python3',wsl_path(ROOT/'src/synth/activate_abort_linux.py'),'prepare',wsl_path(folder),wsl_path(activate_from)],capture_output=True,text=True)
        (folder/'prepare.log').write_text(prepared.stdout+prepared.stderr);prepared.check_returncode()
        selection=json.loads((folder/'candidate-selection.json').read_text())
        candidate=dict(original);candidate['args']=[selection['binary'],*original['args'][1:]]
        candidate['binary_sha256']=selection['files'][selection['binary']]
        candidate['env']=dict(original['env']);candidate['env']['LD_LIBRARY_PATH']=selection['library_directory']+':'+candidate['env'].get('LD_LIBRARY_PATH','')
    (folder/'h3-user-confirmation.json').write_text(json.dumps(dict(
        statement='User confirmed remaining completely quiet during H3 autumn-leaves cutoff.',
        classification='false interruption in human test',new_recording=False),indent=2))
    print('Rollback checkpoint:',checkpoint(folder),flush=True)
    stopped_ui=False;stopped_native=False
    try:
        if not idle(state()):raise RuntimeError('App changed before maintenance')
        ps=f'''$p=Get-CimInstance Win32_Process -Filter "ProcessId={ui_pid}"
if (!$p -or $p.ExecutablePath -ne 'C:\\Python312\\python.exe' -or $p.CommandLine -notin @('\"C:\\Python312\\python.exe\" -u src/app.py','C:\\Python312\\python.exe -u src/app.py')) {{ throw 'Unexpected app process' }}
$listener=Get-NetTCPConnection -State Listen -LocalPort 8000
if ($listener.OwningProcess -ne {ui_pid}) {{ throw 'Unexpected UI listener' }}
Stop-Process -Id {ui_pid} -ErrorAction Stop
'''
        stopping=subprocess.run(['powershell','-NoProfile','-Command',ps],capture_output=True)
        (folder/'ui-stop.log').write_bytes(stopping.stdout+stopping.stderr);stopping.check_returncode()
        stopped_ui=True
        with backend_session(test=True):
            linux('''import pathlib,sys,os,signal,json
pid=int(sys.argv[1]);expected=json.loads(pathlib.Path(sys.argv[2]).read_text())['args']
actual=(pathlib.Path('/proc')/str(pid)/'cmdline').read_bytes().rstrip(b'\\0').decode().split('\\0')
if actual!=expected:raise RuntimeError('Native PID changed; refusing stop')
os.kill(pid,signal.SIGINT)
''',native_pid,wsl_path(folder/'native-before.json'))
            stopped_native=True
            ready_candidate=None
            try:
                deadline=time.monotonic()+30
                while linux("import pathlib,sys;print(int((pathlib.Path('/proc')/sys.argv[1]/'cmdline').exists()))",native_pid).strip()=='1':
                    if time.monotonic()>deadline:raise TimeoutError('Live runner did not stop; no competing GPU test launched')
                    time.sleep(.2)
                rows=[]
                with backend_session('http://127.0.0.1:8081'):
                    versions=['lifecycle'] if candidate else (['lifecycle','before'] if reverse else ['before','lifecycle'])
                    private=selection['runtime_package'] if candidate else wsl_home()+'/localvoice-native-abort-v1'
                    for variant in versions:rows+=exercise(folder/variant,private,variant,gpu=True)
                timings=recovery_timing(rows,.5)
                report=dict(cases=rows,recovery_timing=timings,
                    all_request_controls_passed=all(r['passed'] for r in rows),
                    candidate_recovery_gate_passed=timings['by_version']['lifecycle']['passed'],
                    scope='Staged GPU runtime validation' if candidate else 'Sequential actual GPU comparison',
                    deployed=False,gpu_after=gpu_inventory())
                (folder/'summary.json').write_text(json.dumps(report,indent=2))
                if candidate and not (report['all_request_controls_passed'] and report['candidate_recovery_gate_passed']):
                    candidate=None;raise RuntimeError('Staged runtime failed recovery validation')
                ready_candidate=candidate
            finally:
                candidate=ready_candidate
                # Do not launch a second model if an owned process failed to exit.
                active=linux("import pathlib,sys; p=pathlib.Path('/proc')/sys.argv[1]/'cmdline'; print(int(p.exists() and bool(p.read_bytes())))",native_pid).strip()=='1'
                if active:raise RuntimeError('Original native process is still alive; restoration requires inspection')
                private_active=linux('''import pathlib,json
live=[]
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:command=p.read_bytes().split(b'\\0')[0]
 except FileNotFoundError:continue
 if command.endswith(b'/llama-liquid-audio-server'):live.append(p.parent.name)
print(json.dumps(live))
''').strip()
                if private_active!='[]':raise RuntimeError('Private server remains alive; refuse overlapping GPU restoration')
                # Neither the original executable nor its libraries are modified.
                chosen=candidate or original;live_folder=folder/('candidate-live' if candidate else 'restored-native')
                process=launch(chosen,live_folder)
                activated=False
                try:
                    wait_health(ServerClient('http://127.0.0.1:8080'),process)
                    if candidate:
                        live_smoke(live_folder)
                        action=['wsl.exe','-e','python3',wsl_path(ROOT/'src/synth/activate_abort_linux.py'),'activate',wsl_path(folder)]
                        result=subprocess.run(action,capture_output=True,text=True)
                        (folder/'activation.log').write_text(result.stdout+result.stderr);result.check_returncode();activated=True
                        report['deployed']=True;report['active_selection']=selection
                        (folder/'summary.json').write_text(json.dumps(report,indent=2))
                except Exception:
                    (folder/'deployment-failure.txt').write_text(traceback.format_exc())
                    stop_owned(process,live_folder,chosen['args'][0])
                    if activated:
                        subprocess.run(['wsl.exe','-e','python3',wsl_path(ROOT/'src/synth/activate_abort_linux.py'),'rollback',wsl_path(folder)],check=True)
                    if candidate:
                        process=launch(original,folder/'fallback-native')
                        wait_health(ServerClient('http://127.0.0.1:8080'),process)
                        stopped_native=False
                    raise
                stopped_native=False
    finally:
        if stopped_ui and not stopped_native:
            env=os.environ.copy();env['LOCALVOICE_START_MUTED']='1'
            env.pop('LOCALVOICE_RECORD_RAW',None)
            log=(folder/'restored-ui.log').open('wb')
            app=subprocess.Popen([sys.executable,'-u','src/app.py'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            (folder/'restored-ui-process.json').write_text(json.dumps(dict(pid=app.pid)))
            deadline=time.monotonic()+45
            while True:
                try:
                    after=state()
                    if idle(after) and after['server_ok']:
                        (folder/'after-state.json').write_text(json.dumps(after,indent=2));break
                except OSError:pass
                if app.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Muted app restore did not become healthy')
                time.sleep(.2)
        elif stopped_native:
            (folder/'RESTORATION_REQUIRED.txt').write_text('Native restoration failed; app deliberately left stopped. See native-before.json and rollback-native-server.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--ui-pid',type=int,required=True);p.add_argument('--native-pid',type=int,required=True);p.add_argument('--reverse',action='store_true');p.add_argument('--activate-from',type=Path);a=p.parse_args()
    try:run(a.folder.resolve(),a.ui_pid,a.native_pid,a.reverse,a.activate_from.resolve() if a.activate_from else None)
    except Exception:
        if a.folder.exists():(a.folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if a.folder.exists():print('Final verified checkpoint:',checkpoint(a.folder),flush=True)
