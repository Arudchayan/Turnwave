"""Private native model cancellation/recovery on port 8081 (CPU by default).

No physical audio device is opened. An identical test-only failed-write hook
in both builds makes the missed-stop path deterministic. Ordinary cancellation
is tested separately after that one-shot hook has been consumed.
GPU exercise calls are made only by gpu_abort_maintenance while the app and
main runner are stopped and both backend endpoints are reserved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import time
import wave

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from backend_session import backend_session
from client import ServerClient,Turn,TTS_VOICE
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint
from synth.native_abort_probe import wsl_path,wsl_home

LONG_TEXT='One, two, three, four, five, six, seven, eight, nine, ten. This synthetic passage is deliberately long so generation is still running when the test transport fails. Continue reading the remaining words clearly and slowly.'


def recovery_timing(rows,allowed_excess_s=1.):
    """Valid audio alone must not hide generation still blocking the next turn."""
    report={}
    for version in sorted({row['version'] for row in rows}):
        group=[row for row in rows if row['version']==version]
        control=next((row for row in group if row.get('kind',row['case'])=='healthy_control'),None)
        reference=control['first_audio_s'] if control and control['passed'] else None
        cases=[]
        for row in group:
            if row.get('kind',row['case']) not in ['recovery','natural_recovery']:continue
            excess=row['first_audio_s']-reference if row['first_audio_s'] is not None and reference is not None else None
            cases.append(dict(case=row['case'],first_audio_s=row['first_audio_s'],excess_s=excess,
                passed=bool(row['passed'] and excess is not None and excess<=allowed_excess_s)))
        report[version]=dict(healthy_first_audio_s=reference,cases=cases,passed=bool(cases) and all(c['passed'] for c in cases))
    return dict(allowed_cpu_excess_s=allowed_excess_s,by_version=report)


def gpu_inventory():
    result=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader'],capture_output=True,text=True)
    return dict(exit_code=result.returncode,output=result.stdout.strip(),error=result.stderr.strip())


def port_free():
    with socket.socket() as connection:
        connection.settimeout(.3);return connection.connect_ex(('127.0.0.1',8081))!=0


def stop_owned(process,folder,binary):
    record=folder/'server-process.json'
    launcher=folder/'launcher-process.json';deadline=time.monotonic()+3
    while not record.exists() and not launcher.exists() and process.poll() is None and time.monotonic()<deadline:time.sleep(.02)
    if record.exists() or launcher.exists():
        pid=json.loads((record if record.exists() else launcher).read_text())['pid']
        launcher_hex=json.loads(launcher.read_text())['command_hex'] if launcher.exists() else ''
        code='''import os,sys,signal,pathlib
pid=int(sys.argv[1]);expected=sys.argv[2].encode()+b'\\0'
path=pathlib.Path('/proc')/str(pid)/'cmdline'
if path.exists():
 command=path.read_bytes()
 if not command.startswith(expected) and command.hex()!=sys.argv[4]:raise RuntimeError('Private server PID has unexpected command')
 os.kill(pid,int(sys.argv[3]))
'''
        result=subprocess.run(['wsl.exe','-e','python3','-c',code,str(pid),binary,'2',launcher_hex],capture_output=True)
        (folder/'stop.log').write_bytes(result.stdout+result.stderr);result.check_returncode()
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            result=subprocess.run(['wsl.exe','-e','python3','-c',code,str(pid),binary,'9',launcher_hex],capture_output=True)
            (folder/'stop-forced.log').write_bytes(result.stdout+result.stderr);result.check_returncode();process.wait(timeout=10)
    elif process.poll() is None:
        # The launcher is still hashing model files and has not exec'd a server.
        # It has no listening endpoint or model context yet.
        raise RuntimeError('Private launcher has no PID record; preserve handle for investigation')


def save_audio(path,chunks):
    audio=np.concatenate(chunks) if chunks else np.zeros(0,np.float32)
    if not np.isfinite(audio).all():raise RuntimeError('Non-finite native output')
    if len(audio):
        with wave.open(str(path),'wb') as stream:
            stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(24000)
            stream.writeframes(np.rint(np.clip(audio,-1,1)*32767).astype('<i2').tobytes())
    return float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.


def exercise(folder,private,version,gpu=False):
    folder.mkdir(exist_ok=False)
    # WSL's Windows localhost relay can briefly outlive the owned Linux
    # process. Wait for that endpoint to clear; never stop an unknown owner.
    deadline=time.monotonic()+10
    while not port_free():
        if time.monotonic()>deadline:raise RuntimeError('8081 remains occupied; do not disturb its owner')
        time.sleep(.2)
    command=['wsl.exe','-e','python3',wsl_path(ROOT/'src/synth/serve_abort_variant_linux.py'),private,version,wsl_path(folder)]
    if gpu:command.append('--gpu')
    rows=[]
    with (folder/'server.log').open('wb') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            client=ServerClient('http://127.0.0.1:8081',timeout=120);deadline=time.monotonic()+90
            while not client.health():
                if process.poll() is not None:raise RuntimeError('Private server exited during startup')
                if time.monotonic()>deadline:raise TimeoutError('Private server did not become ready')
                time.sleep(.2)
            print(json.dumps(dict(version=version,phase='ready',gpu=gpu_inventory())),flush=True)
            cases=[(c,c) for c in ['injected_failure','recovery','healthy_control','natural_cancel','natural_recovery']]
            if gpu or version in ['clean','lifecycle']:
                cases=[('healthy_control','healthy_control')]+[(f'{kind}_{cycle}',kind) for cycle in range(3) for kind in ['natural_cancel','natural_recovery']]
                cases += [('continuation_control','continuation_control'),('post_continuation_control','post_continuation_control')]
            if gpu:cases=[('warmup_0','warmup'),('warmup_1','warmup')]+cases
            for label,case in cases:
                turn=Turn();chunks=[];cancel_started=None
                text=LONG_TEXT if case in ['injected_failure','natural_cancel'] else 'Ready.'
                maximum=(512 if gpu else 64) if case in ['injected_failure','natural_cancel'] else 48
                payload=client._payload(TTS_VOICE,text,case!='continuation_control',maximum)
                (folder/(label+'-request.json')).write_text(json.dumps(payload,indent=2))
                def audio(chunk):
                    nonlocal cancel_started
                    chunks.append(chunk.copy())
                    audible=not gpu and version not in ['clean','lifecycle'] or turn.audio_samples>=4800 and float(np.sqrt(np.mean(chunk**2)))>=.005
                    if case=='natural_cancel' and cancel_started is None and audible:
                        cancel_started=time.perf_counter();turn.abort()
                print(json.dumps(dict(version=version,phase='request',case=case)),flush=True)
                started=time.perf_counter();client._stream(payload,turn,audio);elapsed=time.perf_counter()-started
                cancel_elapsed=time.perf_counter()-cancel_started if cancel_started is not None else None
                rms=save_audio(folder/(label+'.wav'),chunks)
                expected_error=case=='injected_failure'
                passed=bool(turn.error) if expected_error else bool(turn.cancel.is_set() and chunks) if case=='natural_cancel' else bool(not turn.error and chunks and rms>1e-5)
                row=dict(version=version,case=label,kind=case,elapsed_s=elapsed,first_audio_s=turn.ttft,headers_s=turn.headers_s,
                    first_event_s=turn.first_event,cancel_to_client_done_s=cancel_elapsed,audio_samples=turn.audio_samples,
                    audio_rms=rms,error=turn.error,finish_reason=turn.finish_reason,text=turn.text,passed=passed)
                if gpu:
                    telemetry=subprocess.run(['nvidia-smi','--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate','--format=csv,noheader'],capture_output=True,text=True)
                    row['gpu_telemetry']=dict(exit_code=telemetry.returncode,output=telemetry.stdout.strip(),error=telemetry.stderr.strip())
                rows.append(row)
                with (folder/'results.jsonl').open('a') as results:results.write(json.dumps(row)+'\n');results.flush()
                print(json.dumps(row),flush=True)
            return rows
        finally:
            stop_owned(process,folder,private+'/'+version+'/llama-liquid-audio-server')
            (folder/'server-exit.json').write_text(json.dumps(dict(exit_code=process.poll()),indent=2))


def run(folder,private,baseline_results=None,clean=False,lifecycle=False):
    folder.mkdir(parents=True,exist_ok=False);capture_files(folder/'source')
    (folder/'protocol.json').write_text(json.dumps(dict(private_path=private,endpoint='http://127.0.0.1:8081',
        device='CPU only, 2 generation/batch threads; CUDA_VISIBLE_DEVICES empty',fault='None; clean candidate, three audible socket cancellations' if clean or lifecycle else 'First emitted stream chunk is rejected once in both builds',
        live_backend_used=False,physical_audio=False,gpu_before=gpu_inventory()),indent=2))
    with backend_session('http://127.0.0.1:8081'):
        rows=[]
        if clean or lifecycle:
            version='lifecycle' if lifecycle else 'clean';rows+=exercise(folder/version,private,version)
        elif baseline_results:
            prior=json.loads((baseline_results.parent/'protocol.json').read_text())
            if prior['private_path']!=private:raise RuntimeError('Reused baseline uses a different private build')
            data=(baseline_results/'results.jsonl').read_bytes();rows=[json.loads(line) for line in data.decode().splitlines()]
            if [r['case'] for r in rows]!=['injected_failure','recovery','healthy_control','natural_cancel','natural_recovery'] or not all(r['passed'] and r['version']=='before' for r in rows):
                raise RuntimeError('Cannot reuse incomplete baseline controls')
            shutil.copytree(baseline_results,folder/'before')
            (folder/'baseline-reuse.json').write_text(json.dumps(dict(source=str(baseline_results),results_sha256=hashlib.sha256(data).hexdigest(),note='Completed baseline sequence retained from the prior run; its next-phase failure was only the transient localhost-relay occupancy check.'),indent=2))
        else:rows+=exercise(folder/'before',private,'before')
        if not clean and not lifecycle:rows+=exercise(folder/'after',private,'after')
    timings=recovery_timing(rows);candidate='lifecycle' if lifecycle else 'clean' if clean else 'after'
    report=dict(cases=rows,all_request_controls_passed=all(r['passed'] for r in rows),gpu_after=gpu_inventory(),deployed=False,
        recovery_timing=timings,candidate_recovery_gate_passed=timings['by_version'][candidate]['passed'],
        scope=('Clean native CPU candidate without a transport hook; three cancellations after audible output and fresh-context recovery.' if clean or lifecycle else
            'Actual native CPU TTS generation, deterministic failed-write injection and ordinary socket cancellation. One process per variant, one sequence each.')+
            ' Not GPU latency, semantic answer quality or physical-room validation.')
    (folder/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(all_request_controls_passed=report['all_request_controls_passed'],candidate_recovery_gate_passed=report['candidate_recovery_gate_passed'])),flush=True)
    return report['all_request_controls_passed'] and report['candidate_recovery_gate_passed']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);parser.add_argument('--private',default=None);parser.add_argument('--baseline-results',type=Path);parser.add_argument('--clean',action='store_true');parser.add_argument('--lifecycle',action='store_true');args=parser.parse_args()
    if args.folder.exists():parser.error('Use a new experiment folder')
    if args.clean and args.lifecycle:parser.error('Choose one clean candidate')
    if (args.clean or args.lifecycle) and args.baseline_results:parser.error('Clean repetition run does not reuse the injected baseline')
    try:passed=run(args.folder,args.private or wsl_home()+'/localvoice-native-abort-v1',args.baseline_results,args.clean,args.lifecycle)
    except Exception:
        if args.folder.exists():
            import traceback
            (args.folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if args.folder.exists():print('Verified backup:',checkpoint(args.folder),flush=True)
    raise SystemExit(0 if passed else 1)
