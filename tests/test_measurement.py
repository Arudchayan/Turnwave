"""Regression checks for trustworthy measurement and archived evidence."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess


def load(name):
    path=Path(__file__).resolve().parents[1]/'src/synth'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_old_report_is_not_archived_as_current_success(tmp_path,monkeypatch):
    loop=load('improve_loop');monkeypatch.setattr(loop,'ROOT',tmp_path)
    (tmp_path/'outputs').mkdir();source=tmp_path/'outputs/quality.json'
    source.write_text('{"passed": true}')
    folder=tmp_path/'archive';folder.mkdir()
    monkeypatch.setattr(loop.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=b'failed before writing',returncode=1))
    result=loop.run_check(folder,0,'quality',[],1,'quality.json')
    assert not result['passed'] and result['artifact_state']=='missing_or_stale'
    assert not (folder/'1-quality.json').exists()
    # Even a buggy child reporting success cannot promote an old artifact.
    monkeypatch.setattr(loop.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=b'',returncode=0))
    assert not loop.run_check(folder,1,'quality',[],1,'quality.json')['passed']


def test_current_failed_report_is_preserved(tmp_path,monkeypatch):
    loop=load('improve_loop');monkeypatch.setattr(loop,'ROOT',tmp_path)
    (tmp_path/'outputs').mkdir();folder=tmp_path/'archive';folder.mkdir()
    def child(*args,**kwargs):
        (tmp_path/'outputs/quality.json').write_text('{"passed": false}')
        return SimpleNamespace(stdout=b'quality failed',returncode=1)
    monkeypatch.setattr(loop.subprocess,'run',child)
    result=loop.run_check(folder,0,'quality',[],1,'quality.json')
    assert not result['passed'] and result['artifact_state']=='updated'
    assert json.loads((folder/'1-quality.json').read_text())=={'passed':False}


def test_timeout_keeps_output_and_fails(tmp_path,monkeypatch):
    loop=load('improve_loop')
    def timeout(*args,**kwargs):raise subprocess.TimeoutExpired('test',1,output=b'partial output')
    monkeypatch.setattr(loop.subprocess,'run',timeout)
    result=loop.run_check(tmp_path,0,'hung',[],1)
    assert result['returncode']==124 and not result['passed']
    assert b'partial output' in (tmp_path/'1-hung.log').read_bytes()


def test_log_summary_does_not_sum_repeated_cumulative_capture_counts(tmp_path):
    analyzer=load('analyze_logs');path=tmp_path/'app.jsonl'
    rows=[dict(ts=1,pid=7,event='health',mic_drops=2,capture_errors=1),
          dict(ts=2,pid=7,event='health',mic_drops=3,capture_errors=1),
          dict(ts=3,pid=7,event='interrupt',reason='barge-in'),
          dict(ts=4,pid=7,event='playback_end',cancelled=True,playback={'underflows':2})]
    path.write_text('\n'.join(json.dumps(r) for r in rows)+'\nbroken\n')
    report=analyzer.summarize([path])
    assert report['capture_counter_maxima']=={'7':{'mic_drops':3,'capture_errors':1}}
    assert report['malformed_lines']==1 and report['playback_underflows']==2
    assert report['interruptions']=={'barge-in':1}
    assert 'false_interruptions' not in report

def test_backend_context_lock_excludes_other_process_and_recovers_after_exit(tmp_path):
    import sys
    import time
    import pytest
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from backend_session import backend_session
    root=Path(__file__).resolve().parents[1]/'src'
    ready=tmp_path/'ready'
    program="""
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from backend_session import backend_session
with backend_session('http://localhost:8080/',directory=sys.argv[2]):
    Path(sys.argv[3]).write_text('ready')
    sys.stdin.read()
"""
    child=subprocess.Popen([sys.executable,'-c',program,str(root),str(tmp_path),str(ready)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    try:
        deadline=time.monotonic()+5
        while not ready.exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(.02)
        assert ready.exists(), 'Child failed to acquire test context'
        with pytest.raises(RuntimeError,match='owns this backend context'):
            with backend_session('http://127.0.0.1:8080',directory=tmp_path):pass
        # Different inference instances have independent contexts.
        with backend_session('http://127.0.0.1:8081',directory=tmp_path):pass
    finally:
        child.terminate();child.communicate(timeout=5)
    # Stale files do not retain ownership after a process dies.
    with backend_session('http://127.0.0.1:8080',directory=tmp_path):pass


def test_backend_tests_refuse_even_a_muted_existing_ui(tmp_path,monkeypatch):
    import sys
    import pytest
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from backend_session import backend_session
    class Connected:
        def close(self):pass
    monkeypatch.setattr('backend_session.socket.create_connection',lambda *a,**k:Connected())
    with pytest.raises(RuntimeError,match='muted conversations still own'):
        with backend_session(test=True,directory=tmp_path):pass
    assert not list(tmp_path.glob('*.lock'))

def test_diagnostics_rotation_retains_every_old_record(tmp_path):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from diagnostics import Diagnostics
    path=tmp_path/'app.jsonl';diag=Diagnostics(path)
    diag._handler.maxBytes=150
    for index in range(30):diag.event('turn_submit',turn_id=index,input_s=1.0)
    diag.close()
    paths=list(tmp_path.glob('app.jsonl*'))
    assert len(paths)>3
    records=[json.loads(line) for file in paths for line in file.read_text(encoding='utf-8').splitlines()]
    assert sorted(row['turn_id'] for row in records)==list(range(30))

def test_log_summary_filters_exact_experiment_and_keeps_pid_type(tmp_path):
    analyzer=load('analyze_logs');path=tmp_path/'app.jsonl'
    rows=[dict(ts=10,pid=1,event='generation_end',first_audio_s=5),
          dict(ts=20,pid=1,event='generation_end',first_audio_s=.2),
          dict(ts=20,pid=2,event='generation_end',first_audio_s=8),
          dict(ts=21,pid=1,event='health',mic_drops=0,capture_errors=0)]
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    report=analyzer.summarize([path],since=15,pid=1)
    assert report['events']==2 and report['first_audio_s']['maximum']==.2
    assert report['pid']==1
    assert analyzer.summarize([path])['pid'] is None


def test_archive_keeps_available_logs_and_records_disappearing_files(tmp_path,monkeypatch):
    archive=load('archive_experiment');monkeypatch.setattr(archive,'ROOT',tmp_path)
    logs=tmp_path/'outputs/logs';logs.mkdir(parents=True)
    stable=logs/'stable.log';stable.write_text('retained')
    vanished=logs/'rotated.log';vanished.write_text('rotated after listing')
    original=Path.read_bytes
    def read(path):
        if path==vanished:raise FileNotFoundError('file rotated after listing')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',read)
    monkeypatch.setattr(archive.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=b'git metadata',returncode=0))
    destination=tmp_path/'experiment';archive.capture_files(destination)
    assert (destination/'outputs/logs/stable.log').read_text()=='retained'
    manifest=json.loads((destination/'files.json').read_text())
    assert any(row['path']=='outputs/logs/stable.log' for row in manifest)
    errors=json.loads((destination/'read-errors.json').read_text())
    assert errors[0]['path']=='outputs/logs/rotated.log'

def test_recording_alignment_restores_idle_speaker_gap_without_touching_raw(tmp_path):
    import wave
    import numpy as np
    alignment=load('align_recording')
    session=dict(started_ts=1000,devices={'speaker':{}})
    (tmp_path/'audio-session.json').write_text(json.dumps(session))
    (tmp_path/'audio-finished.json').write_text('{}')
    events=[dict(track='speaker',offset_frames=0,frames=100,wall_ts=1000,device_timing={'input_buffer_adc_time':10,'current_time':10}),
            dict(track='speaker',offset_frames=100,frames=100,wall_ts=1002,device_timing={'input_buffer_adc_time':12,'current_time':12})]
    (tmp_path/'audio-timeline.jsonl').write_text('\n'.join(json.dumps(event) for event in events))
    source=np.concatenate([np.ones(100,dtype='<i2')*100,np.ones(100,dtype='<i2')*200])
    path=tmp_path/'speaker.wav'
    with wave.open(str(path),'wb') as writer:
        writer.setnchannels(1);writer.setsampwidth(2);writer.setframerate(1000);writer.writeframes(source.tobytes())
    original=path.read_bytes();report=alignment.align(tmp_path)
    with wave.open(str(tmp_path/'speaker-aligned.wav'),'rb') as reader:
        result=np.frombuffer(reader.readframes(reader.getnframes()),dtype='<i2')
    assert path.read_bytes()==original
    np.testing.assert_array_equal(result[:100],source[:100])
    assert not np.any(result[100:2000])
    np.testing.assert_array_equal(result[2000:],source[100:])
    assert report['speaker']['segments']==2 and report['speaker']['inserted_silence_frames']==1900

def test_read_aloud_rubrics_enforce_user_ownership_and_actual_instructions():
    rubrics=load('read_aloud_rubrics')
    assert not rubrics.grade('B1','My name is Maya. My favorite color is green.')['passed']
    assert not rubrics.grade('B1','Please remember both facts and acknowledge briefly.')['passed']
    assert rubrics.grade('B1','Got it. Your name is Maya and your favorite color is green.')['passed']
    assert not rubrics.grade('B4','I am going to Bristol on Tuesday.')['passed']
    assert rubrics.grade('B4',"I'll remember those trip details.")['passed']
    assert not rubrics.grade('B2','You can call me Maya.')['passed']
    assert not rubrics.grade('B3','My favorite color is green.')['passed']
    assert rubrics.grade('B2','Your name is Maya.')['passed']
    assert rubrics.grade('B3','Your favorite color is green.')['passed']
    assert not rubrics.grade('A3','Two plus two equals four.')['passed']
    assert rubrics.grade('A3','Four.')['passed']
    assert rubrics.grade('C1','Here is the code: B7D2')['passed']
    assert rubrics.grade('C1','B7D2')['manual_audio_format_review']
    assert not rubrics.grade('B6','The train leaves at 10:15 in Bristol on Tuesday. Anything else?')['passed']


def test_experiment_checkpoint_never_replaces_prior_backup_and_verifies_bytes(tmp_path):
    backup=load('checkpoint_experiment');source=tmp_path/'source';source.mkdir();(source/'reply.wav').write_bytes(b'original audio')
    first=backup.checkpoint(source,tmp_path/'backups')
    (source/'reply.wav').write_bytes(b'new audio')
    second=backup.checkpoint(source,tmp_path/'backups')
    assert first!=second
    assert (first/'experiment/reply.wav').read_bytes()==b'original audio'
    assert (second/'experiment/reply.wav').read_bytes()==b'new audio'
    assert (first/'READY').exists() and (second/'READY').exists()
    assert json.loads((first/'checkpoint.json').read_text())['verified']

def test_checkpoint_verifier_detects_changed_backup(tmp_path):
    import pytest
    backup=load('checkpoint_experiment');source=tmp_path/'source';source.mkdir();(source/'reply.wav').write_bytes(b'original')
    destination=backup.checkpoint(source,tmp_path/'backups')
    assert backup.verify_checkpoint(destination)==1
    (destination/'experiment/reply.wav').write_bytes(b'corrupted')
    with pytest.raises(RuntimeError,match='integrity failure'):
        backup.verify_checkpoint(destination)


def test_closed_loop_frame_completion_does_not_advance_on_idle_timeouts():
    import threading
    import queue
    rig=load('closed_loop_capture');frames=rig.StepQueue();consumed=threading.Event();release=threading.Event()
    def consumer():
        frames.get(timeout=1);consumed.set();release.wait(1)
        for _ in range(2):
            try:frames.get(timeout=.01)
            except queue.Empty:pass
    worker=threading.Thread(target=consumer);worker.start();frames.put('frame')
    try:
        assert consumed.wait(1) and frames.completed==0
    finally:release.set()
    frames.wait_completed(1);worker.join(1)
    assert not worker.is_alive() and frames.completed==1


def test_capture_scoring_counts_word_loss_and_rejects_changed_negation():
    rig=load('closed_loop_capture')
    assert rig.word_errors('I have twelve apples','I have 12 apples')['wer']==0
    assert rig.word_errors('Say only banana','Hello say only banana')['insertions']==1
    assert rig.word_errors('Wait stop let me think','Wait let me think')['deletions']==1
    assert not rig.no_new_negation('Say only the word banana','Not only the word banana')
    assert not rig.no_new_negation('Stop','Do not stop')
    assert rig.no_new_negation('Do not stop',"Don't stop")


def test_aec_gate_checks_actual_candidate_floor_and_requires_both_delays():
    rig=load('dual_aec_experiment')
    cases=[dict(configured_delay_ms=delay,residual_floor=floor,quiet_h3_onsets=[{}] if floor==0 else [])
           for delay in [40,80] for floor in [0,.005]]
    assert rig.h3_acceptance(cases,0)==dict(false_onsets=2,validated=True)
    assert rig.h3_acceptance(cases,.005)==dict(false_onsets=0,validated=True)
    assert not rig.h3_acceptance(cases,.003)['validated']
    assert not rig.h3_acceptance(cases[:2],.005)['validated']


def test_capture_comparison_exposes_new_misses_even_when_both_versions_fail():
    rig=load('closed_loop_capture')
    result=rig.compare_capture([dict(case=1,version='baseline',passed=False,clips=1,barge_s=.52),
                                dict(case=1,version='candidate',passed=False,clips=0,barge_s=None)])
    assert result['lost_baseline_passes']==[] and result['new_missing_clips']==[1]
    assert not result['full_gate_passed']


def test_contextual_endpoint_restores_fast_fresh_turn_after_interruption():
    import numpy as np
    from duplex import DuplexFrontend
    rig=load('closed_loop_capture');Front=rig.contextual_frontend(DuplexFrontend,1120)
    front=Front(aec=False,end_silence_ms=480,holdoff_s=0)
    front._prob=lambda chunk:float(np.mean(chunk)>.05)
    delays=[]
    for interruption in [True,False]:
        for _ in range(32):
            _,started,_,_=front.update(np.full(160,.1,np.float32))
            if started and interruption:front.use_barge_endpoint()
        begin=front._sample_clock
        for _ in range(200):
            _,_,ended,_=front.update(np.zeros(160,np.float32))
            if ended:break
        else:raise AssertionError('Speech did not end')
        delays.append(front._sample_clock-begin)
        front.reset()
        assert front.end_chunks==15
    assert delays==[1120*16,480*16]


def test_silent_control_rejects_false_speech_even_if_clip_is_discarded():
    rig=load('closed_loop_capture');row=dict(false_cutoff=False,clips=0,error=None,aec_error=None)
    assert rig.silent_control_passed(row,[])
    assert not rig.silent_control_passed(row,[dict(event='speech_start')])


def test_native_recovery_rejects_slow_valid_audio_after_cancellation():
    rig=load('native_abort_integration')
    rows=[dict(version='clean',case='healthy_control',first_audio_s=.5,passed=True),
          dict(version='clean',case='recovery',first_audio_s=.7,passed=True),
          dict(version='clean',case='natural_recovery',first_audio_s=7.,passed=True)]
    result=rig.recovery_timing(rows)['by_version']['clean']
    assert not result['passed']
    assert [case['passed'] for case in result['cases']]==[True,False]


def test_human_replay_refuses_outputs_inside_historical_experiment(tmp_path):
    import pytest
    rig=load('replay_human_acoustics');source=tmp_path/'human';source.mkdir()
    (source/'original.txt').write_text('preserve this recording history')
    with pytest.raises(ValueError,match='separate experiment'):rig.main(source)
    for output in [source,source/'new-results']:
        with pytest.raises(ValueError,match='outside'):rig.main(source,output_folder=output)
    assert sorted(p.name for p in source.iterdir())==['original.txt']
