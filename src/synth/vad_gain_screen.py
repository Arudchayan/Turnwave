"""Offline VAD amplitude screen; acoustic capture and echo guards are unchanged.

Uses the real frontend on seeded acoustics and saved human H3 audio. No device
or model backend is opened, and no historical experiment folder is modified.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from synth import acoustic_stress,replay_human_acoustics
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint


def run(folder):
    import torch
    torch.set_num_threads(1)
    folder.mkdir(parents=True,exist_ok=False);capture_files(folder/'source')
    source=(ROOT/'src/duplex.py').read_text()
    original='torch.from_numpy(chunk[None, :])'
    if source.count(original)!=1:raise RuntimeError('Unexpected frontend VAD input')
    reports=[]
    for gain in [1,2,4,8]:
        destination=folder/f'gain-{gain}';destination.mkdir()
        path=destination/'candidate-duplex.py'
        path.write_text(source.replace(original,f'torch.from_numpy((chunk * {gain}.)[None, :])'))
        spec=importlib.util.spec_from_file_location(f'vad_gain_{gain}',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        previous=acoustic_stress.DuplexFrontend
        try:
            acoustic_stress.DuplexFrontend=module.DuplexFrontend
            acoustic_stress.run(destination/'acoustic.json')
        finally:acoustic_stress.DuplexFrontend=previous
        replay_human_acoustics.main(ROOT/'outputs/experiments/20261016T082545793892Z-human-read-aloud-v1',
            output_folder=destination,frontend=module.DuplexFrontend)
        acoustic=json.loads((destination/'acoustic.json').read_text())
        human=json.loads((destination/'h3-acoustic-replay-floor005.json').read_text())
        guarded=[r for r in human['cases'] if r['residual_floor']==.005]
        row=dict(gain=gain,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            acoustic_passes=sum(r['passed'] for r in acoustic['cases']),acoustic_total=len(acoustic['cases']),
            quiet_double_talk=[r for r in acoustic['cases'] if r.get('scenario')=='quiet_double_talk'],
            h3_false_onsets=sum(len(r['quiet_h3_onsets']) for r in guarded),h3_replay_valid=len(guarded)==2 and not any(r['aec_error'] for r in guarded))
        reports.append(row);(folder/'screen.json').write_text(json.dumps(dict(cases=reports,deployed=False,
            scope='Detection-only fixed gain; original energy/correlation guards and captured audio unchanged. H3 is saved-audio replay, not fresh room acceptance.'),indent=2))
        print('SCREEN',json.dumps(row),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);args=parser.parse_args()
    try:run(args.folder)
    except Exception:
        if args.folder.exists():(args.folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if args.folder.exists():print('Verified backup:',checkpoint(args.folder),flush=True)
