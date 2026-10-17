"""Prepare and archive a private native server build in WSL."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint
from synth.native_abort_probe import wsl_path,wsl_home


def run(folder,private,clean=False,lifecycle=False):
    folder.mkdir(parents=True,exist_ok=False);capture_files(folder/'source')
    probe=ROOT/'outputs/experiments/native-abort-probe-v2'
    shutil.copytree(probe,folder/'source-input')
    (folder/'build-plan.json').write_text(json.dumps(dict(private_path=private,source_probe=str(probe),live_server_modified=False,clean=clean,lifecycle=lifecycle),indent=2))
    command=['wsl.exe','-e','python3',wsl_path(ROOT/'src/synth/build_abort_variant_linux.py'),private,wsl_path(folder)]
    if clean:command.append('--clean')
    if lifecycle:command.append('--lifecycle')
    with (folder/'build.log').open('wb') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Private build archived:',folder,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);parser.add_argument('--private',default=None);parser.add_argument('--clean',action='store_true');parser.add_argument('--lifecycle',action='store_true');args=parser.parse_args()
    if args.folder.exists():parser.error('Use a new experiment folder')
    if args.clean and args.lifecycle:parser.error('Choose one build variant')
    try:run(args.folder,args.private or wsl_home()+'/localvoice-native-abort-v1',args.clean,args.lifecycle)
    except Exception:
        if args.folder.exists():
            import traceback
            (args.folder/'host-failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if args.folder.exists():print('Verified backup:',checkpoint(args.folder),flush=True)
