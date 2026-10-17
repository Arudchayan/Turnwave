"""Build a private AEC variant using upstream near-end masking during normal mode."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint


def run(folder):
    folder=folder.resolve();folder.mkdir(parents=True,exist_ok=False);capture_files(folder/'source')
    original=ROOT/'outputs/experiments/webrtc-linear-v1';source=folder/'variant'
    shutil.copytree(original/'variant',source)
    binding=source/'bindings/webrtc_audio_bindings.cpp';before=binding.read_text()
    needle='config.filter.export_linear_aec_output = true;'
    if before.count(needle)!=2:raise RuntimeError('Expected AudioProcessor constructor and reset config sites')
    after=before.replace(needle,needle+'\n            config.suppressor.normal_tuning = config.suppressor.nearend_tuning;')
    binding.write_text(after)
    (folder/'binding.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='linear/bindings.cpp',tofile='nearend/bindings.cpp')))
    cmake=shutil.which('cmake')
    if not cmake:raise RuntimeError('CMake unavailable')
    commands=[['--version'],['-S',str(source),'-B',str(folder/'build'),'-G','Visual Studio 17 2022','-A','x64',
        '-DPYBIND11_FINDPYTHON=ON','-DPython_EXECUTABLE='+sys.executable,
        '-Dpybind11_DIR='+str(original/'build-deps/pybind11/share/cmake/pybind11')],
        ['--build',str(folder/'build'),'--config','Release','--parallel','4']]
    (folder/'provenance.json').write_text(json.dumps(dict(baseline_source=str(original),
        baseline_binding_sha256=hashlib.sha256((original/'variant/bindings/webrtc_audio_bindings.cpp').read_bytes()).hexdigest(),modified_binding_sha256=hashlib.sha256(binding.read_bytes()).hexdigest(),
        hypothesis='Use the official near-end masking thresholds in normal mode too; leave the native near-end detector, linear AEC, gains and frontend policy unchanged.',
        installed=False,cmake=cmake,commands=commands,python=sys.version),indent=2))
    for index,args in enumerate(commands):
        with (folder/f'build-{index}.log').open('w') as log:
            subprocess.run([cmake,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    artifact=folder/'build/Release/_webrtc_audio_linear.cp312-win_amd64.pyd'
    shutil.copy2(artifact,folder/artifact.name)
    (folder/'build-complete.json').write_text(json.dumps(dict(artifact=artifact.name,bytes=artifact.stat().st_size,
        sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),installed=False),indent=2))
    print('Private research build complete:',folder/artifact.name,flush=True)


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
