"""Stage and atomically select a hash-verified local runtime; no model launch."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runner_selection import selected_runner


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def replace(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.localvoice-tmp')
    with temporary.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def prepare(folder,proof):
    summary=json.loads((proof/'summary.json').read_text())
    if not summary['all_request_controls_passed'] or not summary['candidate_recovery_gate_passed']:
        raise RuntimeError('GPU candidate proof failed')
    tested=json.loads((proof/'lifecycle/server-process.json').read_text())
    if tested['failure_injection'] or tested['command'][tested['command'].index('-ngl')+1]!='99':
        raise RuntimeError('Proof is not the clean GPU candidate')
    private=(Path.home()/'localvoice-native-abort-v1')
    binary=private/'lifecycle/llama-liquid-audio-server'
    if digest(binary)!=tested['binary_sha256']:raise RuntimeError('Candidate binary changed since GPU validation')
    build=json.loads((private/'lifecycle/manifest.json').read_text())
    if build['libraries_manifest_sha256']!=digest(private/'manifest.json'):raise RuntimeError('Native library manifest changed')
    for item in json.loads((private/'manifest.json').read_text())['files']:
        if item['copy'].startswith('lib/') and digest(private/item['copy'])!=item['sha256']:
            raise RuntimeError('Native build library changed: '+item['copy'])
    for model in tested['models']:
        if digest(model['path'])!=model['sha256']:raise RuntimeError('Model changed since GPU validation')
    if b'LOCALVOICE_ABORT_PROBE_FAIL_ONCE' in binary.read_bytes():raise RuntimeError('Test hook in candidate')
    package=Path.home()/'.local/share/localvoice/runners'/folder.name
    package.mkdir(parents=True,exist_ok=False)
    for name in ['lifecycle','lib']:shutil.copytree(private/name,package/name,symlinks=True)
    for name in ['native-source.tar.gz','native-working.patch','manifest.json']:
        shutil.copy2(private/name,package/name)
    runtime_binary=package/'lifecycle/llama-liquid-audio-server'
    files={str(p.resolve()):digest(p) for p in [runtime_binary,*list((package/'lib').glob('*.so*'))]}
    # Relocation must not change any binary or library bytes.
    if digest(runtime_binary)!=tested['binary_sha256']:raise RuntimeError('Relocation changed the binary')
    for source in (private/'lib').glob('*.so*'):
        if digest(source)!=digest(package/'lib'/source.name):raise RuntimeError('Relocation changed a library')
    config=Path.home()/'.config/localvoice/runner.json'
    record=dict(format=1,binary=str(runtime_binary),library_directory=str(package/'lib'),files=files,
        gpu_proof=str(proof),gpu_summary_sha256=digest(proof/'summary.json'),source_revision='ec9d1fdd9',
        local_patch='liquid-abort-lifecycle.patch',runtime_package=str(package))
    candidate=folder/'candidate-selection.json';candidate.write_text(json.dumps(record,indent=2))
    selected_runner(candidate)
    (folder/'selection-rollback.json').write_text(json.dumps(dict(path=str(config),
        previous_hex=config.read_bytes().hex() if config.exists() else None),indent=2))
    with tarfile.open(folder/'candidate-runtime.tar.gz','x:gz') as archive:archive.add(package,arcname='runtime')
    print(json.dumps(record))


def switch(folder,rollback=False):
    saved=json.loads((folder/'selection-rollback.json').read_text());config=Path(saved['path'])
    expected=Path.home()/'.config/localvoice/runner.json'
    if config!=expected:raise RuntimeError('Unexpected selection path')
    previous=bytes.fromhex(saved['previous_hex']) if saved['previous_hex'] is not None else None
    candidate=(folder/'candidate-selection.json').read_bytes()
    current=config.read_bytes() if config.exists() else None
    if current!=(candidate if rollback else previous):raise RuntimeError('Selection changed outside this transaction')
    if rollback and previous is None:config.unlink()
    else:
        if not rollback:selected_runner(folder/'candidate-selection.json')
        replace(config,previous if rollback else candidate)


if __name__=='__main__':
    mode=sys.argv[1];folder=Path(sys.argv[2])
    if mode=='prepare':prepare(folder,Path(sys.argv[3]))
    elif mode=='activate':switch(folder)
    elif mode=='rollback':switch(folder,True)
    else:raise ValueError('Unknown activation mode')
