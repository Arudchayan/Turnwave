"""Exec a private CPU or explicitly reserved GPU runner for integration tests."""
import hashlib
import json
import os
from pathlib import Path
import sys

private=Path(sys.argv[1]);version=sys.argv[2];output=Path(sys.argv[3])
gpu='--gpu' in sys.argv[4:]
if version not in ['before','after','clean','lifecycle']:raise ValueError('Unknown private variant')
binary=private/version/'llama-liquid-audio-server'
models=Path.home()/'.cache/localvoice/models'
names=['LFM2.5-Audio-1.5B-Q8_0.gguf','mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf',
       'vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf','tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf']
(output/'launcher-process.json').write_text(json.dumps(dict(pid=os.getpid(),command_hex=Path('/proc/self/cmdline').read_bytes().hex()),indent=2))
inventory=[]
for name in names:
    path=models/name;digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    inventory.append(dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest()))
command=[str(binary),'-m',str(models/names[0]),'-mm',str(models/names[1]),'-mv',str(models/names[2]),
    '--tts-speaker-file',str(models/names[3]),'-ngl','0','--device','none','--no-mmproj-offload','--no-op-offload',
    '--fit','off','-t','2','-tb','2','-c','2048','--host','127.0.0.1','--port','8081']
os.environ['LD_LIBRARY_PATH']=str(private/'lib')+':'+str(Path.home()/'miniforge/envs/cudaenv/lib')
if gpu:
    command=[str(binary),'-m',str(models/names[0]),'-mm',str(models/names[1]),'-mv',str(models/names[2]),
        '--tts-speaker-file',str(models/names[3]),'-ngl','99','--fit','off','-c','4096',
        '--host','127.0.0.1','--port','8081']
    os.environ.pop('CUDA_VISIBLE_DEVICES',None)
else:os.environ['CUDA_VISIBLE_DEVICES']=''
if gpu or version in ['clean','lifecycle']:os.environ.pop('LOCALVOICE_ABORT_PROBE_FAIL_ONCE',None)
else:os.environ['LOCALVOICE_ABORT_PROBE_FAIL_ONCE']='1'
(output/'server-process.json').write_text(json.dumps(dict(pid=os.getpid(),command=command,models=inventory,
    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
    ld_library_path=os.environ['LD_LIBRARY_PATH'],failure_injection=not gpu and version not in ['clean','lifecycle']),indent=2))
os.execv(command[0],command)
