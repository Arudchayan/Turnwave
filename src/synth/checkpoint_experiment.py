"""Make an immutable, hash-verified local copy of a saved experiment."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def checkpoint(source,backup_root=None):
    source=Path(source).resolve()
    if not source.is_dir():raise ValueError('Experiment folder does not exist')
    backup_root=Path(backup_root) if backup_root else Path.home()/'LocalVoice-Experiment-Backups'
    backup_root=backup_root.resolve()
    if backup_root==source or source in backup_root.parents:raise ValueError('Backup must be outside the source experiment')
    backup_root.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination=backup_root/(source.name+'-'+stamp)
    destination.mkdir(exist_ok=False)
    files=[]
    try:
        for path in sorted(source.rglob('*')):
            if not path.is_file():continue
            relative=path.relative_to(source);target=destination/'experiment'/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
            original=digest(path);copied=digest(target)
            if original!=copied:raise RuntimeError('Source changed during copy or verification failed: '+str(relative))
            files.append(dict(path=relative.as_posix(),bytes=target.stat().st_size,sha256=copied))
        manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=str(source),files=files,
                      total_bytes=sum(row['bytes'] for row in files),verified=True,
                      note='Separate local path. This is not an off-device backup and does not protect against loss of both storage locations.')
        (destination/'checkpoint.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        (destination/'READY').write_text('All copied files matched their source SHA-256.\n',encoding='utf-8')
    except Exception as exc:
        (destination/'FAILED.json').write_text(json.dumps(dict(error=str(exc))),encoding='utf-8')
        raise
    return destination


def verify_checkpoint(folder):
    folder=Path(folder)
    if not (folder/'READY').is_file():raise RuntimeError('Checkpoint is incomplete; READY marker is missing')
    manifest=json.loads((folder/'checkpoint.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        path=folder/'experiment'/row['path']
        if not path.is_file() or path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
            raise RuntimeError('Checkpoint integrity failure: '+row['path'])
    return len(manifest['files'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('--backup-root',type=Path);parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    if args.verify:
        print('Verified existing checkpoint files:',verify_checkpoint(args.source),flush=True)
    else:
        destination=checkpoint(args.source,args.backup_root)
        print('Verified checkpoint:',destination,flush=True)
