"""Verify a locally selected native runner before the WSL launcher uses it."""
import argparse
import hashlib
import json
from pathlib import Path


def selected_runner(config):
    config=Path(config)
    if not config.exists():return None
    record=json.loads(config.read_text())
    if record.get('format')!=1:raise ValueError('Unsupported runner selection format')
    binary=Path(record['binary']).resolve(strict=True)
    library=Path(record['library_directory']).resolve(strict=True)
    if binary.name!='llama-liquid-audio-server' or not library.is_dir():raise ValueError('Invalid native runner layout')
    required={str(binary)}|{str(p.resolve(strict=True)) for p in library.glob('*.so*')}
    files=record['files']
    if not required.issubset(files):raise ValueError('Runner verification omits runtime files')
    for name,expected in files.items():
        path=Path(name)
        if not path.is_absolute():raise ValueError('Runner file paths must be absolute')
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=expected:raise ValueError('Selected runner file changed: '+str(path))
    if any(c in str(binary.parent)+str(library) for c in '\n\r'):raise ValueError('Runner paths contain a newline')
    return str(binary.parent),str(library)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path.home()/'.config/localvoice/runner.json')
    result=selected_runner(parser.parse_args().config)
    if result:print('\n'.join(result))
