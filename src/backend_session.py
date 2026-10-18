"""Process-wide ownership of the runner's single conversation context.

The OS releases the lock on process exit, including a crash. A leftover lock file
is not evidence that another owner is alive. No request or context reset occurs.
"""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import socket
import sys
import tempfile


def canonical_base(base):
    return base.rstrip('/').replace('http://localhost:', 'http://127.0.0.1:')


@contextmanager
def backend_session(base='http://127.0.0.1:8080', *, test=False, directory=None):
    if test:
        try:
            connection=socket.create_connection(('127.0.0.1',8000),timeout=.5)
        except OSError:
            pass
        else:
            connection.close()
            raise RuntimeError('Stop the UI before backend tests; muted conversations still own model context.')
    folder=Path(directory) if directory is not None else Path(tempfile.gettempdir())/'localvoice-context'
    folder.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256(canonical_base(base).encode('utf-8')).hexdigest()[:24]
    handle=(folder/(key+'.lock')).open('a+b')
    locked=False
    try:
        handle.seek(0,2)
        if handle.tell()==0:
            handle.write(b'0');handle.flush()
        handle.seek(0)
        try:
            if sys.platform=='win32':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            locked=True
        except OSError as exc:
            raise RuntimeError('Another Turnwave app or test owns this backend context. Let it finish before starting another.') from exc
        yield
    finally:
        if locked:
            handle.seek(0)
            if sys.platform=='win32':
                msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
        handle.close()
