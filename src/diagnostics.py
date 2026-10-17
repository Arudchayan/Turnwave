"""Bounded in-memory diagnostics with lossless, rotating disk history.

No microphone audio or transcript is stored by this component.
"""
import collections
import json
import hashlib
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import threading
import time
import uuid


def source_fingerprint():
    """Record source hashes at startup so later edits cannot blur provenance."""
    folder=Path(__file__).resolve().parent
    return {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(folder.glob('*.py'))}


class HistoryFileHandler(RotatingFileHandler):
    """Rotate into unique history files; never prune an older experiment."""
    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        path = Path(self.baseFilename)
        if path.exists():
            destination = path.with_name(path.name + '.' + str(time.time_ns()) + '.' + uuid.uuid4().hex[:8])
            path.rename(destination)
        if not self.delay:
            self.stream = self._open()


class Diagnostics:
    def __init__(self, path=None):
        self._events = collections.deque(maxlen=200)
        self._lock = threading.Lock()
        self._handler = None
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handler = HistoryFileHandler(path, maxBytes=2_000_000, backupCount=0, encoding='utf-8')
            self._handler.setFormatter(logging.Formatter('%(message)s'))

    def event(self, event, **fields):
        record = dict(ts=round(time.time(), 3), pid=os.getpid(), event=event, **fields)
        with self._lock:
            self._events.append(record)
            if self._handler:
                message = json.dumps(record, ensure_ascii=True, allow_nan=False)
                self._handler.handle(logging.LogRecord('localvoice', logging.INFO, '', 0, message, (), None))
        return record

    def recent(self, limit=40):
        with self._lock:
            return list(self._events)[-limit:]

    def close(self):
        with self._lock:
            if self._handler:
                self._handler.close()
                self._handler = None
