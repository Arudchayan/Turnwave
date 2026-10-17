"""Research-only longer pause allowance after two seconds of an active turn."""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint


def run(folder,source):
    folder.mkdir(parents=True,exist_ok=False);capture_files(folder/'source')
    text=source.read_text();(folder/'input-duplex.py').write_bytes(source.read_bytes())
    edits={
        '        self.end_chunks = max(1, round(end_silence_ms / 32))\n':'''        self.end_chunks = max(1, round(end_silence_ms / 32))
        self._short_end_chunks = self.end_chunks
        self._utterance_start_clock = None
''',
        '    def reset(self):\n        with self._lock:\n':'''    def reset(self):
        with self._lock:
            self.end_chunks = self._short_end_chunks
            self._utterance_start_clock = None
''',
        '                        self.speaking = True\n                        started = True\n':'''                        self.speaking = True
                        self._utterance_start_clock = self._sample_clock
                        started = True
''',
        '            else:\n                self._quiet = 0 if hot else self._quiet + 1\n':'''            else:
                if hot and self._utterance_start_clock is not None and self._sample_clock - self._utterance_start_clock >= 32000:
                    self.end_chunks = max(self._short_end_chunks, 35)
                self._quiet = 0 if hot else self._quiet + 1
''',
        '                    self.speaking = False\n                    ended = True\n':'''                    self.speaking = False
                    self.end_chunks = self._short_end_chunks
                    self._utterance_start_clock = None
                    ended = True
'''}
    for old,new in edits.items():
        if text.count(old)!=1:raise RuntimeError('Unexpected endpoint source anchor')
        text=text.replace(old,new)
    (folder/'candidate-duplex.py').write_text(text)
    (folder/'protocol.json').write_text(json.dumps(dict(active_turn_ms_before_extension=2000,
        require_current_speech=True,extended_silence_ms=1120,short_silence_ms=480,
        input_source=str(source.resolve()),deployed=False),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('source',type=Path);a=p.parse_args()
    try:run(a.folder,a.source)
    except Exception:
        if a.folder.exists():(a.folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if a.folder.exists():print('Verified backup:',checkpoint(a.folder),flush=True)
