"""Research-only AEC bypass after sustained audible render with a quiet mic.

Does not edit the live frontend. The candidate keeps AEC adapting internally
and revokes bypass on strong echo correlation. Requires full acoustic gates.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint


def run(folder):
    folder.mkdir(parents=True,exist_ok=False);capture_files(folder/'source')
    source=(ROOT/'src/duplex.py').read_text()
    edits={
        '        self._far_active = False\n        self._since_far = 16000\n':'        self._far_active = False\n        self._since_far = 16000\n        self._uncoupled_samples = 0\n        self._uncoupled_render = False\n        self._coupling_seen = False\n',
        '            far_out.append(far)\n':'''            far_out.append(far)
            dry = near.copy()
            if not self._far_active:
                self._uncoupled_samples = 0
                self._uncoupled_render = self._coupling_seen = False
            if self._uncoupled_render and self.echo_score >= .6:
                self._uncoupled_render = False
                self._coupling_seen = True
            if self._far_active and not self._coupling_seen and not self._uncoupled_render:
                far_rms = float(np.sqrt(np.mean(far ** 2)))
                raw_rms = float(np.sqrt(np.mean(dry ** 2)))
                if far_rms >= .01:
                    if raw_rms >= .002:
                        self._coupling_seen = True
                        self._uncoupled_samples = 0
                    elif raw_rms < .001:
                        self._uncoupled_samples += F_AEC
                if self._uncoupled_samples >= 4800:
                    self._uncoupled_render = True
''',
        '            out.append(near)\n':'''            if self._uncoupled_render:
                near = dry
            out.append(near)
'''}
    for old,new in edits.items():
        # The first far-active initialization is unique to __init__; later
        # assignments use different indentation and are deliberately unchanged.
        if source.count(old)!=1:raise RuntimeError('Unexpected candidate anchor: '+old)
        source=source.replace(old,new)
    (folder/'candidate-duplex.py').write_text(source)
    (folder/'protocol.json').write_text(json.dumps(dict(calibration_audible_ms=300,far_rms_min=.01,
        quiet_mic_rms_max=.001,coupling_seen_rms=.002,revoke_correlation=.6,
        deployed=False,scope='Research-only no-coupling bypass. Does not solve quiet double talk with real echo.'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);args=p.parse_args()
    try:run(args.folder)
    except Exception:
        if args.folder.exists():(args.folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if args.folder.exists():print('Verified backup:',checkpoint(args.folder),flush=True)
