"""Compare synthesized short utterances through the virtual capture loop.

Research only: this uses one generated far-end voice, a simple echo path and
explicit local WAV fixtures. It neither opens devices nor changes app settings.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from synth import duplex_barge_probe as probe
from floor import FloorPolicy
import app as app_module

FIXTURES = tuple(f'{voice}-{phrase}.wav' for voice in ('david', 'zira')
                 for phrase in ('ack', 'yes', 'stop', 'wait'))


def prepare_fixture(path):
    audio, rate = probe.read_wav(path)
    audio = probe.resample(audio, rate)
    hot = [i for i in range(0, len(audio) - 319, 320)
           if float(np.sqrt(np.mean(audio[i:i + 320] ** 2))) > .003]
    if not hot:
        raise ValueError(f'No audible frames in {path.name}')
    audio = audio[max(0, hot[0] - 320):min(len(audio), hot[-1] + 640)]
    return audio * (.06 / float(np.sqrt(np.mean(audio ** 2))))


def run(folder, output, floor_ms=300, active_echo_gate=None):
    if floor_ms < 50 or (active_echo_gate is not None and not 0 < active_echo_gate <= 1):
        raise ValueError('Invalid experiment setting')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    paths = [folder / name for name in FIXTURES]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise ValueError('Missing fixtures: ' + ', '.join(missing))
    far, rate = probe.read_wav(ROOT / 'research/examples/pass-reply.wav')
    far = probe.resample(far, rate)
    far *= .05 / float(np.sqrt(np.mean(far ** 2)))
    old_front, old_floor = probe.DuplexFrontend, app_module.FloorPolicy
    env_names = ('LOCALVOICE_FLOOR_MODE', 'LOCALVOICE_EOT_MODEL')
    previous_env = {name: os.environ.pop(name, None) for name in env_names}
    try:
        class GatedFrontend(old_front):
            def _prob(self, chunk):
                probability = super()._prob(chunk)
                if (active_echo_gate is not None and self.speaking and self._far_active
                        and self.echo_score >= active_echo_gate):
                    return 0.0
                return probability
        probe.DuplexFrontend = GatedFrontend
        app_module.FloorPolicy = lambda **kwargs: FloorPolicy(
            backchannel_max_ms=floor_ms - 50, floor_confirm_ms=floor_ms, **kwargs)
        rows = []
        control = probe.run_case('echo_only', 'reversible', 1.0, far,
                                 np.zeros(16000, dtype=np.float32))
        for path in paths:
            near = prepare_fixture(path)
            for gain in (1.0, .5):
                result = probe.run_case('utterance', 'reversible', gain, far, near)
                result.update(fixture=path.name,
                              fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              near_normalized_sha256=hashlib.sha256(near.tobytes()).hexdigest())
                rows.append(result)
        if any(row['error'] or row['aec_error'] for row in [control, *rows]):
            raise RuntimeError('Capture or echo processing failed')
        report = {'scope': 'Synthesized utterances, virtual devices, one simulated echo path; no human or device.',
                  'floor_confirm_ms': floor_ms, 'backchannel_max_ms': floor_ms - 50,
                  'active_echo_gate': active_echo_gate,
                  'far_normalized_sha256': hashlib.sha256(far.tobytes()).hexdigest(),
                  'echo_only': control, 'rows': rows}
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8', newline='\n') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')
    finally:
        probe.DuplexFrontend, app_module.FloorPolicy = old_front, old_floor
        for name, value in previous_env.items():
            if value is not None:
                os.environ[name] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--floor-ms', type=int, default=300)
    parser.add_argument('--active-echo-gate', type=float)
    args = parser.parse_args()
    run(args.fixtures, args.out, args.floor_ms, args.active_echo_gate)


if __name__ == '__main__':
    main()
