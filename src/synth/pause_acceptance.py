"""Offline pause-boundary screen; never opens audio devices or a backend.

The duration candidate is research source only. Clip boundaries and timing do
not establish word retention, intelligibility, or real-room echo acceptance.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
import traceback
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def duration_candidate_source(source):
    """Reproduce the prior 2 s / 1120 ms duration-only research rule exactly."""
    edits = {
        '        self.end_chunks = max(1, round(end_silence_ms / 32))\n':
        '        self.end_chunks = max(1, round(end_silence_ms / 32))\n'
        '        self._short_end_chunks = self.end_chunks\n'
        '        self._utterance_start_clock = None\n',
        '    def reset(self):\n        with self._lock:\n':
        '    def reset(self):\n        with self._lock:\n'
        '            self.end_chunks = self._short_end_chunks\n'
        '            self._utterance_start_clock = None\n',
        '                        self.speaking = True\n                        started = True\n':
        '                        self.speaking = True\n'
        '                        self._utterance_start_clock = self._sample_clock\n'
        '                        started = True\n',
        '            else:\n                self._quiet = 0 if hot else self._quiet + 1\n':
        '            else:\n'
        '                if hot and self._utterance_start_clock is not None and self._sample_clock - self._utterance_start_clock >= 32000:\n'
        '                    self.end_chunks = max(self._short_end_chunks, 35)\n'
        '                self._quiet = 0 if hot else self._quiet + 1\n',
        '                    self.speaking = False\n                    ended = True\n':
        '                    self.speaking = False\n'
        '                    self.end_chunks = self._short_end_chunks\n'
        '                    self._utterance_start_clock = None\n'
        '                    ended = True\n',
    }
    for old, new in edits.items():
        if source.count(old) != 1:
            raise ValueError('Duration candidate source anchor changed')
        source = source.replace(old, new)
    return source


def load_frontend(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DuplexFrontend


def replay(frontend, audio, far=None):
    """20 ms CPU replay, with a frame trace preserved for boundary review."""
    front = frontend(aec=False)
    front.warmup()
    trace = []
    for offset in range(0, len(audio), 320):
        if far is not None:
            front.push_far(far[offset:offset + 320])
        begin = time.perf_counter()
        speaking, started, ended, _ = front.update(audio[offset:offset + 320])
        trace.append(dict(sample_end=min(offset + 320, len(audio)),
                          started=started, ended=ended, speaking=speaking,
                          probability=front.probability, echo_score=front.echo_score,
                          end_chunks=front.end_chunks,
                          processing_ms=(time.perf_counter() - begin) * 1000))
        if ended:
            front.reset()
    return trace


def summarize(trace):
    return dict(starts_s=[r['sample_end'] / 16000 for r in trace if r['started']],
                ends_s=[r['sample_end'] / 16000 for r in trace if r['ended']],
                max_end_chunks=max((r['end_chunks'] for r in trace), default=0),
                p99_processing_ms=float(np.percentile([r['processing_ms'] for r in trace], 99)))


def assess(case, baseline, candidate):
    """Require the observed fix AND preserve controls; matching totals is unsafe."""
    reasons = []
    if case['kind'] == 'pause':
        boundary = case['protected_until_s']
        reproduced = any(t < boundary for t in baseline['ends_s'])
        if not reproduced:
            reasons.append('baseline_failure_not_reproduced')
        if any(t < boundary for t in candidate['ends_s']):
            reasons.append('premature_endpoint_remains')
        if not candidate['ends_s']:
            reasons.append('candidate_never_completes')
    elif case['kind'] == 'short':
        if len(baseline['ends_s']) != 1 or len(candidate['ends_s']) != 1:
            reasons.append('short_clip_count')
        elif candidate['ends_s'][0] - baseline['ends_s'][0] > case.get('latency_tolerance_s', .032) + 1e-9:
            reasons.append('short_latency_regression')
        if len(candidate['starts_s']) != len(baseline['starts_s']):
            reasons.append('short_onset_count_regression')
    elif case['kind'] == 'quiet':
        if baseline['starts_s'] or candidate['starts_s']:
            reasons.append('quiet_false_onset')
    else:
        raise ValueError('Unknown pause acceptance case kind')
    return dict(passed=not reasons, reasons=reasons)


def read_audio(case, root):
    path = (root / case['path']).resolve()
    if digest(path) != case['sha256']:
        raise ValueError('Input audio hash mismatch')
    with wave.open(str(path), 'rb') as reader:
        if reader.getsampwidth() != 2:
            raise ValueError('Only PCM16 fixtures are supported')
        rate, channels = reader.getframerate(), reader.getnchannels()
        start = case.get('start_frame', 0)
        reader.setpos(start)
        count = case.get('frames', reader.getnframes() - start)
        pcm = np.frombuffer(reader.readframes(count), dtype='<i2').reshape(-1, channels)
    if len(pcm) != count:
        raise ValueError('Input window is truncated')
    channel = case.get('channel', 0)
    if not 0 <= channel < channels:
        raise ValueError('Invalid channel')
    samples = pcm[:, channel].astype(np.float32) / 32768
    positions = np.arange(round(len(samples) * 16000 / rate)) * rate / 16000
    samples = np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)
    return np.pad(samples, (0, round(case.get('tail_s', 0) * 16000)))


def validate_roster(protocol):
    """Validate output names and case coverage before any source load or writes."""
    cases = protocol.get('cases') if isinstance(protocol, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError('Protocol cases must be a nonempty list')
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError('Each case must be an object')
        case_id = case.get('id')
        if not isinstance(case_id, str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', case_id) is None:
            raise ValueError('Case ID must be 1-64 ASCII letters, digits, underscores or hyphens, starting with a letter or digit')
        if case_id.casefold() in seen:
            raise ValueError('Case IDs must be unique, ignoring case')
        seen.add(case_id.casefold())
        if case.get('kind') not in ('pause', 'short', 'quiet'):
            raise ValueError('Unsupported case kind')


def run(protocol_path, folder, source):
    if folder.exists():
        raise FileExistsError(folder)
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    validate_roster(protocol)
    folder.mkdir(parents=True, exist_ok=False)
    try:
        return _run(protocol_path, folder, source, protocol_bytes, protocol)
    except Exception:
        (folder / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def _run(protocol_path, folder, source, protocol_bytes, protocol):
    (folder / 'protocol.json').write_bytes(protocol_bytes)
    (folder / 'baseline-duplex.py').write_bytes(source.read_bytes())
    (folder / 'candidate-duplex.py').write_text(duration_candidate_source(source.read_text(encoding='utf-8')), encoding='utf-8')
    (folder / 'evaluation-source.py').write_bytes(Path(__file__).read_bytes())
    versions = {name: load_frontend(folder / f'{name}-duplex.py', f'pause_{name}') for name in ['baseline', 'candidate']}
    results = []
    for case in protocol['cases']:
        audio = read_audio(case, protocol_path.parent)
        np.save(folder / f"{case['id']}-input.npy", audio)
        far = read_audio(case['far'], protocol_path.parent) if 'far' in case else None
        if far is not None:
            if len(far) != len(audio):
                raise ValueError('Render and microphone fixtures must be aligned')
            np.save(folder / f"{case['id']}-render.npy", far)
        summaries = {}
        for name, frontend in versions.items():
            trace = replay(frontend, audio, far)
            (folder / f"{case['id']}-{name}-trace.json").write_text(json.dumps(trace), encoding='utf-8')
            summaries[name] = summarize(trace)
        result = dict(id=case['id'], kind=case['kind'], **summaries,
                      **assess(case, summaries['baseline'], summaries['candidate']))
        results.append(result)
        print(json.dumps(result), flush=True)
    report = dict(cases=results, accepted=all(r['passed'] for r in results), deployed=False,
                  scope=protocol['scope'], source_sha256=digest(source),
                  files_sha256={p.name: digest(p) for p in folder.iterdir() if p.is_file()})
    (folder / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('protocol', type=Path)
    p.add_argument('folder', type=Path)
    p.add_argument('--source', type=Path, default=ROOT / 'src/duplex.py')
    args = p.parse_args()
    import torch
    torch.set_num_threads(1)
    report = run(args.protocol, args.folder, args.source)
    raise SystemExit(0 if report['accepted'] else 1)
