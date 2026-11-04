"""Acoustic floor replay through the real VAD/AEC frontend.

This is a stress test, not a room acceptance study. It mixes a public generated
reply as the far-end with a second generated reply as the near-end, adds a
delayed echo, and drives DuplexFrontend + FloorPolicy in 20 ms steps. No model,
microphone or speaker is opened.
"""
import argparse
import json
import pathlib
import sys
import wave

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from duplex import DuplexFrontend
from floor import HARD, REVERSIBLE, FloorPolicy

STEP = .02
FAR_RATE = 24000


def read_wav(path):
    with wave.open(str(path)) as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError('Expected mono PCM16')
        rate = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2')
    return data.astype(np.float32) / 32768.0, rate


def resample(audio, rate, target=16000):
    if rate == target:
        return audio.astype(np.float32)
    positions = np.arange(round(len(audio) * target / rate)) * rate / target
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)


def speech_slice(audio, rate, duration, start=.25):
    count = round(duration * rate)
    begin = round(start * rate)
    if begin + count > len(audio):
        begin = max(0, len(audio) - count)
    return audio[begin:begin + count].copy()


def human_near():
    """A public human LibriSpeech clip, used as the near-end speaker."""
    import io
    import pandas as pd
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    path = hf_hub_download('hf-internal-testing/librispeech_asr_dummy',
                           'clean/validation-00000-of-00001.parquet',
                           repo_type='dataset')
    frame = pd.read_parquet(path)
    audio, rate = sf.read(io.BytesIO(frame.iloc[0]['audio']['bytes']),
                          dtype='float32')
    # Start at the first clear speech energy rather than a fixed offset.
    window = max(1, round(.02 * rate))
    for begin in range(0, len(audio) - window, window):
        if float(np.sqrt(np.mean(audio[begin:begin + window] ** 2))) > .02:
            return audio[begin:].copy(), rate
    return audio, rate


def build_case(kind, far, near):
    duration = 4.6
    samples = round(duration * 16000)
    mic = np.zeros(samples, np.float32)
    far_track = np.zeros(samples, np.float32)
    near_start = 1.5
    # Far-end playback from 0.5 s to 3.8 s, using the generated reply.
    far_begin = round(.5 * 16000)
    far_len = min(len(far), round(3.3 * 16000))
    far_track[far_begin:far_begin + far_len] = far[:far_len]
    # Delayed, attenuated echo of the far track.
    delay = round(.08 * 16000)
    echo = np.zeros(samples, np.float32)
    echo[delay:] = far_track[:-delay] * .45
    near_track = np.zeros(samples, np.float32)
    if kind in ('backchannel', 'floor_take', 'mid_pause'):
        length = {'backchannel': .22, 'floor_take': .9, 'mid_pause': .4}[kind]
        begin = round(near_start * 16000)
        piece = near[:round(length * 16000)]
        near_track[begin:begin + len(piece)] = piece
        if kind == 'mid_pause':
            second = round((near_start + .4 + .5) * 16000)
            piece2 = near[round(.5 * 16000):round(.5 * 16000) + round(.4 * 16000)]
            near_track[second:second + len(piece2)] = piece2
    mic = far_track + echo + near_track
    # Deterministic low-level room noise.
    rng = np.random.default_rng(20261028)
    mic += rng.normal(0, .0015, samples).astype(np.float32)
    return mic, far_track, near_track


def run_case(kind, mode, far, near):
    front = DuplexFrontend(aec=True, holdoff_s=0)
    front.set_delay(80)
    policy = FloorPolicy(mode=mode)
    mic, far_track, near_track = build_case(kind, far, near)
    actions = []
    started = False
    cancel_at = None
    for offset in range(0, len(mic), round(STEP * 16000)):
        block = mic[offset:offset + round(STEP * 16000)]
        far_block = far_track[offset:offset + len(block)]
        front.push_far(far_block, rate=16000)
        speaking, onset, ended, clean = front.update(block)
        if onset:
            started = True
        action = policy.update(offset / 16000, speaking=front.hot,
                               started=onset, ended=ended,
                               block_s=len(clean) / 16000,
                               onset_ms=front.onset_speech_ms,
                               evidence_ms=front.speech_evidence_ms,
                               assistant_active=True, audible=True)
        if action.duck or action.resume or action.cancel:
            actions.append(dict(kind='duck' if action.duck else
                                'resume' if action.resume else 'commit',
                                at_ms=round(offset / 16, 1),
                                speech_ms=round(action.speech_ms, 1),
                                reason=action.reason))
        if action.cancel and cancel_at is None:
            cancel_at = offset / 16
    return dict(id=kind, mode=mode, started=started,
                ducked=any(a['kind'] == 'duck' for a in actions),
                resumed=any(a['kind'] == 'resume' for a in actions),
                cancelled=any(a['kind'] == 'commit' for a in actions),
                cancel_at_ms=round(cancel_at, 1) if cancel_at is not None else None,
                near_energy=round(float(np.sqrt(np.mean(near_track ** 2))), 5),
                actions=actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(ROOT / 'outputs' / 'floor_acoustic.json'))
    args = parser.parse_args()
    far, far_rate = read_wav(ROOT / 'research/examples/pass-reply.wav')
    far = resample(far, far_rate)
    try:
        near, near_rate = human_near()
        near_source = 'librispeech-dummy'
    except Exception:
        near, near_rate = read_wav(ROOT / 'research/examples/failure-reply.wav')
        near_source = 'generated-failure-reply'
    near = resample(near, near_rate)
    # Normalize the near-end to a conversational level so the VAD is not the
    # limiting factor; the AEC/double-talk interaction is the thing under test.
    near_rms = float(np.sqrt(np.mean(near ** 2)))
    if near_rms:
        near = near * (.06 / near_rms)
    far_rms = float(np.sqrt(np.mean(far ** 2)))
    if far_rms:
        far = far * (.05 / far_rms)
    cases = ['echo_only', 'backchannel', 'floor_take', 'mid_pause']
    rows = []
    for mode in (HARD, REVERSIBLE):
        for kind in cases:
            rows.append(run_case(kind, mode, far, near))
    expected_cancel = {'echo_only': False, 'backchannel': False,
                       'floor_take': True, 'mid_pause': True}
    for row in rows:
        row['expected_cancel'] = expected_cancel[row['id']]
        row['false_cancel'] = bool(row['cancelled'] and not row['expected_cancel'])
        row['missed'] = bool(row['expected_cancel'] and not row['cancelled'])
    summary = {}
    for mode in (HARD, REVERSIBLE):
        subset = [row for row in rows if row['mode'] == mode]
        summary[mode] = dict(
            cases=len(subset),
            started=sum(row['started'] for row in subset),
            ducked=sum(row['ducked'] for row in subset),
            resumed=sum(row['resumed'] for row in subset),
            cancelled=sum(row['cancelled'] for row in subset),
            false_cancels=sum(row['false_cancel'] for row in subset),
            missed=sum(row['missed'] for row in subset),
        )
    report = dict(
        schema_version=1,
        evidence_class='offline acoustic stress replay; public human near-end + generated far-end + synthetic echo',
        near_source=near_source,
        step_ms=20,
        cases=rows,
        summary=summary,
        limits=('One speaker/model voice pair, synthetic echo, no real room, '
                'microphone, speaker or live model. Mid-pause is a known EOT '
                'limitation. Not a room acceptance test.'),
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8',
                   newline='\n')
    print(json.dumps(dict(out=str(out), summary=summary), indent=2))


if __name__ == '__main__':
    main()
