"""Compare pause thresholds through the real Silero frontend, without devices.

Inserts a 650 ms pause near a low-energy point in each clean read-speech clip.
This synthetic pause screen measures detected boundaries, not ASR accuracy or
real spontaneous conversation. Results contain counts/timings, not audio/text.
"""
import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from duplex import DuplexFrontend
from lfm3_realtime_asr_bench import DATASET, REVISION, DATA_FILE


def pause_position(audio):
    """Pick the quietest 20 ms block in the middle 30% of the clip."""
    points = np.arange(int(len(audio) * .35), int(len(audio) * .65), 320)
    if not len(points):
        raise ValueError('Clip is too short')
    return int(min(points, key=lambda p: float(np.mean(audio[p:p + 320] ** 2))))


def evaluate(audio, threshold):
    split = pause_position(audio)
    audio = np.concatenate((np.zeros(8000, np.float32), audio[:split],
                            np.zeros(10400, np.float32), audio[split:],
                            np.zeros(24000, np.float32)))
    pause_end = 8000 + split + 10400
    audio_end = len(audio) - 24000
    front = DuplexFrontend(aec=False, holdoff_s=0, end_silence_ms=threshold)
    front.warmup()
    starts, ends = [], []
    for offset in range(0, len(audio), 320):
        _, started, ended, _ = front.update(audio[offset:offset + 320])
        if started:
            starts.append(offset + 320)
        if ended:
            ends.append(offset + 320)
            front.reset()
    return dict(starts=len(starts), ends=len(ends),
                premature=any(end < pause_end for end in ends),
                final_delay_ms=round((ends[-1] - audio_end) / 16, 1) if ends else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=20)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose a fresh output path')
    if not 1 <= args.limit <= 20:
        parser.error('Limit must be between 1 and 20')
    import pandas as pd
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    dataset = hf_hub_download(DATASET, DATA_FILE, repo_type='dataset', revision=REVISION)
    frame = pd.read_parquet(dataset).head(args.limit)
    rows = []
    for _, row in frame.iterrows():
        audio, rate = sf.read(io.BytesIO(row['audio']['bytes']), dtype='float32')
        if rate != 16000 or audio.ndim != 1:
            raise ValueError('Expected mono 16 kHz fixture')
        rows.append({'id': row['id'], 'quick': evaluate(audio, 480),
                     'balanced': evaluate(audio, 800)})
        print(f'{len(rows)}/{args.limit}', flush=True)
    report = dict(dataset=DATASET, revision=REVISION, pause_ms=650, rows=rows,
                  scope='Synthetic inserted pause; Silero frontend only, no devices or ASR.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2)


if __name__ == '__main__':
    main()
