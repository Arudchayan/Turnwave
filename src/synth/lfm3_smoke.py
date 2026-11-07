"""One-shot smoke test for the LFM3 realtime transport.

Requires a running llama.cpp b11960+ `llama-server` with LFM3-Audio-2B and its
mmproj. It sends one WAV, streams the reply audio, and writes a summary. This is
a transport/quality check, not a benchmark.

    python -m src.synth.lfm3_smoke --wav research/examples/pass-reply.wav \
        --url ws://127.0.0.1:8080/v1/realtime --out outputs/lfm3-reply.wav
"""
import argparse
import json
import pathlib
import sys
import wave

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from client import Turn
from realtime_client import RealtimeClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='ws://127.0.0.1:8080/v1/realtime')
    parser.add_argument('--wav', default=str(ROOT / 'research/examples/pass-reply.wav'))
    parser.add_argument('--out', default=None)
    parser.add_argument('--max-tokens', type=int, default=256)
    args = parser.parse_args()

    client = RealtimeClient(args.url, max_output_tokens=args.max_tokens)
    turn = Turn()
    audio = []
    client.chat_turn(pathlib.Path(args.wav).read_bytes(), turn, first=True,
                     on_audio=audio.append)
    client.close()
    if audio and args.out:
        samples = np.concatenate(audio)
        pcm = (np.clip(samples, -1, 1) * 32767).astype('<i2')
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(pcm.tobytes())
    print(json.dumps(dict(
        text=turn.text,
        first_audio_s=round(turn.ttft, 3) if turn.ttft is not None else None,
        audio_s=round(turn.audio_samples / 24000, 3),
        finish_reason=turn.finish_reason,
        error=turn.error,
        out=args.out if audio else None,
    ), indent=2))


if __name__ == '__main__':
    main()
