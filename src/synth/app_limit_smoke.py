"""Real GPU reply cap -> labelled app transcript -> context-preserving follow-up."""
import argparse
import json
from pathlib import Path
import sys
import time
import wave
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from app import Assistant
from client import ServerClient
from diagnostics import Diagnostics
from backend_session import backend_session
from synth.app_smoke import SyntheticMic, PacedSpeaker, wav
from synth.checkpoint_experiment import checkpoint
from synth.archive_experiment import capture_files


def run(folder):
    folder.mkdir(parents=True, exist_ok=False)
    capture_files(folder / 'source')
    class LimitedClient(ServerClient):
        def __init__(self):
            super().__init__(); self.first_flags = []
        def chat_turn(self, audio, turn, first, max_tokens=512, on_audio=None):
            self.first_flags.append(first)
            index = len(self.first_flags)
            (folder / f'{index}-input.wav').write_bytes(audio)
            chunks = []
            def receive(samples):
                chunks.append(samples.copy())
                if on_audio is not None: on_audio(samples)
            try:
                return super().chat_turn(audio, turn, first,
                    max_tokens=24 if index == 1 else 512, on_audio=receive)
            finally:
                if chunks:
                    with wave.open(str(folder / f'{index}-reply.wav'), 'wb') as writer:
                        writer.setnchannels(1); writer.setsampwidth(2); writer.setframerate(24000)
                        writer.writeframes((np.clip(np.concatenate(chunks), -1, 1)*32767).astype('<i2').tobytes())
    client = LimitedClient()
    assistant = Assistant(mic=SyntheticMic(), player=PacedSpeaker(), client=client,
                          diagnostics=Diagnostics(folder / 'events.jsonl'))
    rows = []
    try:
        for name in ['user_greet.wav', 'quality/banana.wav']:
            job = assistant.submit_audio(wav(name))
            deadline = time.monotonic() + 30
            while assistant._active is not None and time.monotonic() < deadline:
                time.sleep(.02)
            if assistant._active is not None: raise TimeoutError('Application did not finish its reply')
            rows.append(dict(input=name, text=job.turn.text, error=job.turn.error,
                             finish_reason=job.turn.finish_reason, audio_samples=job.turn.audio_samples,
                             transcript=assistant.state()['transcript'][-1]))
        passed = (rows[0]['finish_reason'] == 'length' and rows[0]['audio_samples'] > 0
                  and rows[0]['transcript'].endswith(' [reply reached length limit]')
                  and rows[1]['finish_reason'] == 'stop'
                  and rows[1]['text'].strip().lower().strip('.!') == 'banana'
                  and client.first_flags == [True, False] and not assistant.error)
        report = dict(passed=bool(passed), cases=rows, context_reset_flags=client.first_flags,
                      scope='Real GPU + Assistant + paced simulated speaker; no physical audio capture/playback. No broad memory claim.')
        (folder / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return passed
    finally:
        assistant.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    if args.folder.exists(): parser.error('Use a new experiment folder')
    try:
        with backend_session(test=True): passed = run(args.folder)
    except Exception:
        if args.folder.exists():
            import traceback
            (args.folder / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if args.folder.exists(): print('Verified backup:', checkpoint(args.folder), flush=True)
    raise SystemExit(0 if passed else 1)
