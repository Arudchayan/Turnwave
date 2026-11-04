"""Run one synthetic speech/search check with a live provider, without devices.

This makes external model and search requests and may consume provider quota.
It tests the application path, not microphone capture or real-world playback.
"""
import argparse
import json
import os
from pathlib import Path
import queue
import sys
import time
from urllib.parse import urlsplit

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from agent import AgentClient
from app import Assistant
from backend_session import backend_session
from client import Turn
from lfm3_client import LFM3Client

PROMPT = 'Search the web for Python downloads on python dot org.'


class SilentMic:
    device = -1
    def __init__(self): self.q = queue.Queue()
    def stop(self): pass
    def drain(self):
        while not self.q.empty(): self.q.get_nowait()


class SilentFront:
    level = 0.0
    def push_far(self, *args, **kwargs): pass
    def clear_playback(self): pass
    def reset(self): pass


class CapturingPlayer:
    def __init__(self): self.playing = False
    def set_gain(self, *args): pass
    def stop(self): self.playing = False
    def play_chunks(self, chunks, cancel, on_start, on_play, on_written=None):
        try:
            for audio in chunks:
                if cancel.is_set(): break
                if not self.playing:
                    self.playing = True
                    on_start()
                on_play(audio)
                if on_written: on_written(audio)
        finally:
            self.playing = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18085')
    parser.add_argument('--model', required=True, help='OpenCode Go model to call')
    parser.add_argument('--out', required=True, help='Path for synthetic result JSON')
    args = parser.parse_args()
    key = os.getenv('OPENCODE_GO_API_KEY')
    if not key:
        parser.error('Set OPENCODE_GO_API_KEY in the process environment')
    # Refuse to reset an active app's model context.
    with backend_session(args.url, test=True):
        voice = LFM3Client(args.url, timeout=180)
        agent = AgentClient(key, provider='go', model=args.model, search_provider='bing',
                            local_tools=False, timeout=30)
        app = None
        start = time.monotonic()
        try:
            chunks, fixture = [], Turn()
            voice.tts(PROMPT, fixture, max_tokens=512, on_audio=chunks.append)
            if fixture.error or not chunks:
                raise RuntimeError('Synthetic input generation failed: ' + str(fixture.error))
            audio = resample_poly(np.concatenate(chunks), 2, 3).astype(np.float32)
            app = Assistant(mic=SilentMic(), front=SilentFront(), player=CapturingPlayer(),
                            client=voice, agent=agent)
            job = app.submit_audio(audio)
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if job.playback_closed.is_set() and job.generation_closed.is_set() and app._active is None:
                    break
                time.sleep(.1)
            else:
                app.interrupt('benchmark timeout', announce=False)
                raise TimeoutError('Synthetic spoken search did not finish')
            state = app.state()
            trace = agent.last_trace or {}
            report = {'scope': 'synthetic input and fake playback; live OpenCode Go and Bing',
                      'model': args.model, 'prompt': PROMPT,
                      'transcript': state['transcript'], 'error': state['error'],
                      'warning': state['warning'], 'audio_samples': job.played_samples,
                      'source_urls': [item.get('url') for item in state['agent_sources']],
                      'tools': [item.get('tool') for item in trace.get('tool_calls', [])],
                      'elapsed_seconds_including_input_synthesis': round(time.monotonic() - start, 2)}
            report['official_source'] = any((urlsplit(url or '').hostname or '').lower() in ('python.org', 'www.python.org', 'docs.python.org')
                                        for url in report['source_urls'])
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(report))
            if report['error'] or not report['audio_samples'] or not report['official_source'] or 'web_search' not in report['tools']:
                raise SystemExit('The synthetic spoken search did not complete all checks')
        finally:
            if app is not None: app.close()
            else: voice.close()


if __name__ == '__main__':
    main()
