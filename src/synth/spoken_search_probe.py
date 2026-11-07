"""Exercise a synthetic speech-to-search-to-speech pipeline.

Uses the real LFM3 voice backend and live Bing RSS search, with a deterministic
local stand-in for the cloud model. No microphone or speaker is opened. The
stand-in deliberately requests search; this is not a model, human listening, or
real-device evaluation. Run only with the app stopped and an LFM3 server ready.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import re
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from agent import AgentClient
from app import Assistant
from backend_session import backend_session
from client import Turn, wav16k_bytes
from lfm3_client import LFM3Client

PROMPT = 'Search the web for Python downloads on python dot org.'
QUERY = 'Python downloads site:python.org'
ANSWER = 'I found Python downloads on the official Python website.'


class StubModel(BaseHTTPRequestHandler):
    """Deterministic tool use; records only synthetic text and request shapes."""
    def log_message(self, *args):
        pass

    def do_POST(self):
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
            messages = payload['messages']
            if messages[-1]['role'] == 'user':
                question = messages[-1]['content']
                self.server.synthetic_input = question
                message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                    'id': 'synthetic-search', 'type': 'function', 'function': {
                        'name': 'web_search', 'arguments': json.dumps({'query': QUERY})}}]}
            else:
                self.server.tool_result = json.loads(messages[-1]['content'])
                result = self.server.tool_result
                items = result.get('results', []) if isinstance(result, dict) else result
                official = any(isinstance(item, dict) and
                               item.get('url', '').startswith(('https://www.python.org/', 'https://docs.python.org/'))
                               for item in items or [])
                message = {'role': 'assistant', 'content': ANSWER if official else
                           'I could not find an official Python download source.'}
            data = json.dumps({'choices': [{'message': message}]}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (KeyError, ValueError, TypeError):
            self.send_error(400)


class SilentMic:
    device = -1
    def __init__(self): self.q = queue.Queue()
    def stop(self): pass
    def drain(self): pass


class SilentFront:
    level = 0.0
    def push_far(self, *args, **kwargs): pass
    def clear_playback(self): pass
    def reset(self): pass


class CapturingPlayer:
    """Collect output samples, without opening a system audio device."""
    def __init__(self): self.playing = False; self.samples = []
    def set_gain(self, *args): pass
    def stop(self): self.playing = False
    def play_chunks(self, chunks, cancel, on_start, on_play, on_written=None):
        started = False
        try:
            for chunk in chunks:
                if cancel.is_set():
                    return
                if not started:
                    self.playing = True
                    started = True
                    on_start()
                piece = np.asarray(chunk, np.float32).copy()
                self.samples.append(piece)
                on_play(piece)
                if on_written:
                    on_written(piece)
        finally:
            self.playing = False


def resample_24_to_16(audio):
    positions = np.arange(round(len(audio) * 2 / 3)) * 1.5
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)


def words(text):
    return set(re.findall(r'[a-z0-9]+', str(text).lower()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', 0), StubModel)
    server.synthetic_input, server.tool_result = None, None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with backend_session(args.url, test=True):
            client = LFM3Client(args.url)
            app = None
            try:
                if not client.health():
                    raise RuntimeError('Start the local LFM3 server first')
                pieces, fixture = [], Turn()
                client.tts(PROMPT, fixture, max_tokens=512, on_audio=pieces.append)
                if fixture.error or not pieces:
                    raise RuntimeError('Synthetic prompt generation failed: ' + str(fixture.error))
                input_audio = resample_24_to_16(np.concatenate(pieces))
                speaker = CapturingPlayer()
                agent = AgentClient(None, provider='custom', search_provider='bing',
                                    custom_endpoint=f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                                    local_tools=False, timeout=30)
                app = Assistant(mic=SilentMic(), player=speaker, front=SilentFront(), client=client, agent=agent)
                start = time.perf_counter()
                job = app.submit_audio(input_audio)
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    if job.playback_closed.is_set() and job.generation_closed.is_set() and app._active is None:
                        break
                    time.sleep(.02)
                else:
                    app.interrupt('probe timeout', announce=False)
                    raise TimeoutError('Synthetic spoken search did not finish')
                elapsed = time.perf_counter() - start
                state = app.state()
                reply = next((m for m in state['messages'] if m['id'] == job.reply_id), {})
                output_audio = np.concatenate(speaker.samples) if speaker.samples else np.zeros(0, np.float32)
                if len(output_audio):
                    heard = client.asr_text(wav16k_bytes(resample_24_to_16(output_audio)))
                else:
                    heard = ''
                tool_result = server.tool_result
                result_list = tool_result.get('results', []) if isinstance(tool_result, dict) else tool_result or []
                checks = {
                    'input_recognized': {'python', 'downloads'} <= words(server.synthetic_input),
                    'live_search_returned_sources': bool(result_list),
                    'reply_links_attached': bool(reply.get('sources')),
                    'audio_generated': len(output_audio) > 0 and not job.turn.error,
                    'answer_transcribed': {'python', 'downloads'} <= words(heard),
                }
                report = dict(scope='Synthetic speech, live Bing, deterministic local model stub, no microphone or speaker; output ASR uses the same model family.',
                              prompt=PROMPT, recognized=server.synthetic_input,
                              tool_result_count=len(result_list), sources=reply.get('sources', []),
                              reply=job.turn.text, output_transcript=heard, error=job.turn.error,
                              output_audio_s=round(len(output_audio)/24000, 3), elapsed_s=round(elapsed, 3),
                              checks=checks)
                Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
                print(json.dumps(report, indent=2), flush=True)
                if not all(checks.values()):
                    raise SystemExit(1)
            finally:
                if app is not None:
                    app.close()
                else:
                    client.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == '__main__':
    main()
