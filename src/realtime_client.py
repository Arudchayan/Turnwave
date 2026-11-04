"""Optional client for llama.cpp's /v1/realtime WebSocket.

LFM3-Audio-2B is served by llama.cpp b11960+ and streams interleaved text and
24 kHz PCM audio over the OpenAI-compatible realtime endpoint. The existing
SSE client remains the default for the pinned LFM2.5 runner.

The client is deliberately turn-oriented: the app sends a completed user WAV,
commits it, and streams the response. A persistent socket keeps conversation
context between turns. `first=True` opens a fresh socket, which is how the app
expresses a context reset after an interruption.
"""
import base64
import io
import json
import os
import socket
import threading
import time
import urllib.parse
import wave

import numpy as np

try:
    import websocket
except ImportError:  # Kept optional so the HTTP path has no new hard dependency.
    websocket = None

from client import CHAT_SYS, TTS_VOICE, Turn


class RealtimeClient:
    def __init__(self, url=None, timeout=120, max_output_tokens=512):
        if websocket is None:
            raise RuntimeError('Realtime transport needs websocket-client')
        self.url = url or os.getenv('LOCALVOICE_REALTIME',
                                   'ws://127.0.0.1:8080/v1/realtime')
        self.base = self.url
        self.timeout = float(timeout)
        self.max_output_tokens = int(max_output_tokens)
        self.request_lock = threading.Lock()
        self._lock = threading.Lock()
        self._ws = None

    # -- transport -------------------------------------------------------
    def _connect(self, instructions, modalities, max_tokens):
        ws = websocket.create_connection(self.url, timeout=self.timeout)
        try:
            created = json.loads(ws.recv())
            if created.get('type') == 'error':
                raise RuntimeError(self._error_text(created))
            ws.send(json.dumps({'type': 'session.update', 'session': {
                'instructions': instructions,
                'output_modalities': list(modalities),
                'max_output_tokens': int(max_tokens),
            }}))
            while True:
                event = json.loads(ws.recv())
                if event.get('type') == 'error':
                    raise RuntimeError(self._error_text(event))
                if event.get('type') == 'session.updated':
                    break
            ws.settimeout(.2)
            return ws
        except Exception:
            try:
                ws.close()
            except Exception:
                pass
            raise

    def _ensure(self, first):
        with self._lock:
            if first and self._ws is not None:
                self._close_locked()
            if self._ws is None:
                self._ws = self._connect(CHAT_SYS, ('text', 'audio'),
                                         self.max_output_tokens)
            return self._ws

    def _close_locked(self):
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def close(self):
        with self._lock:
            self._close_locked()

    def health(self):
        try:
            parsed = urllib.parse.urlparse(self.url)
            host = parsed.hostname or '127.0.0.1'
            port = parsed.port or 8080
            with socket.create_connection((host, port), timeout=1):
                return True
        except (OSError, ValueError):
            return False

    # -- events ----------------------------------------------------------
    @staticmethod
    def _error_text(event):
        error = event.get('error') or {}
        return str(error.get('message') or error or 'The realtime server reported an error.')

    @staticmethod
    def _audio(delta):
        pcm = np.frombuffer(base64.b64decode(delta, validate=True),
                            dtype='<i2').astype(np.float32) / 32768.0
        if not np.isfinite(pcm).all():
            raise ValueError('Non-finite audio samples')
        return np.clip(pcm, -1, 1)

    def _send(self, ws, payload):
        ws.send(json.dumps(payload))

    def _stream(self, ws, turn, on_audio=None):
        if not turn.t_req:
            turn.t_req = time.perf_counter()
        while not turn.cancel.is_set():
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as exc:
                if turn.cancel.is_set():
                    return
                raise
            elapsed = time.perf_counter() - turn.t_req
            if turn.first_event is None:
                turn.first_event = elapsed
            event = json.loads(raw)
            kind = event.get('type')
            if kind == 'error':
                raise RuntimeError(self._error_text(event))
            if kind == 'response.created':
                turn.headers_s = elapsed
                continue
            if kind == 'response.output_text.delta':
                delta = event.get('delta') or ''
                if delta:
                    turn.text_parts.append(delta)
                    if turn.degenerate:
                        raise RuntimeError('The reply became repetitive. Please try again.')
                continue
            if kind == 'response.output_audio.delta':
                audio = self._audio(event.get('delta') or '')
                if turn.ttft is None:
                    turn.ttft = elapsed
                turn.audio_frames += 1
                turn.audio_samples += len(audio)
                if on_audio is not None and len(audio):
                    on_audio(audio)
                continue
            if kind == 'response.done':
                response = event.get('response') or {}
                turn.finish_reason = ('length' if response.get('status') == 'incomplete'
                                      else 'stop')
                return
        # Cancellation observed while waiting. Ask the server to stop if the
        # socket is still usable; the next turn reconnects with first=True.
        try:
            self._send(ws, {'type': 'response.cancel'})
        except Exception:
            pass

    def _audio_input(self, wav_bytes):
        with wave.open(io.BytesIO(wav_bytes)) as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError('Realtime input must be mono PCM16')
            rate = wav.getframerate()
            pcm = wav.readframes(wav.getnframes())
        if rate != 16000:
            samples = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.0
            positions = np.arange(round(len(samples) * 16000 / rate)) * rate / 16000
            samples = np.interp(positions, np.arange(len(samples)), samples)
            pcm = (np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes()
        return pcm

    def chat_turn(self, wav_bytes, turn, first, max_tokens=None,
                  on_audio=None, require_audio=True, _attempt=1):
        pcm = self._audio_input(wav_bytes)
        with self.request_lock:
            ws = None
            try:
                if turn.request_observer is not None:
                    turn.request_observer(dict(
                        transport='realtime', url=self.url,
                        audio_bytes=len(wav_bytes),
                        max_tokens=int(max_tokens or self.max_output_tokens)))
                turn.t_req = time.perf_counter()
                ws = self._ensure(first)
                for offset in range(0, len(pcm), 6400):
                    if turn.cancel.is_set():
                        return
                    chunk = pcm[offset:offset + 6400]
                    self._send(ws, {'type': 'input_audio_buffer.append',
                                    'audio': base64.b64encode(chunk).decode()})
                if turn.cancel.is_set():
                    return
                self._send(ws, {'type': 'input_audio_buffer.commit'})
                self._send(ws, {'type': 'response.create', 'response': {
                    'output_modalities': ['text', 'audio'],
                    'max_output_tokens': int(max_tokens or self.max_output_tokens)}})
                self._stream(ws, turn, on_audio=on_audio)
                if (require_audio and not turn.cancel.is_set() and not turn.error
                        and not turn.audio_samples):
                    turn.error = ('The reply reached its length limit before producing audio.'
                                  if turn.truncated else
                                  'The server returned no audio. Please try again.')
            except Exception as exc:
                if not turn.cancel.is_set():
                    turn.error = str(exc)
            finally:
                if turn.cancel.is_set():
                    # The server context now contains a partially generated
                    # response. Close it so the next first=True turn is clean.
                    self.close()
                turn.done.set()

    def tts(self, text, turn, max_tokens=128, on_audio=None):
        """One-shot speech synthesis; does not touch the conversation socket."""
        ws = None
        try:
            turn.t_req = time.perf_counter()
            ws = self._connect(TTS_VOICE, ('audio',), max_tokens)
            self._send(ws, {'type': 'conversation.item.create', 'item': {
                'type': 'message', 'role': 'user',
                'content': [{'type': 'input_text', 'text': text}]}})
            self._send(ws, {'type': 'response.create', 'response': {
                'output_modalities': ['audio'],
                'max_output_tokens': int(max_tokens)}})
            self._stream(ws, turn, on_audio=on_audio)
        except Exception as exc:
            turn.error = str(exc)
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
            turn.done.set()
