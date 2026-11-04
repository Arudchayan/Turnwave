"""LFM3 client for llama-server's realtime WebSocket and streaming TTS APIs."""
import base64
import io
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
import wave

import numpy as np
import websocket

from client import Turn


class LFM3Client:
    supports_context_recovery = True

    def __init__(self, base='http://127.0.0.1:8080', timeout=60):
        self.base = base.rstrip('/').replace('http://localhost:', 'http://127.0.0.1:')
        self.timeout = timeout
        self.request_lock = threading.Lock()
        self._chat = None
        self._out_rate = 24000
        self.voice_params = {
            'temperature': 0.7,
            'top_k': 40,
            'top_p': 0.9,
            'min_p': 0.05,
            'repeat_penalty': 1.05,
            'voice': os.getenv('LOCALVOICE_VOICE', '').strip(),
        }
        self._custom_voice_params = False

    def set_voice_params(self, **params):
        for key in ('temperature', 'top_k', 'top_p', 'min_p', 'repeat_penalty', 'voice'):
            if key in params and params[key] is not None:
                self.voice_params[key] = params[key]
        self._custom_voice_params = True
        if self.request_lock.acquire(timeout=.1):
            try:
                self._close_chat()
            finally:
                self.request_lock.release()
        return dict(self.voice_params)

    def health(self):
        try:
            with urllib.request.urlopen(self.base + '/health', timeout=1) as response:
                return response.status == 200
        except (OSError, ValueError):
            return False

    def _close_chat(self):
        if self._chat is not None:
            try:
                self._chat.close()
            except (OSError, websocket.WebSocketException):
                pass
            finally:
                self._chat = None

    def close(self):
        if self.request_lock.acquire(timeout=.3):
            try:
                self._close_chat()
            finally:
                self.request_lock.release()

    def _receive(self, sock, turn):
        deadline = time.monotonic() + self.timeout
        while not turn.cancel.is_set():
            try:
                event = json.loads(sock.recv())
                if event.get('type') == 'error':
                    raise RuntimeError(str((event.get('error') or {}).get('message', 'Voice server error')))
                return event
            except (websocket.WebSocketTimeoutException, TimeoutError):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Voice server timed out')
        return None

    def _connect(self, turn, transcription=False):
        url = self.base.replace('http://', 'ws://', 1).replace('https://', 'wss://', 1) + '/v1/realtime'
        sock = websocket.create_connection(
            url, timeout=8, enable_multithread=True,
            sock_opt=((socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),))
        try:
            sock.settimeout(.2)
            event = self._receive(sock, turn)
            if event is None:
                sock.close()
                return None, 24000
            session = event.get('session') or {}
            audio = session.get('audio') or {}
            input_rate = ((audio.get('input') or {}).get('format') or {}).get('rate', 16000)
            output_rate = ((audio.get('output') or {}).get('format') or {}).get('rate', 24000)
            if input_rate != 16000 or not 8000 <= output_rate <= 192000:
                raise ValueError('Unsupported voice server sample rate')
            settings = {'type': 'transcription'} if transcription else {
                'type': 'realtime', 'instructions': 'Respond in interleaved text and audio. Answer briefly and directly.',
                'output_modalities': ['text', 'audio']}
            if not transcription and self._custom_voice_params:
                settings['temperature'] = self.voice_params['temperature']
            sock.send(json.dumps({'type': 'session.update', 'session': settings}))
            event = self._receive(sock, turn)
            if event is None:
                sock.close()
                return None, output_rate
            if event.get('type') != 'session.updated':
                raise RuntimeError('Unexpected voice server session response')
            return sock, output_rate
        except Exception:
            sock.close()
            raise

    @staticmethod
    def _pcm(wav_bytes):
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                raise ValueError('Expected 16 kHz mono PCM16 audio')
            return wav.readframes(wav.getnframes())

    @staticmethod
    def _append(sock, pcm, turn):
        if turn.cancel.is_set() or not pcm:
            return
        sock.settimeout(5.0)
        try:
            for offset in range(0, len(pcm), 131072):
                if turn.cancel.is_set():
                    return
                sock.send(json.dumps({
                    'type': 'input_audio_buffer.append',
                    'audio': base64.b64encode(pcm[offset:offset + 131072]).decode('ascii'),
                }))
        finally:
            sock.settimeout(.2)

    def _acquire(self, turn):
        while not turn.cancel.is_set():
            if self.request_lock.acquire(timeout=.05):
                return True
        return False

    def chat_turn(self, wav_bytes, turn, first, max_tokens=512, on_audio=None, require_audio=True,
                  recovery_wav=None, recovery_reply='', _text_input=None):
        context = ''
        if first and recovery_wav and not turn.cancel.is_set():
            try:
                context = self.asr_text(recovery_wav, turn=turn).strip()[:1000]
            except Exception:
                # A failed optional recovery must not prevent a new voice turn.
                turn.error = None
            finally:
                turn.text_parts.clear()
                turn.done.clear()
        if turn.cancel.is_set():
            turn.done.set()
            return
        acquired = False
        try:
            acquired = self._acquire(turn)
            if not acquired or turn.cancel.is_set():
                return
            if first:
                self._close_chat()
            if self._chat is None:
                self._chat, self._out_rate = self._connect(turn)
            if self._chat is None or turn.cancel.is_set():
                return
            if context:
                draft = str(recovery_reply or '').strip()[:600]
                prior = ('Earlier the user said: ' + context +
                         '\nThe assistant reply was interrupted.')
                if draft:
                    prior += (' The assistant generated this draft, but some or all may not have been heard: ' + draft)
                prior += '\nContinue with the next user request.'
                self._chat.send(json.dumps({'type': 'conversation.item.create', 'item': {
                    'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': prior}]}}))
            if turn.request_observer:
                payload = {'protocol': 'realtime', 'reset_context': bool(first), 'max_tokens': max_tokens}
                if _text_input is None:
                    payload['wav_base64'] = base64.b64encode(wav_bytes).decode('ascii')
                else:
                    payload['text'] = _text_input
                turn.request_observer(payload)
            if turn.cancel.is_set():
                return
            turn.t_req = time.perf_counter()
            if _text_input is None:
                self._append(self._chat, self._pcm(wav_bytes), turn)
            else:
                self._chat.send(json.dumps({'type': 'conversation.item.create', 'item': {
                    'type': 'message', 'role': 'user',
                    'content': [{'type': 'input_text', 'text': _text_input}]}}))
            if turn.cancel.is_set():
                return
            self._chat.send(json.dumps({'type': 'response.create', 'response': {'max_output_tokens': max_tokens}}))
            while not turn.cancel.is_set():
                event = self._receive(self._chat, turn)
                if event is None:
                    break
                kind = event.get('type')
                elapsed = time.perf_counter() - turn.t_req
                if kind == 'response.output_text.delta':
                    if turn.first_text is None:
                        turn.first_text = elapsed
                    turn.text_parts.append(event.get('delta', ''))
                elif kind == 'response.output_audio.delta':
                    pcm = base64.b64decode(event.get('delta', ''), validate=True)
                    if len(pcm) % 2:
                        raise ValueError('Invalid PCM16 audio')
                    samples = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768
                    if self._out_rate != 24000 and len(samples):
                        positions = np.arange(round(len(samples) * 24000 / self._out_rate)) * self._out_rate / 24000
                        samples = np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)
                    if len(samples):
                        if turn.ttft is None:
                            turn.ttft = elapsed
                        turn.audio_frames += 1
                        turn.audio_samples += len(samples)
                        if on_audio:
                            on_audio(samples)
                elif kind == 'response.done':
                    response = event.get('response') or {}
                    status = response.get('status')
                    if status in ('failed', 'cancelled', 'incomplete'):
                        raise RuntimeError('Voice response did not complete: ' + status)
                    turn.finish_reason = 'stop'
                    break
            if require_audio and not turn.cancel.is_set() and not turn.audio_samples:
                raise RuntimeError('The voice server returned no audio.')
        except Exception as exc:
            if not turn.cancel.is_set():
                turn.error = str(exc)
            if acquired:
                self._close_chat()
        finally:
            if acquired:
                if turn.cancel.is_set():
                    self._close_chat()
                self.request_lock.release()
            turn.done.set()

    def chat_text(self, text, turn, first, max_tokens=512, on_audio=None, require_audio=True,
                  recovery_wav=None, recovery_reply=''):
        return self.chat_turn(b'', turn, first, max_tokens, on_audio, require_audio,
                              recovery_wav, recovery_reply, _text_input=text)

    def asr_text(self, wav_bytes, max_tokens=128, turn=None):
        turn = turn if turn is not None else Turn()
        acquired = False
        sock = None
        try:
            acquired = self._acquire(turn)
            if not acquired or turn.cancel.is_set():
                return ''
            self._close_chat()
            sock, _ = self._connect(turn, transcription=True)
            if sock is None:
                return ''
            self._append(sock, self._pcm(wav_bytes), turn)
            if not turn.cancel.is_set():
                sock.send(json.dumps({'type': 'input_audio_buffer.commit', 'final': True}))
            while not turn.cancel.is_set():
                event = self._receive(sock, turn)
                if event is None:
                    break
                if event.get('type') == 'conversation.item.input_audio_transcription.completed':
                    turn.text_parts.append(event.get('transcript', ''))
                    break
        except Exception as exc:
            if not turn.cancel.is_set():
                turn.error = str(exc)
        finally:
            if sock:
                try:
                    sock.close()
                except (OSError, websocket.WebSocketException):
                    pass
            if acquired:
                self.request_lock.release()
            turn.done.set()
        if turn.error:
            raise RuntimeError(turn.error)
        return turn.text.strip()

    def tts(self, text, turn, max_tokens=128, on_audio=None):
        acquired = False
        response = None
        try:
            acquired = self._acquire(turn)
            if not acquired or turn.cancel.is_set():
                return
            self._close_chat()
            payload = {'input': text, 'response_format': 'pcm', 'stream': True, 'n_predict': max_tokens}
            voice = (self.voice_params.get('voice') or os.getenv('LOCALVOICE_VOICE', '')).strip()
            speaker_ref = os.getenv('LOCALVOICE_SPEAKER_REF', '').strip()
            if voice:
                payload['voice'] = voice
            if speaker_ref:
                payload['speaker_ref'] = speaker_ref
            if self._custom_voice_params:
                payload['temperature'] = self.voice_params['temperature']
                payload['top_k'] = self.voice_params['top_k']
                payload['top_p'] = self.voice_params['top_p']
                payload['min_p'] = self.voice_params['min_p']
                payload['repeat_penalty'] = self.voice_params['repeat_penalty']
            if turn.request_observer:
                turn.request_observer(payload)
            if turn.cancel.is_set():
                return
            req = urllib.request.Request(self.base + '/tts', data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
            turn.t_req = time.perf_counter()
            response = urllib.request.urlopen(req, timeout=self.timeout)
            with turn._lock:
                turn._response = response
            content_type = response.headers.get('Content-Type', '').lower().replace(' ', '')
            if not all(part in content_type for part in ('audio/pcm', 'rate=24000', 'encoding=float', 'bits=32')):
                raise ValueError('Unsupported speech synthesis audio format')
            pending = b''
            read = getattr(response, 'read1', None) or response.read
            while not turn.cancel.is_set():
                data = read(8192)
                if not data:
                    break
                pending += data
                count = len(pending) // 4 * 4
                if not count:
                    continue
                samples = np.frombuffer(pending[:count], dtype='<f4').copy()
                pending = pending[count:]
                if not np.isfinite(samples).all():
                    raise ValueError('Non-finite generated audio')
                samples = np.clip(samples, -1, 1)
                if turn.ttft is None:
                    turn.ttft = time.perf_counter() - turn.t_req
                turn.audio_frames += 1
                turn.audio_samples += len(samples)
                if on_audio:
                    on_audio(samples)
            if pending and not turn.cancel.is_set():
                raise ValueError('Truncated generated audio')
        except Exception as exc:
            if not turn.cancel.is_set():
                turn.error = str(exc)
        finally:
            with turn._lock:
                turn._response = None
            try:
                if response is not None:
                    response.close()
            except Exception:
                pass
            finally:
                if acquired:
                    self.request_lock.release()
                turn.done.set()
