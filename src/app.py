"""LocalVoice: continuous listening, cancellable turns, and a local web UI."""
import collections
import json
import os
import queue
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from audio_io import Mic, Player, list_inputs, audio_thread
from client import ServerClient, Turn, wav16k_bytes
from duplex import DuplexFrontend
from diagnostics import Diagnostics, source_fingerprint
from ui import PAGE

UI_PORT = 8000


@dataclass
class Job:
    turn: Turn = field(default_factory=Turn)
    chunks: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=128))
    produced: threading.Event = field(default_factory=threading.Event)
    epoch: int = 0
    reply_id: int = 0
    user_id: int = 0
    first: bool = True
    wav: bytes = b''
    t0: float = field(default_factory=time.perf_counter)
    played_samples: int = 0
    test: bool = False
    id: int = 0
    first_play_at: float = 0.0
    archive: object = None
    generation_closed: threading.Event = field(default_factory=threading.Event)
    playback_closed: threading.Event = field(default_factory=threading.Event)


class Assistant:
    """No import-time devices or threads; dependencies can be simulated in tests."""
    def __init__(self, mic=None, player=None, front=None, client=None, asr_client=None, diagnostics=None, start_muted=False, pre_roll_ms=250, session_archive=None):
        if not 20 <= pre_roll_ms <= 2000:
            raise ValueError('Pre-roll must be between 20 and 2000 ms')
        self.pre_roll_samples = round(pre_roll_ms * 16)
        self.mic = mic if mic is not None else Mic()
        self.player = player if player is not None else Player()
        self.front = front if front is not None else DuplexFrontend()
        self.client = client if client is not None else ServerClient(os.getenv('LOCALVOICE_SERVER', 'http://localhost:8080'))
        self.asr_client = asr_client
        if asr_client is not None and asr_client.base.rstrip('/') == self.client.base.rstrip('/'):
            raise ValueError('Transcription needs a separate server context')
        self.diagnostics = diagnostics if diagnostics is not None else Diagnostics()
        self._job_id = 0
        self._lock = threading.RLock()
        self._play_lock = threading.Lock()
        self._stop = threading.Event()
        self._capture_reset = threading.Event()
        self._threads = []
        self._active = None
        self._pending_jobs = {}
        self._archive_pending = set()
        self._archive_faults = set()
        self.session_archive = session_archive
        self.archive_error = None
        self._epoch = 0
        self._next_id = 0
        self._messages = []
        self.transcript_revision = 0
        self.first = True
        self.mic_on = not start_muted
        self.status = 'starting'
        self.error = None
        self.warning = None
        self.heard_at = self.gen_start = self.first_audio_at = None
        self.interrupted_at = None
        self.interrupt_candidate = False
        self.last_ttft = self.last_audio_s = self.last_turn_s = None
        self.turns = 0
        self._latencies = collections.deque(maxlen=12)
        self._server_ok = False
        self._mics = []
        self._asr_q = queue.Queue(maxsize=8)
        self._device_requests = queue.Queue()

    def _spawn(self, target, *args):
        def managed():
            with audio_thread():
                target(*args)
        thread = threading.Thread(target=managed, daemon=True)
        with self._lock:
            self._threads = [t for t in self._threads if t.is_alive()]
            self._threads.append(thread)
            thread.start()
        return thread

    def start(self):
        if any(t.is_alive() for t in self._threads):
            return
        self._spawn(self._health_loop)
        if self.asr_client is not None:
            self._spawn(self._asr_worker)
        self._spawn(self._startup)

    def _startup(self):
        try:
            self.diagnostics.event('startup', source_files_sha256=source_fingerprint(), residual_floor=getattr(self.front, 'residual_floor', None))
            self.front.warmup()
            self.warning = getattr(self.front, 'aec_error', None)
            # Warmup finishes BEFORE listening can submit chat, so it cannot
            # reset an established conversation. No delayed background TTS.
            if self.client.health() and not self._stop.is_set():
                warm = Turn()
                with self._lock:
                    self._warmup = warm
                self.client.tts('OK', warm, max_tokens=64)
            while not self._stop.is_set():
                try:
                    self._apply_device_requests()
                    self.mic.start()
                    self.error = None
                    self.status = 'listening'
                    self.diagnostics.event('mic_ready', device=self.mic.device, native_rate=getattr(self.mic, 'native_rate', None), gain=getattr(self.mic, 'gain', None), latency_s=getattr(self.mic, 'latency_s', None))
                    self._loop()
                    break
                except Exception as exc:
                    self.diagnostics.event('mic_error', error=str(exc), device=self.mic.device)
                    self.error = 'Microphone unavailable: ' + str(exc)
                    self.status = 'error'
                    self._stop.wait(1)
        except Exception as exc:
            self.error = 'Audio startup failed: ' + str(exc)
            self.status = 'error'

    def close(self):
        self._stop.set()
        with self._lock:
            warm = getattr(self, '_warmup', None)
        if warm is not None:
            warm.abort()
        self.interrupt('shutdown')
        try:
            self.mic.stop()
        finally:
            for thread in list(self._threads):
                if thread is not threading.current_thread():
                    thread.join(timeout=1)
            self.diagnostics.event('shutdown')
            self.diagnostics.close()

    def _health_loop(self):
        while not self._stop.is_set():
            if self._active is None:
                self._server_ok = self.client.health()
            if not self._mics:
                self._mics = list_inputs()
            self.diagnostics.event('health', server_ok=self._server_ok, state=self.status, mic_drops=getattr(self.mic, 'dropped_blocks', 0), capture_errors=getattr(self.mic, 'capture_errors', 0), clipped_fraction=getattr(self.mic, 'clipped_fraction', 0), residual_rejections=getattr(self.front, 'residual_rejections', 0), level=round(self.front.level, 5))
            self._stop.wait(5)

    def _say(self, text):
        with self._lock:
            self._next_id += 1
            self._messages.append({'id': self._next_id, 'text': text})
            self._messages = self._messages[-40:]
            self.transcript_revision += 1
            return self._next_id

    def _replace(self, message_id, text):
        with self._lock:
            for message in self._messages:
                if message['id'] == message_id:
                    message['text'] = text
                    self.transcript_revision += 1
                    return True
            return False

    def _asr_worker(self):
        while not self._stop.is_set():
            try:
                message_id, wav = self._asr_q.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                text = self.asr_client.asr_text(wav)
                if text:
                    self._replace(message_id, 'user: ' + text)
            except Exception as exc:
                self.warning = 'Transcription unavailable: ' + str(exc)

    def _loop(self):
        pre = collections.deque()
        rec = []
        rec_samples = 0
        recording = False
        while not self._stop.is_set():
            self._apply_device_requests()
            try:
                block = self.mic.q.get(timeout=.05)
            except queue.Empty:
                continue
            if self._capture_reset.is_set():
                self._capture_reset.clear()
                pre.clear()
                rec = []
                rec_samples = 0
                recording = False
                self.front.reset()
                if self.status == 'recording':
                    self.status = ('speaking' if self.first_audio_at else 'thinking') if self._active else 'listening'
            if not self.mic_on:
                continue
            try:
                speaking, started, ended, clean = self.front.update(block)
                self.warning = getattr(self.front, 'aec_error', None)
            except Exception as exc:
                with self._lock:
                    # Reset/mute can invalidate this block while detection is
                    # running, just as they can invalidate a detected onset.
                    if not self.mic_on or self._capture_reset.is_set():
                        continue
                    self.error = 'Speech detection failed: ' + str(exc)
                    self.status = 'error'
                self._stop.wait(.1)
                continue
            if not len(clean):
                continue
            if not recording:
                pre.append(clean)
                # Retain speech before confirmation without delaying onset.
                while len(pre) > 1 and sum(map(len, pre)) > self.pre_roll_samples:
                    pre.popleft()
                if started:
                    with self._lock:
                        if not self.mic_on or self._capture_reset.is_set():
                            continue
                        self.diagnostics.event('speech_start', state=self.status, raw_rms=round(float(np.sqrt(np.mean(block ** 2))), 5), clean_rms=round(float(np.sqrt(np.mean(clean ** 2))), 5), vad_probability=getattr(self.front, 'probability', None), echo_score=getattr(self.front, 'echo_score', None))
                        if self._active is not None:
                            self.interrupt('barge-in')
                        recording = True
                        rec = list(pre)
                        rec_samples = sum(map(len, rec))
                        self.status = 'recording'
            else:
                rec.append(clean)
                rec_samples += len(clean)
            if recording and (ended or rec_samples >= 20 * 16000):
                audio = np.concatenate(rec)
                self.diagnostics.event('speech_end', seconds=round(len(audio) / 16000, 3), rms=round(float(np.sqrt(np.mean(audio ** 2))), 5), capped=not ended)
                recording = False
                rec = []
                rec_samples = 0
                pre.clear()
                self.front.reset()
                if len(audio) >= 3200 and float(np.sqrt(np.mean(audio ** 2))) >= .006:
                    # Mute/reset may arrive while the frontend processes this
                    # block. Do not submit a stale automatic turn after mute.
                    with self._lock:
                        if self.mic_on and not self._capture_reset.is_set():
                            self.submit_audio(audio)
                else:
                    self.status = 'listening'

    def submit_audio(self, audio):
        wav = wav16k_bytes(audio)
        with self._lock:
            self.interrupt('new turn', announce=False)
            self.error = None
            self._job_id += 1
            job = Job(epoch=self._epoch, first=self.first, wav=wav, id=self._job_id)
            self._pending_jobs[job.id] = job
            if self.session_archive is not None:
                self._archive_pending.add(job.id)
            self.diagnostics.event('turn_submit', turn_id=job.id, first=job.first, input_s=round(len(audio)/16000, 3))
            job.user_id = self._say('user: [%.1fs voice]' % (len(audio) / 16000))
            job.reply_id = self._say('')
            self._active = job
            self.status = 'thinking'
            self.heard_at = time.time()
            self.gen_start = self.first_audio_at = None
            self.interrupt_candidate = False
            self._spawn(self._run_job, job)
            return job

    def _archive_call(self, job, method, *args, **kwargs):
        if job.archive is None:
            return
        try:
            return getattr(job.archive, method)(*args, **kwargs)
        except Exception as exc:
            self._archive_faults.add(job.id)
            self.archive_error = 'Recording could not be saved: ' + str(exc)
            self.diagnostics.event('archive_error', turn_id=job.id, error=str(exc))

    def _close_job_half(self, job, generation=False):
        with self._lock:
            (job.generation_closed if generation else job.playback_closed).set()
            if job.generation_closed.is_set() and job.playback_closed.is_set():
                self._pending_jobs.pop(job.id, None)
                if job.id not in self._archive_faults:
                    self._archive_pending.discard(job.id)

    def _run_job(self, job):
        turn = job.turn
        if self.session_archive is not None:
            try:
                job.archive = self.session_archive.begin_turn(job.id, job.wav, job.first, test=job.test)
            except Exception as exc:
                self._archive_faults.add(job.id)
                self.archive_error = 'Recording could not be started: ' + str(exc)
                self.diagnostics.event('archive_error', turn_id=job.id, error=str(exc))
        turn.request_observer = lambda payload: self._archive_call(job, 'record_request', payload)
        def produce():
            try:
                with self._lock:
                    if self._active is job:
                        self.gen_start = time.time()
                def receive(audio):
                    self._archive_call(job, 'write_audio', audio)
                    # Backpressure instead of dropping speech. Cancellation
                    # wakes the producer even when the queue is full.
                    for offset in range(0, len(audio), 1440):
                        while not turn.cancel.is_set():
                            try:
                                job.chunks.put(audio[offset:offset + 1440], timeout=.05)
                                break
                            except queue.Full:
                                continue
                if job.test:
                    self.client.tts('Speaker check. One, two, three.', turn, on_audio=receive)
                else:
                    self.client.chat_turn(job.wav, turn, job.first, on_audio=receive)
            except Exception as exc:
                turn.error = str(exc)
            finally:
                self.diagnostics.event('generation_end', turn_id=job.id, cancelled=turn.cancel.is_set(), error=turn.error, finish_reason=turn.finish_reason, text_chars=len(turn.text), audio_s=round(turn.audio_samples/24000, 3), first_audio_s=turn.ttft, request_queue_s=turn.queue_s, headers_s=turn.headers_s)
                self._archive_call(job, 'finish_generation', turn)
                self._close_job_half(job, generation=True)
                job.produced.set()
        def chunks():
            while not turn.cancel.is_set():
                try:
                    yield job.chunks.get(timeout=.02)
                except queue.Empty:
                    if job.produced.is_set():
                        # The producer may have queued its last audio after
                        # get() timed out but before setting produced. Once
                        # produced is set no further chunks can arrive, so
                        # drain that boundary chunk before declaring EOF.
                        try:
                            final_audio = job.chunks.get_nowait()
                        except queue.Empty:
                            return
                        yield final_audio
        def on_start():
            with self._lock:
                if self._active is job:
                    self.first_audio_at = time.time()
                    job.first_play_at = time.perf_counter()
                    self.diagnostics.event('playback_start', turn_id=job.id, submit_to_play_s=round(job.first_play_at-job.t0, 3), output_latency_s=getattr(self.player, 'stats', {}).get('latency_s'))
                    input_latency = getattr(self.mic, 'latency_s', None)
                    output_latency = getattr(self.player, 'stats', {}).get('latency_s')
                    if input_latency is not None and output_latency is not None and hasattr(self.front, 'set_delay'):
                        self.front.set_delay(round((input_latency + output_latency) * 1000))
                    self._server_ok = True
                    self.status = 'speaking'
        def on_play(audio):
            if not turn.cancel.is_set():
                self.front.push_far(audio, rate=24000)
        def on_written(audio):
            # The blocking write may finish while cancellation arrives. Count
            # accepted samples even then; exclude writes that raised an error.
            job.played_samples += len(audio)
        with self._play_lock:
            if turn.cancel.is_set():
                self._archive_call(job, 'finish_generation', turn)
                self._archive_call(job, 'finish_playback', 0, True, error=turn.error)
                self._close_job_half(job, generation=True)
                self._close_job_half(job)
                job.produced.set()
                turn.done.set()
                if job.epoch == self._epoch:
                    self._replace(job.reply_id, 'assistant: [interrupted]')
                return
            request = self._spawn(produce)
            try:
                self.player.play_chunks(chunks(), cancel=turn.cancel,
                                        on_start=on_start, on_play=on_play,
                                        on_written=on_written)
            except Exception as exc:
                turn.error = 'Playback failed: ' + str(exc)
                turn.abort()
            finally:
                # No blocking join needed for audio to stop; the client lock
                # keeps the next request behind the closing response.
                request.join(timeout=.2)
                self.diagnostics.event('playback_end', turn_id=job.id, cancelled=turn.cancel.is_set(), error=turn.error, played_s=round(job.played_samples/24000, 3), duration_s=round(time.perf_counter()-job.t0, 3), playback=getattr(self.player, 'stats', {}))
                self._archive_call(job, 'finish_playback', job.played_samples, turn.cancel.is_set(), error=turn.error)
                self._close_job_half(job)
                with self._lock:
                    if job.epoch == self._epoch:
                        text = turn.text.strip()
                        # A playback failure also aborts generation. That
                        # transport cancellation is not a user interruption.
                        suffix = ((' [reply incomplete]' if text else '') if turn.error else
                                  ' [interrupted]' if turn.cancel.is_set() else
                                  ' [reply reached length limit]' if turn.truncated else
                                  '')
                        self._replace(job.reply_id, 'assistant: ' + (text or turn.error or ('[speaker check complete]' if job.test and job.played_samples else '[voice reply]' if job.played_samples else '[no reply]')) + suffix)
                    if self._active is job:
                        self._active = None
                        self.first = bool(turn.error or turn.cancel.is_set() or job.test)
                        self.error = turn.error
                        self.last_ttft = round(turn.ttft, 3) if turn.ttft is not None else None
                        self.last_audio_s = round(job.played_samples / 24000, 2)
                        self.last_turn_s = round(time.perf_counter() - job.t0, 2)
                        self.turns += int(not job.test)
                        if self.first_audio_at and self.heard_at:
                            self._latencies.append(self.first_audio_at - self.heard_at)
                        if self.status != 'recording':
                            self.status = 'listening'
                        self.front.clear_playback()
                if self.asr_client is not None and job.wav:
                    try:
                        self._asr_q.put_nowait((job.user_id, job.wav))
                    except queue.Full:
                        pass

    def interrupt(self, why='button', announce=True):
        with self._lock:
            job = self._active
            if job is not None:
                self.diagnostics.event('interrupt', turn_id=job.id, reason=why, played_s=round(job.played_samples/24000, 3))
                job.turn.abort()
                self._active = None
                # Backend cannot truncate its KV to the played prefix.
                self.first = True
            self.player.stop()
            self.front.clear_playback()
            self.interrupt_candidate = False
            if announce:
                self.interrupted_at = time.time()
            if self.status not in ('starting', 'error', 'recording'):
                self.status = 'listening'

    _interrupt = interrupt

    def reset(self):
        with self._lock:
            self.interrupt('reset', announce=False)
            self._epoch += 1
            self.first = True
            self._messages.clear()
            self.transcript_revision += 1
            self._capture_reset.set()
            self.heard_at = self.gen_start = self.first_audio_at = None
            self.error = None
            self.status = 'listening'

    def toggle_mic(self):
        with self._lock:
            self.set_mic_enabled(not self.mic_on)

    def set_mic_enabled(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError('Microphone enabled must be boolean')
        with self._lock:
            if self.mic_on == enabled:
                return
            self.mic_on = enabled
            self._capture_reset.set()
            self.mic.drain()
            if not self.mic_on and self.status == 'recording':
                self.status = 'listening'

    def mic_index(self):
        if self.mic.device is not None:
            return self.mic.device
        try:
            import sounddevice as sd
            return int(sd.default.device[0])
        except Exception:
            return -1

    def mic_name(self):
        return next((name for index, name in self._mics if index == self.mic_index()), 'default microphone')

    def _apply_device_requests(self):
        # Called only by the COM-initialized capture worker. HTTP threads
        # request changes; they never create long-lived WASAPI input streams.
        while True:
            try:
                index, done, result, cancelled = self._device_requests.get_nowait()
            except queue.Empty:
                return
            if cancelled.is_set():
                done.set()
                continue
            replacement = Mic(device=index)
            try:
                replacement.start()
                if cancelled.is_set():
                    replacement.stop()
                    continue
                with self._lock:
                    old, self.mic = self.mic, replacement
                    self._capture_reset.set()
                    self.error = None
                old.stop()
                result.append(True)
                self.diagnostics.event('mic_changed', device=index, latency_s=self.mic.latency_s)
            except Exception as exc:
                self.error = 'Could not open microphone: ' + str(exc)
                replacement.stop()
                result.append(False)
            finally:
                done.set()

    def set_mic(self, index):
        done, cancelled = threading.Event(), threading.Event()
        result = []
        self._device_requests.put((index, done, result, cancelled))
        if not done.wait(5):
            cancelled.set()
            self.error = 'Microphone change timed out. Try again.'
            return False
        return bool(result and result[0])

    def test_speaker(self):
        with self._lock:
            self.interrupt('speaker test', announce=False)
            self._job_id += 1
            job = Job(epoch=self._epoch, test=True, id=self._job_id)
            self._pending_jobs[job.id] = job
            if self.session_archive is not None:
                self._archive_pending.add(job.id)
            job.reply_id = self._say('[speaker test]')
            self._active = job
            self.status = 'thinking'
            self.heard_at = time.time()
            self.gen_start = self.first_audio_at = None
            self._spawn(self._run_job, job)
            return job

    def state(self):
        with self._lock:
            now = time.time()
            budget = float(np.median(self._latencies)) if self._latencies else .8
            partial = self._active.turn.text[-1500:] if self._active is not None else ''
            eta = None
            if self._active is not None and self.heard_at:
                eta = 0 if self.first_audio_at else max(0, round((budget - (now - self.heard_at)) * 1000))
            phase = 'hearing' if self.status == 'recording' else self.status
            playing = bool(getattr(self.player, 'playing', False))
            if self.interrupted_at and now - self.interrupted_at < .6:
                phase = 'interrupted'
            return dict(status=self.status, phase=phase, mic_on=self.mic_on,
                        active_turn=self._active is not None,
                        pending_turns=len(self._pending_jobs),
                        playback_active=playing,
                        archive_pending=len(self._archive_pending),
                        archive_session=str(self.session_archive.folder) if self.session_archive is not None else None,
                        archive_error=self.archive_error,
                        recording_safe=not self.mic_on and self._active is None and not self._pending_jobs and not playing and not self._archive_pending and not self.archive_error,
                        mic=self.mic_name(), mic_index=self.mic_index(), mics=self._mics,
                        server_ok=self._server_ok, level=self.front.level,
                        transcript=[m['text'] for m in self._messages if m['text']],
                        transcript_revision=self.transcript_revision,
                        assistant_partial=partial, heard_at=self.heard_at,
                        gen_start=self.gen_start, first_audio_at=self.first_audio_at,
                        eta_first_audio_ms=eta, eta_budget_ms=round(budget * 1000),
                        interrupt_candidate=self.interrupt_candidate, interrupted_at=self.interrupted_at,
                        last_ttft=self.last_ttft, last_audio_s=self.last_audio_s,
                        last_turn_s=self.last_turn_s, turns=self.turns,
                        error=self.error, warning='; '.join(w for w in [self.warning, self.archive_error] if w) or None,
                        audio_diagnostics=dict(residual_rejections=getattr(self.front, 'residual_rejections', 0), input_gain=getattr(self.mic, 'effective_gain', None), echo_score=getattr(self.front, 'echo_score', None), input_latency_s=getattr(self.mic, 'latency_s', None), input_drops=getattr(self.mic, 'dropped_blocks', 0), clipped_fraction=getattr(self.mic, 'clipped_fraction', 0), playback=getattr(self.player, 'stats', {})))


class Handler(BaseHTTPRequestHandler):
    def _reject_body(self, body, status, length):
        """Send the error before a bounded, graceful request-body drain.

        Closing a Windows socket with unread request data can replace the
        response with a reset. Half-close output first; never wait unboundedly
        for a client to finish an oversized or incomplete request.
        """
        self.close_connection = True
        self._send(body, status)
        self.wfile.flush()
        try:
            self.connection.shutdown(socket.SHUT_WR)
            deadline = time.monotonic() + .1
            remaining = min(max(length, 0), 65536)
            while remaining and time.monotonic() < deadline:
                self.connection.settimeout(max(.001, deadline - time.monotonic()))
                data = self.rfile.read1(min(4096, remaining))
                if not data:
                    break
                remaining -= len(data)
        except OSError:
            pass

    def _send(self, body, status=200, ctype='application/json'):
        data = body if isinstance(body, bytes) else body.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', ctype + '; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def _local(self):
        host = self.headers.get('Host', '')
        allowed = {f'localhost:{self.server.server_port}', f'127.0.0.1:{self.server.server_port}'}
        origin = self.headers.get('Origin')
        return host in allowed and (not origin or origin in {'http://' + h for h in allowed})

    def do_GET(self):
        if not self._local():
            return self._send('{"error":"Forbidden origin"}', 403)
        if self.path == '/api/state':
            self._send(json.dumps(self.server.assistant.state()))
        elif self.path == '/api/diagnostics':
            self._send(json.dumps(self.server.assistant.diagnostics.recent(100)))
        elif self.path == '/':
            self._send(PAGE, ctype='text/html')
        else:
            self._send('{"error":"Not found"}', 404)

    def do_POST(self):
        if not self._local():
            return self._send('{"error":"Forbidden origin"}', 403)
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 <= length <= 4096:
                return self._reject_body('{"error":"Request too large"}', 413, length)
            raw = self.rfile.read(length)
            assistant = self.server.assistant
            if self.path == '/api/mic':
                if raw.strip():
                    desired = json.loads(raw)
                    if not isinstance(desired, dict):
                        raise ValueError('Expected microphone settings')
                    assistant.set_mic_enabled(desired['enabled'])
                else:
                    assistant.toggle_mic()
            elif self.path == '/api/interrupt':
                assistant.interrupt()
            elif self.path == '/api/reset':
                assistant.reset()
            elif self.path == '/api/micdev':
                index = int(json.loads(raw or b'{}')['index'])
                if index not in [i for i, _ in assistant._mics]:
                    raise ValueError('Unknown microphone')
                if not assistant.set_mic(index):
                    return self._send('{"error":"Microphone unavailable"}', 409)
            elif self.path == '/api/test':
                job = assistant.test_speaker()
                self._send(json.dumps(dict(ok=True, turn_id=job.id)))
                return
            else:
                return self._send('{"error":"Not found"}', 404)
            self._send('{"ok":true}')
        except (ValueError, KeyError):
            self._send('{"error":"Invalid request"}', 400)

    def log_message(self, *args):
        pass


def main():
    asr_url = os.getenv('LOCALVOICE_ASR_SERVER')
    from pathlib import Path
    log_path = Path(__file__).resolve().parents[1] / 'outputs' / 'logs' / 'app.jsonl'
    archive_path = os.getenv('LOCALVOICE_RECORD_SESSION')
    archive = None
    if archive_path:
        from session_archive import SessionArchive
        archive = SessionArchive(Path(archive_path))
        with (archive.folder/'provenance.json').open('x', encoding='utf-8') as stream:
            json.dump(dict(source_files_sha256=source_fingerprint(),
                           backend=os.getenv('LOCALVOICE_SERVER', 'http://127.0.0.1:8080'),
                           scope='App source and endpoint; native build/model identity must come from the experiment protocol.'), stream, indent=2)
    assistant = Assistant(asr_client=ServerClient(asr_url) if asr_url else None, diagnostics=Diagnostics(log_path), start_muted=os.getenv("LOCALVOICE_START_MUTED") == "1", session_archive=archive)
    server = ThreadingHTTPServer(('127.0.0.1', UI_PORT), Handler)
    server.assistant = assistant
    assistant.start()
    print(f'LocalVoice on http://localhost:{UI_PORT}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        assistant.close()
        server.server_close()


if __name__ == '__main__':
    from contextlib import ExitStack
    from backend_session import backend_session, canonical_base
    endpoints = {canonical_base(os.getenv('LOCALVOICE_SERVER', 'http://127.0.0.1:8080'))}
    if os.getenv('LOCALVOICE_ASR_SERVER'):
        endpoints.add(canonical_base(os.environ['LOCALVOICE_ASR_SERVER']))
    with ExitStack() as sessions:
        for endpoint in sorted(endpoints):
            sessions.enter_context(backend_session(endpoint))
        main()
