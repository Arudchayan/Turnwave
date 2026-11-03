"""Turnwave: continuous listening, cancellable turns, and a local web UI."""
import collections
import json
import math
import os
import queue
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from audio_io import Mic, Player, list_input_devices, audio_thread
from client import ServerClient, Turn, wav16k_bytes
from duplex import DuplexFrontend
from diagnostics import Diagnostics, HistoryFileHandler, source_fingerprint
from ui import PAGE
from session_archive import ArchiveSealedError
import logging

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
    agent_input: str = ''
    agent_input_saved: bool = False
    agent_reply_saved: bool = False
    asr_s: float = 0.0
    agent_s: float = 0.0
    agent_trace: object = None
    sources: list = field(default_factory=list)


class Assistant:
    """No import-time devices or threads; dependencies can be simulated in tests."""
    def __init__(self, mic=None, player=None, front=None, client=None, asr_client=None, diagnostics=None, start_muted=False, pre_roll_ms=250, session_archive=None, agent=None, trace_path=None):
        if not 20 <= pre_roll_ms <= 2000:
            raise ValueError('Pre-roll must be between 20 and 2000 ms')
        self.pre_roll_ms = pre_roll_ms
        self.pre_roll_samples = round(pre_roll_ms * 16)
        self.mic = mic if mic is not None else Mic()
        self.player = player if player is not None else Player()
        self.front = front if front is not None else DuplexFrontend()
        self.client = client if client is not None else ServerClient(os.getenv('LOCALVOICE_SERVER', 'http://localhost:8080'))
        self.asr_client = asr_client
        self.agent = agent
        self._saved_agent = agent
        self._agent_history = []
        self._agent_sources = []
        self._agent_notes = []
        self._agent_search_warning = None
        self._agent_phase = None
        self._voice_params = {
            'temperature': 0.7,
            'top_k': 40,
            'top_p': 0.9,
            'min_p': 0.05,
            'repeat_penalty': 1.05,
            'voice': os.getenv('LOCALVOICE_VOICE', '').strip(),
        }
        if asr_client is not None and asr_client.base.rstrip('/') == self.client.base.rstrip('/'):
            raise ValueError('Transcription needs a separate server context')
        self.diagnostics = diagnostics if diagnostics is not None else Diagnostics()
        self._traces = collections.deque(maxlen=50)
        self.trace_path = Path(trace_path) if trace_path else None
        self._trace_handler = None
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._trace_handler = HistoryFileHandler(self.trace_path, maxBytes=2_000_000, backupCount=0, encoding='utf-8')
            self._trace_handler.setFormatter(logging.Formatter('%(message)s'))
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
        self._recording_sealed = False
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
        self._barge = None
        self._warmup = None
        self._audio_warmup = None
        self._mics = []
        self._mic_devices = []
        self._mic_identity = None
        # Injected microphones may have no PortAudio implementation at all.
        self._owns_mic_catalog = mic is None
        if self._owns_mic_catalog:
            self._set_mic_catalog(self._read_mic_catalog())
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

    def _warm_audio_input(self):
        """Exercise the audio-input chat path without playing or transcribing it."""
        samples = int(.6 * 16000)
        t = np.arange(samples, dtype=np.float32) / 16000
        voice = (.03 * np.sin(2 * np.pi * 170 * t)
                 + .012 * np.sin(2 * np.pi * 340 * t))
        envelope = .65 + .35 * np.sin(2 * np.pi * 3 * t)
        noise = np.random.default_rng(20261029).normal(0, .002, samples).astype(np.float32)
        audio = np.clip(voice * envelope + noise, -1, 1).astype(np.float32)
        turn = Turn()
        with self._lock:
            self._audio_warmup = turn
        try:
            self.client.chat_turn(wav16k_bytes(audio), turn, True,
                                  max_tokens=8, require_audio=False)
        finally:
            with self._lock:
                if self._audio_warmup is turn:
                    self._audio_warmup = None
        return turn

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
                # The first real audio turn pays a cold audio-input path. Warm
                # that path too, discarding the reply, while no conversation
                # exists yet. Failure here must never block the microphone.
                if not self._stop.is_set():
                    started = time.perf_counter()
                    try:
                        audio_warm = self._warm_audio_input()
                        self.diagnostics.event(
                            'audio_warmup', ok=not audio_warm.error,
                            error=audio_warm.error,
                            first_audio_s=audio_warm.ttft,
                            first_text_s=audio_warm.first_text,
                            duration_s=round(time.perf_counter() - started, 3),
                            max_tokens=8)
                    except Exception as exc:
                        self.diagnostics.event('audio_warmup_error', error=str(exc))
                        if self.warning is None:
                            self.warning = 'Audio warmup skipped: ' + str(exc)
            while not self._stop.is_set():
                try:
                    self._apply_device_requests()
                    self.mic.start()
                    with self._lock:
                        self._cache_mic_identity()
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
            audio_warm = self._audio_warmup
        if warm is not None:
            warm.abort()
        if audio_warm is not None:
            audio_warm.abort()
        self.interrupt('shutdown')
        try:
            self.mic.stop()
        finally:
            for thread in list(self._threads):
                if thread is not threading.current_thread():
                    thread.join(timeout=1)
            if hasattr(self.client, 'close'):
                self.client.close()
            self.diagnostics.event('shutdown')
            self.diagnostics.close()
            with self._lock:
                if self._trace_handler:
                    self._trace_handler.close()
                    self._trace_handler = None

    def traces(self, limit=20):
        with self._lock:
            return list(self._traces)[-limit:]

    def _record_trace(self, job, checks):
        record = dict(
            ts=round(time.time(), 3),
            turn_id=job.id,
            epoch=job.epoch,
            test=job.test,
            agent_mode=bool(self.agent and not job.test),
            status=('cancelled' if job.turn.cancel.is_set() else 'error' if job.turn.error else 'ok'),
            error=job.turn.error,
            user_text=job.agent_input if self.agent and not job.test else None,
            reply_text=job.turn.text.strip() or None,
            asr_s=job.asr_s if self.agent and not job.test else None,
            agent_s=job.agent_s if self.agent and not job.test else None,
            agent_trace=job.agent_trace,
            sources=list(job.sources),
            tts_first_audio_s=round(job.turn.ttft, 3) if job.turn.ttft is not None else None,
            generated_audio_s=round(job.turn.audio_samples / 24000, 3),
            played_audio_s=round(job.played_samples / 24000, 3),
            total_turn_s=round(time.perf_counter() - job.t0, 3),
            checks=checks,
        )
        with self._lock:
            self._traces.append(record)
            if self._trace_handler:
                message = json.dumps(record, ensure_ascii=False, allow_nan=False)
                self._trace_handler.handle(logging.LogRecord('localvoice.trace', logging.INFO, '', 0, message, (), None))
        return record

    def _health_loop(self):
        while not self._stop.is_set():
            if self._active is None:
                self._server_ok = self.client.health()
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

    def _remember_agent_input(self, job):
        """Keep a recognized request once, even if its reply is interrupted.

        Caller holds _lock. An unfinished reply is intentionally not recorded
        as though it had been heard.
        """
        if self.agent and job.epoch == self._epoch and job.agent_input and not job.agent_input_saved:
            self._agent_history.append({'role': 'user', 'content': job.agent_input})
            job.agent_input_saved = True
            self._agent_history = self._agent_history[-20:]
            while self._agent_history and self._agent_history[0]['role'] != 'user':
                self._agent_history.pop(0)

    def _remember_agent_reply(self, job, interrupted=False):
        """Record a reply once playback has begun so the agent knows what it already said."""
        if (self.agent and not job.test and job.epoch == self._epoch
                and job.played_samples > 0 and not job.agent_reply_saved):
            text = job.turn.text.strip()
            if text:
                self._remember_agent_input(job)
                suffix = ' [interrupted]' if interrupted and not text.endswith('[interrupted]') else ''
                self._agent_history.append({'role': 'assistant', 'content': text + suffix})
                job.agent_reply_saved = True
                self._agent_history = self._agent_history[-20:]
                while self._agent_history and self._agent_history[0]['role'] != 'user':
                    self._agent_history.pop(0)
                if job.sources:
                    self._agent_sources = list(job.sources)

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
            capture_queue = self.mic.q
            read = getattr(capture_queue, 'get_with_generation', None)
            try:
                if read is None:
                    generation, block = None, capture_queue.get(timeout=.05)
                else:
                    generation, block = read(timeout=.05)
            except queue.Empty:
                continue
            with self._lock:
                # Reset/mute may happen after dequeue, or while a callback is
                # still preparing this block. Keep the first genuinely fresh
                # block even when reset happened during a blocking read.
                if (capture_queue is not self.mic.q
                        or generation is not None and generation != capture_queue.generation):
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
                    candidate = self._drop_barge()
                    if candidate is not None:
                        job = candidate.get('job')
                        self.diagnostics.event(
                            'barge_restore', turn_id=job.id if job is not None else None,
                            duck_level=1.0, reason='capture reset',
                            barge_played_s=round(candidate.get('played_s', 0), 3))
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
                        job = self._active
                        if job is not None:
                            playing = bool(getattr(self.player, 'playing', False))
                            # Wait for the submit gate before cancelling an in-flight turn.
                            # A cough, breath, or echo false positive must not destroy a reply
                            # whether it is already speaking or still thinking.
                            self._barge = dict(job=job, at=time.time(),
                                               played_s=job.played_samples / 24000)
                            if hasattr(self.front, 'use_barge_endpoint'):
                                self.front.use_barge_endpoint()
                            duck = .2 if (playing or job.played_samples > 0) else 1.0
                            if duck < 1.0:
                                self.player.set_gain(duck)
                            self.diagnostics.event(
                                'barge_candidate', turn_id=job.id,
                                duck_level=duck,
                                barge_played_s=round(job.played_samples / 24000, 3))
                        recording = True
                        rec = list(pre)
                        rec_samples = sum(map(len, rec))
                        self.status = 'recording'
            else:
                rec.append(clean)
                rec_samples += len(clean)
            if recording and (ended or rec_samples >= 20 * 16000):
                audio = np.concatenate(rec)
                self.diagnostics.event('speech_end', seconds=round(len(audio) / 16000, 3), rms=round(float(np.sqrt(np.mean(audio ** 2))), 5), capped=not ended, end_silence_ms=getattr(self.front, 'end_chunks', 0) * 32, pre_roll_ms=self.pre_roll_ms)
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
                            self._confirm_barge()
                            self.submit_audio(audio)
                        else:
                            # Mute/reset during endpointing: discard the stale
                            # turn and make sure playback is not left ducked.
                            self._drop_barge()
                else:
                    with self._lock:
                        candidate = self._drop_barge()
                        if candidate is not None:
                            job = candidate.get('job')
                            self.diagnostics.event(
                                'barge_restore', turn_id=job.id if job is not None else None,
                                duck_level=1.0, reason='short utterance',
                                barge_played_s=round(candidate.get('played_s', 0), 3))
                            self.status = (('speaking' if getattr(self.player, 'playing', False) or self.first_audio_at else 'thinking')
                                           if job is self._active else 'listening')
                        else:
                            self.status = 'listening'

    def submit_audio(self, audio):
        wav = wav16k_bytes(audio)
        with self._lock:
            self._require_recording_open()
            self.interrupt('new turn', announce=False)
            self.error = None
            self._job_id += 1
            job = Job(epoch=self._epoch, first=self.first, wav=wav, id=self._job_id)
            if self.agent:
                self._agent_sources = []
                self._agent_notes = []
                self._agent_search_warning = None
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

    def submit_text(self, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            raise ValueError('Prompt text must be 1 to 2000 characters')
        clean_text = text.strip()
        with self._lock:
            self._require_recording_open()
            self.interrupt('text prompt', announce=False)
            self.error = None
            self._job_id += 1
            job = Job(epoch=self._epoch, first=self.first, wav=b'', id=self._job_id, agent_input=clean_text)
            if self.agent:
                self._agent_sources = []
                self._agent_notes = []
                self._agent_search_warning = None
            self._pending_jobs[job.id] = job
            if self.session_archive is not None:
                self._archive_pending.add(job.id)
            self.diagnostics.event('turn_submit', turn_id=job.id, first=job.first, input_s=0.0, text_prompt=True)
            job.user_id = self._say('user: ' + clean_text)
            job.reply_id = self._say('')
            self._active = job
            self.status = 'thinking'
            self.heard_at = time.time()
            self.gen_start = self.first_audio_at = None
            self.interrupt_candidate = False
            self._spawn(self._run_job, job)
            return job

    def _speak_answer(self, answer, turn, receive):
        """Synthesize short answers in one call; pipeline multi-sentence answers for fast first audio."""
        text = answer.strip()
        if len(text) <= 110:
            self.client.tts(text, turn, max_tokens=512, on_audio=receive)
            return
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])(?<!\b[A-Za-z]\.)\s+', text) if s.strip()]
        if len(sentences) <= 1:
            self.client.tts(text, turn, max_tokens=512, on_audio=receive)
            return
        segments = []
        current = ''
        for sent in sentences:
            limit = 120 if not segments else 220
            min_len = 25 if not segments else 80
            if not current:
                current = sent
            elif len(current) < min_len or len(current) + 1 + len(sent) <= limit:
                current += ' ' + sent
            else:
                segments.append(current)
                current = sent
        if current:
            segments.append(current)
        if len(segments) <= 1:
            self.client.tts(text, turn, max_tokens=512, on_audio=receive)
            return
        first_ttft = None
        for idx, segment in enumerate(segments):
            if turn.cancel.is_set() or turn.error:
                break
            if idx > 0:
                turn.done.clear()
                turn.finish_reason = None
                turn.request_observer = None
            self.client.tts(segment, turn, max_tokens=512, on_audio=receive)
            if first_ttft is None and turn.ttft is not None:
                first_ttft = turn.ttft
        if first_ttft is not None:
            turn.ttft = first_ttft

    def _drop_barge(self):
        """Restore full playback gain and forget the candidate. Caller holds lock."""
        candidate = self._barge
        self._barge = None
        if candidate is not None:
            self.player.set_gain(1.0)
        return candidate

    def _confirm_barge(self):
        """Announce a confirmed candidate before submit cancels the old reply."""
        candidate = self._barge
        self._barge = None
        if candidate is None:
            return
        job = candidate.get('job')
        if job is self._active:
            self.diagnostics.event(
                'barge_confirm', turn_id=job.id, duck_level=.2,
                barge_played_s=round(candidate.get('played_s', 0), 3))

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
        turn.request_observer = None if self.agent and not job.test else lambda payload: self._archive_call(job, 'record_request', payload)
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
                elif self.agent:
                    if job.agent_input:
                        text = job.agent_input
                        job.asr_s = 0.0
                    else:
                        with self._lock:
                            if self._active is job:
                                self._agent_phase = 'transcribing'
                        # ASR and TTS are serialized on the same local context.
                        # Both reset it, while cloud history is maintained separately.
                        asr_t0 = time.perf_counter()
                        try:
                            text = self.client.asr_text(job.wav, turn=turn)
                        finally:
                            job.asr_s = round(time.perf_counter() - asr_t0, 3)
                            # ASR uses the turn for cancellation, but its partial
                            # text must never appear as an assistant reply.
                            turn.text_parts.clear()
                    with self._lock:
                        if turn.cancel.is_set() or self._active is not job:
                            return
                        job.agent_input = text
                        history = list(self._agent_history)
                        self._replace(job.user_id, 'user: ' + text)
                        self._agent_phase = 'thinking'
                    self.diagnostics.event('agent_asr', turn_id=job.id, duration_s=job.asr_s,
                                           text_chars=len(text.strip()), empty=not bool(text.strip()))
                    turn.done.clear()
                    turn.first_event = turn.first_text = turn.ttft = None
                    turn.finish_reason = None
                    def searching(active):
                        with self._lock:
                            if self._active is job:
                                self._agent_phase = 'searching' if active else 'thinking'
                    agent_t0 = time.perf_counter()
                    try:
                        if getattr(self.agent, 'supports_input_modality', False):
                            result = self.agent.respond(text, history, turn, on_search=searching,
                                                        input_modality='voice' if job.wav else 'text')
                        else:
                            result = self.agent.respond(text, history, turn, on_search=searching)
                    finally:
                        job.agent_s = round(time.perf_counter() - agent_t0, 3)
                        job.agent_trace = getattr(self.agent, 'last_trace', None)
                        with self._lock:
                            if self._active is job:
                                self._agent_search_warning = (job.agent_trace or {}).get('search_warning')
                    if result is None or turn.cancel.is_set():
                        return
                    answer, job.sources = result
                    notes = list((job.agent_trace or {}).get('expert_notes') or [])
                    tools_called = [tc.get('tool') for tc in (job.agent_trace or {}).get('tool_calls', [])]
                    self.diagnostics.event('agent_turn', turn_id=job.id, duration_s=job.agent_s,
                                           provider=getattr(self.agent, 'label', None),
                                           model=getattr(self.agent, 'model', None),
                                           tools_called=tools_called,
                                           sources_count=len(job.sources),
                                           reply_chars=len(answer))
                    with self._lock:
                        if self._active is job:
                            self._agent_phase = 'voicing'
                            if job.sources:
                                self._agent_sources = list(job.sources)
                            if notes:
                                self._agent_notes = notes
                    turn.text_parts.append(answer)
                    turn.request_observer = lambda payload: self._archive_call(job, 'record_request', payload)
                    self._speak_answer(answer, turn, receive)
                    if not turn.error and not turn.cancel.is_set() and not turn.audio_samples:
                        turn.error = 'Speech synthesis returned no audio.'
                    elif not turn.error and not turn.cancel.is_set() and turn.truncated:
                        turn.error = 'Speech synthesis reached its length limit; the spoken reply may be incomplete.'
                elif job.agent_input and not job.wav:
                    turn.text_parts.append(job.agent_input)
                    self._speak_answer(job.agent_input, turn, receive)
                else:
                    self.client.chat_turn(job.wav, turn, job.first, on_audio=receive)
            except Exception as exc:
                if not turn.cancel.is_set():
                    turn.error = str(exc)
            finally:
                self.diagnostics.event('generation_end', turn_id=job.id, cancelled=turn.cancel.is_set(), error=turn.error, finish_reason=turn.finish_reason, text_chars=len(turn.text), audio_s=round(turn.audio_samples/24000, 3), first_audio_s=turn.ttft, first_text_s=turn.first_text, first_event_s=turn.first_event, request_queue_s=turn.queue_s, headers_s=turn.headers_s)
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
                    if self.agent and not job.test and self._active is job and turn.error:
                        self._remember_agent_input(job)
                    if (self.agent and not job.test and job.epoch == self._epoch
                            and job.played_samples > 0 and not turn.error):
                        self._remember_agent_reply(job, interrupted=turn.cancel.is_set())
                    if job.epoch == self._epoch:
                        text = (turn.text.strip() if not self.agent or job.test or job.agent_input else '')
                        # A playback failure also aborts generation. That
                        # transport cancellation is not a user interruption.
                        suffix = ((' [reply incomplete]' if text else '') if turn.error else
                                  ' [interrupted]' if turn.cancel.is_set() else
                                  ' [reply reached length limit]' if turn.truncated else
                                  '')
                        self._replace(job.reply_id, 'assistant: ' + (text or turn.error or ('[speaker check complete]' if job.test and job.played_samples else '[voice reply]' if job.played_samples else '[no reply]')) + suffix)
                    if self._active is job:
                        self._agent_phase = None
                        if self._barge is not None and self._barge.get('job') is job:
                            candidate = self._drop_barge()
                            self.diagnostics.event(
                                'barge_restore', turn_id=job.id, duck_level=1.0,
                                reason='playback ended',
                                barge_played_s=round(candidate.get('played_s', 0), 3))
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
                pb_stats = getattr(self.player, 'stats', {}) or {}
                agent_checks = (job.agent_trace or {}).get('checks') or {}
                checks = dict(
                    asr_ok=bool(not self.agent or job.test or job.agent_input.strip() or turn.cancel.is_set()),
                    tools_ok=bool(agent_checks.get('tool_calls_succeeded', True)),
                    reply_non_empty=bool(turn.text.strip() or job.test or turn.cancel.is_set()),
                    tts_audio_ok=bool(turn.audio_samples > 0 or turn.cancel.is_set()),
                    playback_clean=int(pb_stats.get('underflows', 0) or 0) == 0,
                    error_free=turn.error is None,
                )
                checks['all_passed'] = all(checks.values())
                self.diagnostics.event('turn_checks', turn_id=job.id, all_passed=checks['all_passed'], checks=checks)
                self._record_trace(job, checks)
                if self.asr_client is not None and job.wav and not self.agent:
                    try:
                        self._asr_q.put_nowait((job.user_id, job.wav))
                    except queue.Full:
                        pass

    def interrupt(self, why='button', announce=True):
        with self._lock:
            job = self._active
            if job is not None:
                self.diagnostics.event('interrupt', turn_id=job.id, reason=why, played_s=round(job.played_samples/24000, 3))
                self._remember_agent_input(job)
                self._remember_agent_reply(job, interrupted=True)
                job.turn.abort()
                self._active = None
                # Backend cannot truncate its KV to the played prefix.
                self.first = True
            self._agent_phase = None
            self._drop_barge()
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
            self._agent_history.clear()
            self._agent_sources.clear()
            self._agent_notes.clear()
            self._agent_search_warning = None
            if self.agent and hasattr(self.agent, 'reset'):
                self.agent.reset()
            self.transcript_revision += 1
            self._capture_reset.set()
            self.mic.drain()
            self.heard_at = self.gen_start = self.first_audio_at = None
            self.error = None
            self.status = 'listening'

    def toggle_mic(self):
        with self._lock:
            self.set_mic_enabled(not self.mic_on)

    def set_end_silence(self, milliseconds):
        with self._lock:
            if self.status == 'recording':
                raise RuntimeError('Finish speaking before changing the pause')
            return self.front.set_end_silence(milliseconds)

    def set_agent_mode(self, mode, model=None):
        if not isinstance(mode, str):
            raise ValueError('Invalid agent mode')
        mode = mode.strip().lower()
        if mode not in ('local', 'opencode-go', 'opencode-zen', 'go', 'zen', 'custom'):
            raise ValueError('Unsupported agent mode')
        if model is not None and (not isinstance(model, str) or not model.strip() or len(model) > 100):
            raise ValueError('Invalid model name')
        with self._lock:
            self.interrupt('mode change', announce=False)
            if mode == 'local':
                self.agent = None
                self._agent_sources = []
                self._agent_search_warning = None
                self.diagnostics.event('agent_mode_changed', agent_mode=False, provider=None, model=None)
                return {'ok': True, 'agent_mode': False}
            provider = ('custom' if mode == 'custom'
                        else 'zen' if mode in ('opencode-zen', 'zen') else 'go')
            desired_model = (model.strip() if model else None) or getattr(self._saved_agent, 'model', None) or os.getenv('LOCALVOICE_AGENT_MODEL', 'beam-free')
            if self._saved_agent is not None and getattr(self._saved_agent, 'provider', 'go') == provider:
                self._saved_agent.model = desired_model
                self.agent = self._saved_agent
            else:
                from agent import AgentClient
                key_name = ('LOCALVOICE_CUSTOM_API_KEY' if provider == 'custom'
                            else 'OPENCODE_ZEN_API_KEY' if provider == 'zen' else 'OPENCODE_GO_API_KEY')
                search_key = os.getenv('BRAVE_SEARCH_API_KEY')
                previous = self.agent or self._saved_agent
                search_provider = (getattr(previous, 'search_provider', None) or
                                   os.getenv('LOCALVOICE_SEARCH', '').strip().lower() or
                                   ('brave' if search_key else 'ddg'))
                mem_path = self.trace_path.parent / 'agent_memory.json' if self.trace_path is not None else None
                prev_persona = getattr(self.agent or self._saved_agent, 'persona', 'jarvis') or 'jarvis'
                self._saved_agent = AgentClient(
                    os.getenv(key_name), provider=provider, model=desired_model,
                    search_key=search_key, search_provider=search_provider,
                    local_tools=True, persona=prev_persona, memory_path=mem_path)
                self.agent = self._saved_agent
            self.diagnostics.event('agent_mode_changed', agent_mode=True,
                                   provider=getattr(self.agent, 'label', None),
                                   model=getattr(self.agent, 'model', None))
            return {'ok': True, 'agent_mode': True, 'provider': getattr(self.agent, 'label', None),
                    'model': getattr(self.agent, 'model', None)}

    def set_search_provider(self, provider):
        if provider not in ('none', 'ddg', 'bing', 'brave'):
            raise ValueError('Unsupported search provider')
        with self._lock:
            if self.agent is None:
                raise ValueError('Select a cloud agent before configuring web search')
            if provider == 'brave' and not getattr(self.agent, '_search_key', None):
                raise ValueError('Brave search requires BRAVE_SEARCH_API_KEY in the app environment')
            if getattr(self.agent, 'search_provider', None) != provider:
                self.interrupt('search provider change', announce=False)
                self.agent.search_provider = provider
                self.agent.search_enabled = provider != 'none'
                self._agent_sources = []
                self._agent_search_warning = None
                self.diagnostics.event('search_provider_changed', provider=provider)
            return {'ok': True, 'search_provider': provider, 'search_enabled': provider != 'none'}

    def set_persona(self, persona):
        if not isinstance(persona, str) or not persona.strip():
            raise ValueError('Invalid persona')
        with self._lock:
            target = self.agent or self._saved_agent
            if target is None or not hasattr(target, 'set_persona'):
                raise ValueError('Voice persona requires an active cloud agent')
            selected = target.set_persona(persona)
            self.diagnostics.event('persona_changed', persona=selected)
            return {'ok': True, 'persona': selected}

    def set_voice_params(self, params):
        if not isinstance(params, dict) or not params:
            raise ValueError('Expected voice parameters')
        allowed = {'temperature', 'top_k', 'top_p', 'min_p', 'repeat_penalty', 'voice'}
        if not set(params) <= allowed:
            raise ValueError('Unsupported voice parameter')
        updated = dict(self._voice_params)
        if 'temperature' in params:
            val = float(params['temperature'])
            if isinstance(params['temperature'], bool) or not math.isfinite(val) or not 0.0 <= val <= 2.0:
                raise ValueError('Temperature must be between 0.0 and 2.0')
            updated['temperature'] = round(val, 3)
        if 'top_k' in params:
            val = params['top_k']
            if isinstance(val, bool) or not isinstance(val, int) or not 1 <= val <= 200:
                raise ValueError('top_k must be an integer between 1 and 200')
            updated['top_k'] = val
        if 'top_p' in params:
            val = float(params['top_p'])
            if isinstance(params['top_p'], bool) or not math.isfinite(val) or not 0.05 <= val <= 1.0:
                raise ValueError('top_p must be between 0.05 and 1.0')
            updated['top_p'] = round(val, 3)
        if 'min_p' in params:
            val = float(params['min_p'])
            if isinstance(params['min_p'], bool) or not math.isfinite(val) or not 0.0 <= val <= 1.0:
                raise ValueError('min_p must be between 0.0 and 1.0')
            updated['min_p'] = round(val, 3)
        if 'repeat_penalty' in params:
            val = float(params['repeat_penalty'])
            if isinstance(params['repeat_penalty'], bool) or not math.isfinite(val) or not 0.8 <= val <= 2.0:
                raise ValueError('repeat_penalty must be between 0.8 and 2.0')
            updated['repeat_penalty'] = round(val, 3)
        if 'voice' in params:
            val = params['voice']
            if not isinstance(val, str) or len(val.strip()) > 64 or any(ord(c) < 32 for c in val):
                raise ValueError('Invalid voice identifier')
            updated['voice'] = val.strip()
        with self._lock:
            self._voice_params = updated
            if hasattr(self.client, 'set_voice_params'):
                self.client.set_voice_params(**updated)
            self.diagnostics.event('voice_params_changed', **updated)
            return {'ok': True, 'voice_params': dict(self._voice_params)}

    def _archive_sealed(self):
        return self._recording_sealed or bool(getattr(self.session_archive, 'sealed', False))

    def _require_recording_open(self):
        if self._archive_sealed():
            raise ArchiveSealedError('This recording is finalized. Start a new session to continue.')

    def seal_archive(self, expected_session):
        """Atomically fence new activity after verifying the exact drained session."""
        if expected_session is not None:
            if (not isinstance(expected_session, str) or not expected_session
                    or len(expected_session) > 4096 or '\x00' in expected_session
                    or not Path(expected_session).is_absolute()):
                raise ValueError('Expected session must be an absolute local path or null')
            expected = Path(expected_session).resolve()
        else:
            expected = None
        with self._lock:
            actual = self.session_archive.folder.resolve() if self.session_archive is not None else None
            if actual != expected:
                raise ValueError('Recording session identity differs')
            if (self.mic_on or self._active is not None or self._pending_jobs
                    or bool(getattr(self.player, 'playing', False)) or self._archive_pending
                    or self.archive_error or self.status not in {'listening', 'error'}):
                raise RuntimeError('Turn off the microphone and wait for the reply to finish.')
            if self.session_archive is not None:
                try:
                    self.session_archive.seal()
                except Exception as exc:
                    self.archive_error = 'Recording could not be sealed: ' + str(exc)
                    raise
            self._recording_sealed = True
            return {'ok': True, 'archive_session': str(actual) if actual is not None else None,
                    'archive_sealed': True}

    def set_mic_enabled(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError('Microphone enabled must be boolean')
        with self._lock:
            if enabled:
                self._require_recording_open()
            if self.mic_on == enabled:
                return
            self.mic_on = enabled
            self._capture_reset.set()
            self.mic.drain()
            if not self.mic_on:
                candidate = self._drop_barge()
                if candidate is not None:
                    job = candidate.get('job')
                    self.diagnostics.event(
                        'barge_restore', turn_id=job.id if job is not None else None,
                        duck_level=1.0, reason='microphone disabled',
                        barge_played_s=round(candidate.get('played_s', 0), 3))
                if self.status == 'recording':
                    self.status = 'listening'

    @staticmethod
    def _read_mic_catalog():
        # Only initialization/the capture worker enumerates devices, never a
        # state poll or the real-time audio callback.
        try:
            return [dict(device) for device in list_input_devices()]
        except Exception:
            return []

    def _set_mic_catalog(self, devices):
        self._mic_devices = [dict(device) for device in devices]
        self._mics = [(device['index'], device['full_name'][:48])
                      for device in devices
                      if isinstance(device.get('index'), int)
                      and isinstance(device.get('full_name'), str)
                      and device.get('max_input_channels', 0) > 0]

    def _cache_mic_identity(self):
        self._mic_identity = None
        try:
            identity = dict(self.mic.selected_device_identity)
            integer_fields = ('index', 'host_api_index', 'max_input_channels', 'capture_channels')
            if any(type(identity.get(key)) is not int for key in integer_fields):
                return
            if identity['index'] < 0 or identity['host_api_index'] < 0:
                return
            if not 0 < identity['capture_channels'] <= identity['max_input_channels']:
                return
            if any(not isinstance(identity.get(key), str) or not identity[key].strip()
                   for key in ('full_name', 'host_api')):
                return
            for key in ('default_samplerate', 'capture_samplerate', 'output_samplerate'):
                value = identity.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    return
            if identity.get('capture_dtype') != 'float32':
                return
            # Indices and host-api indices are process-local. A full descriptor
            # must also identify a unique endpoint without either index.
            stable = ('full_name', 'host_api', 'max_input_channels', 'default_samplerate')
            matches = [device for device in self._mic_devices
                       if all(device.get(key) == identity[key] for key in stable)]
            if (len(matches) != 1 or matches[0].get('index') != identity['index']
                    or matches[0].get('host_api_index') != identity['host_api_index']):
                return
            keys = ('index', 'full_name', 'host_api', 'host_api_index', 'max_input_channels',
                    'default_samplerate', 'capture_samplerate', 'capture_channels',
                    'capture_dtype', 'output_samplerate')
            self._mic_identity = {key: identity[key] for key in keys}
        except (AttributeError, TypeError, ValueError):
            # Older injected fakes and incomplete device queries cannot attest
            # an opened endpoint; preserve ordinary operation without guessing.
            return

    def mic_index(self):
        if self._mic_identity is not None:
            return self._mic_identity['index']
        device = getattr(self.mic, 'device', None)
        return device if type(device) is int else -1

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
                devices = self._read_mic_catalog() if self._owns_mic_catalog else None
                with self._lock:
                    publish = not cancelled.is_set()
                    if publish:
                        old, self.mic = self.mic, replacement
                        if devices is not None:
                            self._set_mic_catalog(devices)
                        self._cache_mic_identity()
                        self._capture_reset.set()
                        self.error = None
                if not publish:
                    replacement.stop()
                    continue
                old.stop()
                if hasattr(self.player, 'refresh_device'):
                    self.player.refresh_device()
                result.append(True)
                self.diagnostics.event('mic_changed', device=index, latency_s=self.mic.latency_s)
            except Exception as exc:
                self.error = 'Could not open microphone: ' + str(exc)
                with self._lock:
                    if self.mic is replacement:
                        self._mic_identity = None
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
            self._require_recording_open()
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
            partial = (self._active.turn.text[-1500:] if self._active is not None and
                       (not self.agent or self._active.test or self._active.agent_input) else '')
            eta = None
            if self._active is not None and self.heard_at:
                eta = 0 if self.first_audio_at else max(0, round((budget - (now - self.heard_at)) * 1000))
            phase = 'hearing' if self.status == 'recording' else self.status
            playing = bool(getattr(self.player, 'playing', False))
            if self.interrupted_at and now - self.interrupted_at < .6:
                phase = 'interrupted'
            tools = ([t['function']['name'] for t in self.agent.available_tools()]
                     if self.agent and hasattr(self.agent, 'available_tools') else [])
            latest_trace = None
            if self._traces:
                last = self._traces[-1]
                at = last.get('agent_trace') or {}
                latest_trace = dict(
                    turn_id=last.get('turn_id'),
                    status=last.get('status'),
                    asr_s=last.get('asr_s'),
                    agent_s=last.get('agent_s'),
                    tts_first_audio_s=last.get('tts_first_audio_s'),
                    played_audio_s=last.get('played_audio_s'),
                    tools_used=[tc.get('tool') for tc in at.get('tool_calls', [])],
                    checks=last.get('checks'),
                )
            return dict(status=self.status, phase=phase, mic_on=self.mic_on,
                        end_silence_ms=getattr(self.front, 'end_chunks', 0) * 32 or None,
                        agent_mode=bool(self.agent), agent_phase=self._agent_phase,
                        agent_provider=getattr(self.agent, 'label', None),
                        agent_model=getattr(self.agent, 'model', None),
                        agent_persona=getattr(self.agent, 'persona', None),
                        agent_tools=tools,
                        voice_params=dict(self._voice_params),
                        search_enabled=bool(self.agent and self.agent.search_enabled),
                        search_provider=getattr(self.agent, 'search_provider', None),
                        agent_sources=list(self._agent_sources),
                        agent_notes=list(self._agent_notes),
                        trace_count=len(self._traces),
                        trace_file=self.trace_path.name if self.trace_path is not None else None,
                        last_trace=latest_trace,
                        active_turn=self._active is not None,
                        pending_turns=len(self._pending_jobs),
                        playback_active=playing,
                        archive_pending=len(self._archive_pending),
                        archive_session=str(self.session_archive.folder) if self.session_archive is not None else None,
                        archive_sealed=self._archive_sealed(),
                        archive_error=self.archive_error,
                        recording_safe=not self.mic_on and self._active is None and not self._pending_jobs and not playing and not self._archive_pending and not self.archive_error,
                        mic=self.mic_name(), mic_index=self.mic_index(), mics=list(self._mics),
                        mic_devices=[dict(device) for device in self._mic_devices],
                        mic_identity=dict(self._mic_identity) if self._mic_identity is not None else None,
                        server_ok=self._server_ok, level=self.front.level,
                        transcript=[m['text'] for m in self._messages if m['text']],
                        transcript_revision=self.transcript_revision,
                        assistant_partial=partial, heard_at=self.heard_at,
                        gen_start=self.gen_start, first_audio_at=self.first_audio_at,
                        eta_first_audio_ms=eta, eta_budget_ms=round(budget * 1000),
                        interrupt_candidate=self.interrupt_candidate, interrupted_at=self.interrupted_at,
                        last_ttft=self.last_ttft, last_audio_s=self.last_audio_s,
                        last_turn_s=self.last_turn_s, turns=self.turns,
                        error=self.error, warning='; '.join(w for w in [self.warning, self.archive_error, self._agent_search_warning, getattr(self.agent, 'configuration_error', None)] if w) or None,
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
        elif self.path == '/api/traces':
            self._send(json.dumps(self.server.assistant.traces(20)))
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
            if self.path == '/api/archive/seal':
                desired = json.loads(raw)
                if not isinstance(desired, dict) or set(desired) != {'expected_session'}:
                    raise ValueError('Expected recording session identity')
                try:
                    result = assistant.seal_archive(desired['expected_session'])
                except (RuntimeError, OSError) as exc:
                    self._send(json.dumps({'error': str(exc)}), 409)
                    return
                self._send(json.dumps(result))
                return
            elif self.path == '/api/mic':
                if raw.strip():
                    desired = json.loads(raw)
                    if not isinstance(desired, dict):
                        raise ValueError('Expected microphone settings')
                    assistant.set_mic_enabled(desired['enabled'])
                else:
                    assistant.toggle_mic()
            elif self.path == '/api/interrupt':
                assistant.interrupt()
            elif self.path == '/api/pause':
                desired = json.loads(raw or b'{}')
                if not isinstance(desired, dict) or set(desired) != {'milliseconds'}:
                    raise ValueError('Expected pause setting')
                try:
                    actual = assistant.set_end_silence(desired['milliseconds'])
                except RuntimeError as exc:
                    self._send(json.dumps({'error': str(exc)}), 409)
                    return
                self._send(json.dumps({'ok': True, 'milliseconds': actual}))
                return
            elif self.path == '/api/agent':
                desired = json.loads(raw or b'{}')
                if not isinstance(desired, dict) or 'mode' not in desired or not set(desired) <= {'mode', 'model'}:
                    raise ValueError('Expected agent mode settings')
                result = assistant.set_agent_mode(desired['mode'], desired.get('model'))
                self._send(json.dumps(result))
                return
            elif self.path == '/api/search-provider':
                desired = json.loads(raw or b'{}')
                if not isinstance(desired, dict) or set(desired) != {'provider'}:
                    raise ValueError('Expected a search provider')
                result = assistant.set_search_provider(desired['provider'])
                self._send(json.dumps(result))
                return
            elif self.path == '/api/persona':
                desired = json.loads(raw or b'{}')
                if not isinstance(desired, dict) or set(desired) != {'persona'}:
                    raise ValueError('Expected persona setting')
                result = assistant.set_persona(desired['persona'])
                self._send(json.dumps(result))
                return
            elif self.path == '/api/voice':
                desired = json.loads(raw or b'{}')
                result = assistant.set_voice_params(desired)
                self._send(json.dumps(result))
                return
            elif self.path == '/api/prompt':
                desired = json.loads(raw or b'{}')
                if not isinstance(desired, dict) or set(desired) != {'text'}:
                    raise ValueError('Expected prompt text')
                job = assistant.submit_text(desired['text'])
                self._send(json.dumps(dict(ok=True, turn_id=job.id)))
                return
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
        except ArchiveSealedError as exc:
            self._send(json.dumps({'error': str(exc)}), 409)
        except (ValueError, KeyError):
            self._send('{"error":"Invalid request"}', 400)

    def log_message(self, *args):
        pass


def main():
    asr_url = os.getenv('LOCALVOICE_ASR_SERVER')
    from pathlib import Path
    log_path = Path(__file__).resolve().parents[1] / 'outputs' / 'logs' / 'app.jsonl'
    trace_path = Path(__file__).resolve().parents[1] / 'outputs' / 'logs' / 'traces.jsonl'
    archive_path = os.getenv('LOCALVOICE_RECORD_SESSION')
    archive = None
    if archive_path:
        from session_archive import SessionArchive
        archive = SessionArchive(Path(archive_path))
        with (archive.folder/'provenance.json').open('x', encoding='utf-8') as stream:
            json.dump(dict(source_files_sha256=source_fingerprint(),
                           backend=os.getenv('LOCALVOICE_SERVER', 'http://127.0.0.1:8080'),
                           scope='App source and endpoint; native build/model identity must come from the experiment protocol.'), stream, indent=2)
    agent = None
    agent_mode = os.getenv('LOCALVOICE_AGENT', '').strip().lower()
    if agent_mode in ('opencode', 'opencode-go', 'opencode-zen'):
        from agent import AgentClient
        provider = 'zen' if agent_mode == 'opencode-zen' else 'go'
        key_name = 'OPENCODE_ZEN_API_KEY' if provider == 'zen' else 'OPENCODE_GO_API_KEY'
        search_key = os.getenv('BRAVE_SEARCH_API_KEY')
        search_provider = os.getenv('LOCALVOICE_SEARCH', '').strip().lower() or ('brave' if search_key else 'ddg')
        agent = AgentClient(os.getenv(key_name), provider=provider,
                            model=os.getenv('LOCALVOICE_AGENT_MODEL', 'kimi-k3'),
                            search_key=search_key,
                            search_provider=search_provider,
                            local_tools=True,
                            memory_path=trace_path.parent / 'agent_memory.json')
    elif agent_mode:
        raise ValueError('LOCALVOICE_AGENT must be opencode-go or opencode-zen')
    backend = os.getenv('LOCALVOICE_BACKEND', 'lfm2').strip().lower()
    if backend == 'lfm3':
        from lfm3_client import LFM3Client
        client_type = LFM3Client
    elif backend == 'lfm2':
        client_type = ServerClient
    else:
        raise ValueError('LOCALVOICE_BACKEND must be lfm2 or lfm3')
    client = client_type(os.getenv('LOCALVOICE_SERVER', 'http://localhost:8080'))
    assistant = Assistant(client=client, asr_client=client_type(asr_url) if asr_url else None, diagnostics=Diagnostics(log_path), start_muted=os.getenv("LOCALVOICE_START_MUTED") == "1", session_archive=archive, agent=agent, trace_path=trace_path)
    server = ThreadingHTTPServer(('127.0.0.1', UI_PORT), Handler)
    server.assistant = assistant
    assistant.start()
    print(f'Turnwave on http://localhost:{UI_PORT}', flush=True)
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
