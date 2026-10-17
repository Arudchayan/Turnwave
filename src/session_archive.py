"""Opt-in per-turn evidence, separate from physical recording and backups.

Only finalized archives should be checkpointed. Generated audio is mono at
24 kHz; it records model output, not a claim about what reached the speaker.
The WAV header is refreshed on every chunk so interrupted files remain readable.
Raw little-endian float32 is authoritative; the WAV is a rounded PCM16 preview.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import threading

import numpy as np


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def _exclusive_json(path, value):
    data = _json_bytes(value)
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


class SessionArchive:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        _exclusive_json(self.folder / 'session.json', {
            'version': 1, 'opt_in_recording': True,
            'generated_audio': 'Authoritative little-endian float32 mono 24000 Hz; PCM16 WAV preview; distinct from played audio',
            'checkpoint_policy': 'Only checkpoint after all turns have finalized.',
        })

    def begin_turn(self, turn_id, wav_bytes, first, test=False):
        name = str(turn_id)
        if not name or name in {'.', '..'} or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('Turn id must contain only letters, numbers, underscore or hyphen')
        return TurnArchive(self.folder / ('turn-' + name), name, wav_bytes, first, test)


class TurnArchive:
    def __init__(self, folder, turn_id, wav_bytes, first, test):
        self.folder = folder
        self._lock = threading.RLock()
        self._generation = None
        self._playback = None
        self._audio = None
        self._raw_audio = None
        self._samples = 0
        # No metadata may fail serialization after creating an otherwise empty turn.
        initial = {'turn_id': turn_id, 'first': bool(first), 'test': bool(test)}
        payload = bytes(wav_bytes)
        folder.mkdir(exist_ok=False)
        with (folder / 'input.wav').open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _exclusive_json(folder / 'turn.json', initial)
        self._save_status()

    def record_request(self, payload):
        """Preserve the payload without normalizing or redacting its fields."""
        with self._lock:
            if self._generation is not None:
                raise RuntimeError('Generation already finalized')
            _exclusive_json(self.folder / 'request.json', payload)

    def _header(self):
        size = self._samples * 2
        return (b'RIFF' + struct.pack('<I', 36 + size) + b'WAVEfmt ' +
                struct.pack('<IHHIIHH', 16, 1, 1, 24000, 48000, 2, 16) +
                b'data' + struct.pack('<I', size))

    def write_audio(self, samples):
        pcm = np.asarray(samples, dtype=np.float32)
        if pcm.ndim != 1 or not np.isfinite(pcm).all():
            raise ValueError('Expected finite mono audio')
        raw = pcm.astype('<f4', copy=False).tobytes()
        data = np.clip(np.rint(np.clip(pcm, -1, 1) * 32768), -32768, 32767).astype('<i2').tobytes()
        with self._lock:
            if self._generation is not None:
                raise RuntimeError('Generation already finalized')
            if not len(pcm):
                return
            if self._samples + len(pcm) > (0xffffffff - 36) // 2:
                raise ValueError('Generated WAV exceeds RIFF capacity')
            if self._audio is None:
                self._audio = (self.folder / 'generated.wav').open('xb')
                self._audio.write(self._header())
                self._raw_audio = (self.folder / 'generated.f32').open('xb')
            self._raw_audio.write(raw)
            self._raw_audio.flush()
            self._audio.seek(0, os.SEEK_END)
            self._audio.write(data)
            self._samples += len(pcm)
            self._audio.seek(0)
            self._audio.write(self._header())
            self._audio.flush()

    def finish_generation(self, turn):
        with self._lock:
            if self._generation is not None:
                raise RuntimeError('Generation already finalized')
            def field(name, default=None):
                return turn.get(name, default) if isinstance(turn, dict) else getattr(turn, name, default)
            cancelled = field('cancel', False)
            cancelled = cancelled.is_set() if hasattr(cancelled, 'is_set') else bool(cancelled)
            metadata = {key: field(key) for key in (
                'text', 'error', 'finish_reason', 't_req', 'ttft', 'first_event',
                'queue_s', 'headers_s', 'audio_frames', 'audio_samples')}
            metadata.update(cancelled=cancelled, generated_samples=self._samples,
                            audio_present=bool(self._samples),
                            generated_audio_format={'authoritative': 'generated.f32',
                                'dtype': '<f4', 'channels': 1, 'sample_rate': 24000,
                                'samples': self._samples, 'preview': 'generated.wav'})
            _json_bytes(metadata)  # Validate before closing and committing.
            if self._audio is not None:
                self._audio.flush()
                os.fsync(self._audio.fileno())
                self._audio.close()
                self._audio = None
            if self._raw_audio is not None:
                self._raw_audio.flush()
                os.fsync(self._raw_audio.fileno())
                self._raw_audio.close()
                self._raw_audio = None
            _exclusive_json(self.folder / 'generation.json', metadata)
            self._generation = metadata
            self._save_manifest()
            self._save_status()

    def finish_playback(self, played_samples, cancelled, error=None):
        with self._lock:
            if self._playback is not None:
                raise RuntimeError('Playback already finalized')
            if isinstance(played_samples, bool) or int(played_samples) != played_samples or played_samples < 0:
                raise ValueError('Invalid played sample count')
            metadata = {'played_samples': int(played_samples), 'cancelled': bool(cancelled), 'error': error}
            _exclusive_json(self.folder / 'playback.json', metadata)
            self._playback = metadata
            if self._generation is not None:
                self._save_manifest()
            self._save_status()

    def status(self):
        with self._lock:
            final = self._generation is not None and self._playback is not None
            return {'folder': str(self.folder), 'state': 'finalized' if final else 'incomplete',
                    'generation_finished': self._generation is not None,
                    'playback_finished': self._playback is not None,
                    'checkpoint_safe': final,
                    'generated_samples': self._samples,
                    'played_samples': None if self._playback is None else self._playback['played_samples'],
                    'generation': self._generation, 'playback': self._playback}

    def _replace_json(self, name, value):
        # Only derived status/roster files are replaced; evidence is exclusive.
        temporary = self.folder / (name + '.tmp')
        with temporary.open('wb') as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.folder / name)

    def _save_status(self):
        self._replace_json('status.json', self.status())

    def _save_manifest(self):
        # Never hash/copy the actively written WAV. Mutable derived files excluded.
        roster = {}
        for path in sorted(self.folder.iterdir()):
            if path.name in {'manifest.json', 'status.json'} or path.suffix == '.tmp':
                continue
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            roster[path.name] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
        self._replace_json('manifest.json', {'version': 1, 'files': roster})
