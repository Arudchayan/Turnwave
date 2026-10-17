"""Drain an explicitly selected recorded test before stopping its recorder.

Never interrupts playback or starts devices. Loopback is not proof that all
generated speech was archived. Failed attempts remain available for inspection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import struct
from urllib.request import Request, urlopen
from urllib.parse import urlparse
import uuid
import wave

try:
    from .checkpoint_experiment import checkpoint, verify_checkpoint
except ImportError:
    from checkpoint_experiment import checkpoint, verify_checkpoint


def _state(base):
    with urlopen(base + '/api/state', timeout=2) as response:
        return json.load(response)


def _mute(base):
    # Desired-state control stays muted even if a simultaneous user action
    # already switched the microphone off after our preceding observation.
    request = Request(base + '/api/mic', data=b'{"enabled":false}', method='POST',
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=2) as response:
        return json.load(response)


def _seal(base, expected_session):
    request = Request(base + '/api/archive/seal',
                      data=json.dumps({'expected_session': expected_session}).encode('utf-8'),
                      method='POST', headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=2) as response:
        return json.load(response)


def _session(folder):
    folder = Path(folder).resolve(strict=True)
    if not folder.is_dir():
        raise ValueError('Select the exact existing experiment or audio folder')
    candidates = [p for p in (folder / 'audio', folder) if (p / 'audio-session.json').is_file()]
    if len(candidates) != 1:
        raise ValueError('Exactly one recording session must exist in this folder or its audio child')
    audio = candidates[0].resolve()
    if audio != folder and audio.parent != folder:
        raise ValueError('Recording session must be directly inside the selected folder')
    metadata = json.loads((audio / 'audio-session.json').read_text(encoding='utf-8'))
    devices = metadata.get('devices')
    if not isinstance(devices, dict) or set(devices) != {'microphone', 'speaker'}:
        raise ValueError('Not an explicit microphone and speaker recording session')
    return folder, audio, metadata


def _verify_audio(audio, metadata):
    finished = json.loads((audio / 'audio-finished.json').read_text(encoding='utf-8'))
    counts = finished.get('frames', {})
    if set(counts) != set(metadata['devices']):
        raise ValueError('Recorder final frame roster is incomplete')
    for track, device in metadata['devices'].items():
        with wave.open(str(audio / (track + '.wav')), 'rb') as reader:
            if (reader.getnframes() != counts[track] or reader.getsampwidth() != 2
                    or reader.getnchannels() != int(device['maxInputChannels'])
                    or reader.getframerate() != int(device['defaultSampleRate'])):
                raise ValueError('Final WAV header differs from recorder metadata: ' + track)
            frames = reader.getnframes()
            if frames <= 0:
                raise ValueError('Recording track is empty: ' + track)
            actual = 0
            while True:
                block = reader.readframes(65536)
                if not block:
                    break
                actual += len(block)
            if actual != frames * reader.getnchannels() * reader.getsampwidth():
                raise ValueError('Final WAV payload is truncated: ' + track)
    return finished


def _verify_turn_archive(folder, state):
    if state.get('archive_error'):
        raise RuntimeError('Per-turn archive failed: ' + str(state['archive_error']))
    selected = state.get('archive_session')
    if selected is None:
        return
    archive = Path(selected).resolve(strict=True)
    if not archive.is_dir() or (archive != folder and folder not in archive.parents):
        raise ValueError('Per-turn archive must be inside the selected experiment; checkpoint external archives separately')
    session = json.loads((archive / 'session.json').read_text(encoding='utf-8'))
    if session.get('version') != 1 or session.get('opt_in_recording') is not True:
        raise ValueError('Per-turn archive opt-in session metadata is missing or unsupported')
    for turn in sorted(archive.glob('turn-*')):
        if not turn.is_dir() or turn.resolve().parent != archive:
            raise ValueError('Turn archive must be a direct owned directory')
        status = json.loads((turn / 'status.json').read_text(encoding='utf-8'))
        if (status.get('checkpoint_safe') is not True or status.get('state') != 'finalized'
                or status.get('generation_finished') is not True or status.get('playback_finished') is not True):
            raise ValueError('Per-turn archive is still incomplete: ' + turn.name)
        manifest = json.loads((turn / 'manifest.json').read_text(encoding='utf-8'))
        roster = manifest.get('files', {})
        if manifest.get('version') != 1 or not {'input.wav', 'turn.json', 'generation.json', 'playback.json'} <= set(roster):
            raise ValueError('Per-turn archive manifest is incomplete: ' + turn.name)
        actual = {p.name for p in turn.iterdir() if p.is_file() and p.name not in {'status.json', 'manifest.json'}}
        if actual != set(roster):
            raise ValueError('Per-turn archive file roster differs: ' + turn.name)
        for name, expected in roster.items():
            path = turn / name
            if Path(name).name != name or path.resolve().parent != turn.resolve():
                raise ValueError('Unsafe turn archive file name')
            data = path.read_bytes()
            if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
                raise ValueError('Per-turn archive hash differs: ' + turn.name + '/' + name)
        generation = json.loads((turn / 'generation.json').read_text(encoding='utf-8'))
        turn_metadata = json.loads((turn / 'turn.json').read_text(encoding='utf-8'))
        generated = generation.get('generated_samples')
        if type(generated) is not int or generated < 0:
            raise ValueError('Invalid generated sample count: ' + turn.name)
        if bool(generated) != ('generated.wav' in roster):
            raise ValueError('Generated audio roster differs: ' + turn.name)
        if bool(generated) != ('generated.f32' in roster):
            raise ValueError('Exact generated float audio is missing or unexpected: ' + turn.name)
        if generated:
            with (turn / 'generated.f32').open('rb') as stream:
                if (turn / 'generated.f32').stat().st_size != generated * 4:
                    raise ValueError('Exact generated float audio length differs: ' + turn.name)
                for block in iter(lambda: stream.read(65536), b''):
                    if not all(math.isfinite(sample[0]) for sample in struct.iter_unpack('<f', block)):
                        raise ValueError('Exact generated float audio has nonfinite samples: ' + turn.name)
        if ('request.json' not in roster and not
                (generation.get('cancelled') is True and generated == 0 and not generation.get('text'))):
            raise ValueError('Request payload is missing: ' + turn.name)
        for name in ['input.wav'] + (['generated.wav'] if generated else []):
            if name == 'input.wav' and (turn / name).stat().st_size == 0 and turn_metadata.get('test') is True:
                continue  # Speaker checks have no captured user input.
            with wave.open(str(turn / name), 'rb') as reader:
                count, width, channels = reader.getnframes(), reader.getsampwidth(), reader.getnchannels()
                if name == 'generated.wav' and (count != generated or width != 2 or channels != 1 or reader.getframerate() != 24000):
                    raise ValueError('Generated WAV header differs: ' + turn.name)
                actual_bytes = 0
                while block := reader.readframes(65536):
                    actual_bytes += len(block)
                if actual_bytes != count * width * channels:
                    raise ValueError('Archived WAV is truncated: ' + turn.name + '/' + name)


def _quiescent(state):
    return (state.get('recording_safe') is True
            and state.get('pending_turns') == 0
            and state.get('playback_active') is False
            and state.get('archive_pending') == 0
            and state.get('active_turn') is False and state.get('mic_on') is False
            and state.get('status') in {'listening', 'error'}
            and state.get('assistant_partial') == '')


def finish(folder, app_base='http://127.0.0.1:8000', timeout_s=30, *,
           read_state=None, mute_mic=None, clock=None, sleep=None,
           make_checkpoint=None, check_checkpoint=None, seal_archive=None):
    """Return a verified checkpoint receipt, or raise with an archived error.

    Injectable helpers allow hardware-free tests. Timeout is shared by playback,
    recorder and observer waits. An unavailable app never implies safe playback.
    """
    folder, audio, metadata = _session(folder)  # No writes before validation.
    parsed = urlparse(app_base)
    if parsed.scheme != 'http' or parsed.hostname not in {'localhost', '127.0.0.1'} or parsed.path not in {'', '/'} or parsed.query or parsed.fragment or parsed.username:
        raise ValueError('App address must be a local HTTP base URL')
    if not 0 < timeout_s <= 600:
        raise ValueError('Timeout must be between 0 and 600 seconds')
    app_base = app_base.rstrip('/')
    read_state = read_state or (lambda: _state(app_base))
    mute_mic = mute_mic or (lambda: _mute(app_base))
    seal_archive = seal_archive or (lambda expected: _seal(app_base, expected))
    clock, sleep = clock or time.monotonic, sleep or time.sleep
    make_checkpoint = make_checkpoint or checkpoint
    check_checkpoint = check_checkpoint or verify_checkpoint
    attempt = folder / ('finalization-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex[:8])
    attempt.mkdir(exist_ok=False)
    deadline = clock() + timeout_s

    def record(phase, **values):
        event = dict(phase=phase, utc=datetime.now(timezone.utc).isoformat(), **values)
        with (attempt / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event) + '\n')
        (attempt / 'status.json').write_text(json.dumps(event, indent=2), encoding='utf-8')

    def wait_tick(message):
        if clock() >= deadline:
            raise TimeoutError(message)
        sleep(min(.2, max(0, deadline-clock())))

    try:
        record('waiting-for-playback', audio_folder=str(audio))
        state = read_state()
        if state.get('archive_error'):
            raise RuntimeError('Per-turn archive failed: ' + str(state['archive_error']))
        if state.get('mic_on') is True:
            mute_mic()
        while True:
            state = read_state()
            if state.get('archive_error'):
                raise RuntimeError('Per-turn archive failed: ' + str(state['archive_error']))
            if _quiescent(state):
                break
            wait_tick('Playback did not drain; recorder was not stopped')
        _verify_turn_archive(folder, state)
        expected_session = state.get('archive_session')
        ack = seal_archive(expected_session)
        if (not isinstance(ack, dict) or ack.get('ok') is not True
                or ack.get('archive_sealed') is not True
                or 'archive_session' not in ack or ack['archive_session'] != expected_session):
            raise RuntimeError('App did not acknowledge the exact recording session seal')
        sealed_state = read_state()
        if (not _quiescent(sealed_state) or sealed_state.get('archive_error')
                or sealed_state.get('archive_sealed') is not True
                or sealed_state.get('archive_session') != expected_session):
            raise RuntimeError('App recording seal is unverified; recorder was not stopped')
        state = sealed_state
        record('recording-stop', app_state=state)
        (audio / 'STOP_AUDIO').touch(exist_ok=True)
        while not (audio / 'audio-finished.json').is_file():
            wait_tick('Recorder did not finish normally; no checkpoint created')
        finished = _verify_audio(audio, metadata)
        observer = folder / 'experiment.json'
        if observer.is_file():
            experiment = json.loads(observer.read_text(encoding='utf-8'))
            if experiment.get('watch_minutes', 0) > 0:
                record('observer-stop')
                (folder / 'STOP').touch(exist_ok=True)
                while not (folder / 'finished.json').is_file():
                    wait_tick('Observer did not finish normally; no checkpoint created')
        # Ad-hoc observers have no standard owned-process stop protocol. Never
        # signal a PID or write its STOP from an ambiguous historical record.
        # Require its explicit completion marker before copying this session.
        if (folder / 'observer').is_dir() or (folder / 'observer-process.json').is_file():
            while not (folder / 'observer' / 'finished.json').is_file():
                wait_tick('Observer completion is unverified; finish the owned observer explicitly before checkpointing')
        final_state = read_state()
        if not _quiescent(final_state) or final_state.get('archive_sealed') is not True:
            raise RuntimeError('App became active during finalization; no checkpoint created')
        if final_state.get('archive_session') != state.get('archive_session'):
            raise RuntimeError('App archive session changed during finalization; no checkpoint created')
        _verify_turn_archive(folder, final_state)
        record('recording-verified', frames=finished['frames'], checkpoint='pending',
               note='Closed WAV payloads match recorder metadata. This does not establish speech completeness or intelligibility.')
        destination = Path(make_checkpoint(folder))
        checked = check_checkpoint(destination)
        after_copy = read_state()
        if (not _quiescent(after_copy) or after_copy.get('archive_error')
                or after_copy.get('archive_sealed') is not True
                or after_copy.get('archive_session') != final_state.get('archive_session')):
            raise RuntimeError('App changed during checkpoint; saved copy is historical, finalization is incomplete')
        copied = json.loads((destination / 'checkpoint.json').read_text(encoding='utf-8'))
        expected = {row['path']: row for row in copied['files']}
        current = {p.relative_to(folder).as_posix(): p for p in folder.rglob('*') if p.is_file()}
        if set(current) != set(expected):
            raise RuntimeError('Recording session file roster changed during checkpoint; finalization is incomplete')
        for relative, path in current.items():
            row = expected[relative]
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            if path.stat().st_size != row['bytes'] or digest.hexdigest() != row['sha256']:
                raise RuntimeError('Recording session changed during checkpoint: ' + relative)
        receipt = dict(phase='verified-checkpoint', attempt=str(attempt), checkpoint=str(destination), files=checked)
        # Keep the checkpointed source unchanged. This receipt belongs outside
        # its experiment copy and does not affect that copy's verified manifest.
        (destination / 'finalization-receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        return receipt
    except Exception as exc:
        record('error', error=str(exc), exception=type(exc).__name__)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--app-base', default='http://127.0.0.1:8000')
    parser.add_argument('--timeout-s', type=float, default=30)
    args = parser.parse_args()
    print(json.dumps(finish(args.folder, args.app_base, args.timeout_s), indent=2))


if __name__ == '__main__':
    main()
