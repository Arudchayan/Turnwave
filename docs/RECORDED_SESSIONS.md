# Recording experiments

On Windows, install the optional raw recorder dependency with
`python -m pip install -r requirements-recording.txt`.

Per-turn archiving is opt-in. Normal app launches do not create audio archives.
To record a session, set `LOCALVOICE_RECORD_SESSION` to a
new `turns` directory inside that experiment before launching the app. The
directory must not already exist; a reused path fails before opening devices.
Use `LOCALVOICE_START_MUTED=1` for the controlled startup.

The separate physical recorder saves microphone and system output to the
experiment's `audio` directory. Per-turn archives contain:

| File | Meaning |
| --- | --- |
| input.wav | Exact 16 kHz request WAV submitted by the app, including its capture processing |
| request.json | Exact request field values, including reset flag and modalities; JSON serialization is not the original HTTP byte stream |
| generated.f32 | Authoritative decoded mono float32 samples, little endian, 24 kHz |
| generated.wav | PCM16 preview; native PCM16 samples round-trip correctly |
| generation.json | Reported text, error/cancellation, finish reason and timings |
| playback.json | Source samples whose blocking output write succeeded, separate from generated samples; buffered audio may still be interrupted before audibility |
| status.json / manifest.json | Lifecycle state and file hashes; finalized only after generation and playback finish |

A speaker check has no captured user input, so its explicitly marked test
archive contains an empty input file. A turn cancelled before its request may
have no request payload or generated audio. Both cases are distinguished from
missing evidence in a normal requested turn.

Audio headers and float samples flush incrementally; closed generation files
are synced before final metadata. Tests cover readable partial files after
abrupt process exit. This is not a guarantee against every power, filesystem
or disk failure. Storage errors remain visible and prevent verified session
finalization; voice playback can continue.

Source hashes and backend endpoint are saved in session provenance. The
experiment protocol must separately retain actual native build/model hashes;
an endpoint URL does not establish model identity. Background startup warmup
is not a conversational turn archive.

## Finish without cutting off the recording

Keep the app and recorder running and invoke from the repository root:

```powershell
python -m src.synth.finish_recorded_test outputs/experiments/<new-test> --timeout-s 90
```

The helper mutes capture without cancelling the reply, waits for generation,
playback and archive completion, then asks the app to seal the exact session.
The app must acknowledge that it is muted, fully drained and sealed before the
helper signals the selected recorder to stop. A sealed app rejects microphone
enable, new audio submissions and speaker checks. Its per-turn archive retains
a durable seal so later activity cannot append to the finalized source.
It verifies closed physical WAVs and finalized per-turn files before copying
the whole experiment. App state and source hashes are checked again after
copying. The success receipt is `finalization-receipt.json` beside the backup's
checkpoint manifest, outside its immutable experiment snapshot.

Timeouts, incomplete archives, external archive paths and changed files produce
an explicit failed attempt rather than a verified session claim. Playback-wait
timeouts leave the recorder running. Missing or mismatched seal acknowledgements
also fail before recorder stop. Failures after sealing do not reopen the session;
failures after recorder stop preserve closed files without claiming a complete
checkpoint. An older running app without seal support must be updated before
using this finalizer; changing files on disk does not update the loaded app.

Use the standard observer protocol (`experiment.json` with `watch_minutes`,
STOP and finished.json). An ad-hoc observer directory needs explicit completion
before finalization; ambiguous processes are never killed automatically.

The helper does not extend an independent recorder's deadline or repair audio
already missed. A finalized snapshot is not proof of intelligibility, complete
physical coverage, correct answers or an off-device backup. Recording again
requires a new app session with a fresh archive directory; an already sealed
experiment is not reopened for another test. Keep each recorded experiment in its own directory.
