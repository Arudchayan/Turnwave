# LocalVoice

An experimental local voice assistant for Windows and WSL2, built around
LiquidAI LFM2.5-Audio-1.5B and the Liquid audio server.

The app keeps listening while a turn-based speech model generates a reply.
Speaking over a reply can cancel playback and start a replacement turn. This
is application-level interruption handling, not a native full-duplex model.

## Status

Capture, playback, cancellation, error recovery and optional recording have
regression tests. Natural pauses, speaker echo, exact instruction following
and model memory still have known failures. See
[known limitations](docs/KNOWN_LIMITATIONS.md) before relying on it.

## Prerequisites

- Windows with working input/output devices and WSL2 Ubuntu.
- Python 3.12 for the application. Node.js for browser-script tests.
- A compatible Liquid audio native runner and separately obtained model files.
- Suitable hardware for the selected runner; performance depends on hardware.

The local runner integration targets upstream source revision `ec9d1fdd9` from
llama.cpp PR #18641. Native toolchains and models are external prerequisites,
not bundled installers. Review their licenses and compatibility independently.
The scripts in `src/` include historical rootless CUDA build helpers; inspect
those before running them. They can download dependencies and modify a build.

## Setup and run

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

Place model files in `models/LFM2.5-Audio-1.5B-GGUF/`:

- `LFM2.5-Audio-1.5B-Q8_0.gguf` or `LFM2.5-Audio-1.5B-Q4_0.gguf`
- `mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf`
- `vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf`
- `tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf`

The launcher prefers a local runner under `~/liquid-build/build/bin` inside
WSL. `LOCALVOICE_GPU_RUNNER` can select another runner directory; a prebuilt
runner may instead be placed under `models/runner/llama-liquid-audio-ubuntu-x64`.
Keep the required shared libraries with the runner. Stop the server before
applying the reviewed patches using `src/apply_runner_fixes.sh`.

From the repository root, start the backend:

```powershell
wsl -d Ubuntu bash src/server_wsl.sh
```

After it prints `Server ready`, use a second terminal:

```powershell
$env:LOCALVOICE_START_MUTED = "1"
python src/app.py
```

Open **http://127.0.0.1:8000**, select the input device and unmute when ready.
Headphones are the more predictable starting setup. Muted startup still opens
an input stream but does not submit microphone turns.

`LOCALVOICE_SERVER` changes the backend URL (default local port 8080).
`LOCALVOICE_CHAT_SYSTEM` changes the chat instruction. Optional transcription
uses `LOCALVOICE_ASR_SERVER` and requires a genuinely separate model context.
The app and backend test entry points reserve exclusive context ownership;
stop the app before running tests that use its backend, even while muted.

## Tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python scripts/check_publication.py
```

The unit/integration suite uses generated fixtures and simulated devices.
No model weights, private recordings or live audio devices are required.
Node.js enables browser-script checks, which otherwise skip explicitly.
Optional CPU recognizer evaluations use `requirements-eval.txt` and separately
obtained local weights. Exploratory tools under `src/synth/` have additional
prerequisites; they are not the default test suite.

### Offline content benchmark preparation

`src/synth/content_benchmark.py` prepares local, already obtained benchmark WAVs
and exports grading inputs. It does not download datasets, start a model, run
judges or play audio. A manifest records `version: 1`, the dataset name, its
40-character source revision, and a nonempty `cases` list. Each case has a safe
unique `id`, `subset` (`ifeval` or `openbookqa`), relative `audio` path, input
`sha256`, and original scorer `metadata`. Inputs must be nonempty 16 kHz mono
PCM16 WAVs. Check dataset terms and keep audio/manifests under ignored local data.

```powershell
python src/synth/content_benchmark.py prepare data/benchmark/manifest.json outputs/benchmark-prepared
python src/synth/content_benchmark.py export outputs/benchmark-run outputs/benchmark-grades --asr outputs/benchmark-asr.json
```

Every destination must be new. There is no inference CLI: the Python API takes
an explicit client, with exclusive backend test ownership for actual requests.
Stop the app before model tests, even while muted. Preserve the exact generated
float32 audio and model text separately. Independent ASR transcripts must identify
their recognizer and match the generated-audio hash; missing transcripts never
fall back to model text. Grade exports retain failed cases for review rather
than treating them as ordinary completed answers. Run upstream VoiceBench
scoring separately after inspecting these artifacts.

The upstream multiple-choice scorer can randomly guess when it cannot extract
an answer, including from a blank failed response. Keep any seeded upstream
score separate from acceptance: require an eligible completed case and a
parseable correct answer. Scores from a small adapted slice are not leaderboard
results, and speech-transcript errors still require listening review.

This adapter is a content-test boundary, not a full benchmark result or an
end-to-end voice latency measurement. It bypasses microphone turn detection;
pause, echo, interruption, room audio and human intelligibility require separate
evaluation.

## Recording and privacy

Normal launches write private local diagnostics under `outputs/logs/`.
Audio archiving is opt-in. See [data handling](docs/PRIVACY.md) and
[recorded sessions](docs/RECORDED_SESSIONS.md) before recording participants.
A successful output write is distinct from proof that every sample was heard.

Models, audio, outputs, credentials, local agent notes and historical research
reports are excluded from publication. Do not expose the UI/backend publicly;
they are local services without multi-user authentication.

## Repository guide

- `src/`: application, audio pipeline and native runner helpers.
- `src/synth/`: optional synthetic/recorded evaluation and research tools.
- `tests/`: automated regression tests with generated fixtures.
- `patches/`: local changes against the pinned external native runner.
- `docs/`: architecture, known limitations, privacy and operational guidance.

Read [architecture](docs/ARCHITECTURE.md), [contributing](CONTRIBUTING.md),
[security](SECURITY.md), and [publication guidance](docs/PUBLICATION.md).
Third-party model and runner terms are independent of this application.

## Licensing

The original application code is currently unlicensed by the maintainer's
choice; no open-source license has been granted. External dependencies,
model weights, native runner sources and derived patches retain their own terms.
