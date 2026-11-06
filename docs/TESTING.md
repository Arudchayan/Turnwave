# Testing Turnwave

This is the real-device protocol for the floor and end-of-turn changes. The
offline suites are already green; this runbook is for the part they cannot
prove: microphone, speaker, room, echo and human timing.

## 0. Fast checks (no devices)

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python -m src.synth.floor_replay --out outputs/floor_replay.json
python -m src.synth.floor_acoustic --out outputs/floor_acoustic.json
python scripts/eot_recompute.py
python scripts/eot_holdout.py
python scripts/check_publication.py --worktree
```

Expected: the test suite passes; the floor replay reports the reversible candidate; the
acoustic replay reports the known negative backchannel transfer; the EOT
held-out metrics match the published summary.

## 1. Environment

- Windows with working input/output devices and WSL2 Ubuntu.
- Python 3.12 environment with `requirements-dev.txt`.
- One backend at a time:
  - **LFM2.5 path:** the pinned Liquid runner and its model files.
  - **LFM3 path:** llama.cpp b11960+ and `LFM3-Audio-2B-Q4_K_M.gguf` plus
    `mmproj-LFM3-Audio-2B-Q8_0.gguf`.
- Headphones for the first session; speakers only after the headphone pass.
- Start the app muted, then unmute deliberately:

```powershell
$env:LOCALVOICE_START_MUTED = "1"
python src/app.py
```

Do not run the backend-sharing tests while the app is open, even muted.
The app's **Guided testing** link opens a local checklist at `/testing`. It
stores each comparison arm separately in the browser and can export results.
The configuration shown there must be applied before restarting the app.

## 2. Arms

Run the same script with the current default first (unset `LOCALVOICE_FLOOR_MODE`
and `LOCALVOICE_EOT_MODEL`, and use HTTP), then in the four experimental arms.
Keep one variable between adjacent experimental arms.

| Arm | Floor | EOT | Backend |
| --- | --- | --- | --- |
| A | `hard` | off | LFM2.5 HTTP |
| B | `reversible` | off | LFM2.5 HTTP |
| C | `reversible` | Smart Turn, 1 s | LFM2.5 HTTP |
| D | `reversible` | Smart Turn, 1 s | LFM3 realtime (optional) |

Start the LFM2.5 server:

```powershell
wsl -d Ubuntu bash src/server_wsl.sh
```

Start the LFM3 server (separate terminal):

```powershell
wsl -d Ubuntu bash -c 'LOCALVOICE_MODEL_DIR="$PWD/models/LFM3-Audio-2B-GGUF" LM=LFM3-Audio-2B-Q4_K_M.gguf MM=mmproj-LFM3-Audio-2B-Q8_0.gguf bash src/server_lfm3_wsl.sh'
```

App environment per arm:

```powershell
# A
$env:LOCALVOICE_BACKEND = "lfm2"
$env:LOCALVOICE_TRANSPORT = "http"
$env:LOCALVOICE_FLOOR_MODE = "hard"
Remove-Item Env:LOCALVOICE_EOT_MODEL -ErrorAction SilentlyContinue

# B
$env:LOCALVOICE_BACKEND = "lfm2"
$env:LOCALVOICE_TRANSPORT = "http"
$env:LOCALVOICE_FLOOR_MODE = "reversible"
Remove-Item Env:LOCALVOICE_EOT_MODEL -ErrorAction SilentlyContinue

# C
$env:LOCALVOICE_BACKEND = "lfm2"
$env:LOCALVOICE_TRANSPORT = "http"
$env:LOCALVOICE_FLOOR_MODE = "reversible"
$env:LOCALVOICE_EOT_MODEL = "smart-turn"
$env:LOCALVOICE_EOT_TIMEOUT_MS = "1000"
$env:LOCALVOICE_EOT_FIRST_THRESHOLD = "0.9"
$env:LOCALVOICE_EOT_THRESHOLD = "0.7"

# D
$env:LOCALVOICE_BACKEND = "lfm3"
Remove-Item Env:LOCALVOICE_AGENT, Env:LOCALVOICE_ASR_SERVER -ErrorAction SilentlyContinue
$env:LOCALVOICE_TRANSPORT = "realtime"
$env:LOCALVOICE_REALTIME = "ws://127.0.0.1:8080/v1/realtime"
```

For the LFM3 arm, install `requirements-realtime.txt`; for EOT arms install
`requirements-eot.txt`.

## 3. Scripted session

Use the same utterances and order in every arm. Run each arm twice: once with
headphones, once with speakers at a normal level. Keep the room and device
fixed within a pair.

| # | Condition | What to do | Expected |
| --- | --- | --- | --- |
| 1 | Normal turn | Ask a short factual question | Reply starts, no interruption |
| 2 | Backchannel | While the reply plays, say “mm-hmm” once, then stop | Playback should not be destroyed |
| 3 | True stop | While the reply plays, say “stop” clearly | Playback stops promptly |
| 4 | Mid-turn pause | Ask a question with a deliberate 0.6–1.0 s pause before the end | Turn should not be cut off |
| 5 | Long turn | Speak for 5–8 s with a short pause | One turn, no premature submit |
| 6 | Memory | State a fact, let the reply finish, then interrupt and ask for it | Fact should survive the interruption |
| 7 | Echo-only | Let the reply play; do not speak | No false interruption |
| 8 | Normal turn | Ask a second short question | Reply starts normally |

Record whether the app’s transcript/status matches what you did. For arm C/D,
note whether the turn released at the first confident score or waited for the
timeout.

## 4. Metrics to record

Use the diagnostics log and, optionally, a session archive:

```powershell
$env:LOCALVOICE_RECORD_SESSION = "outputs/sessions/arm-B-headphones"
```

Per arm, count:

- false cut-off (a mid-turn pause or backchannel ended the turn)
- missed floor take (a clear “stop” did not stop playback)
- stop latency (speech onset to playback stopped)
- resume success (backchannel followed by continued playback)
- endpoint-to-first-audio latency
- memory recall after interruption
- any visible error or archive warning

Do not combine arms into one score. Report each separately.

## 5. Acceptance

A change is a candidate for the default only if:

- **Floor:** reversible reduces backchannel/cough false cut-offs without
  increasing missed floor takes; stop latency p50 is within ~200 ms of hard.
- **EOT:** the 1 s sidecar reduces mid-pause cut-offs without increasing
  endpoint-to-first-audio p90 beyond ~1.2 s.
- **Memory:** interruption recall does not regress.
- **Stability:** no new archive, device or server errors.

If any gate fails, keep the current default and record the failure. The offline
acoustic replay already found a negative backchannel transfer; real-device
results may reject the reversible default even if the authored replay passed.

## 6. Known limits before you start

- The 0.9/0.7 EOT schedule was selected on English data; German and Spanish
  zero-shot transfer was weak. Do not assume it generalizes to another language.
- The acoustic stress replay did not preserve a 220 ms backchannel under
  synthetic echo; headphones are the cleaner first test.
- LFM3 latency and long-session behavior are not yet controlled benchmarks.
- ASR checks are recognition evidence, not human adjudication.

## 7. Stop conditions

Stop and report if:

- the backend cannot be reached or the model files are missing;
- the app cannot open the selected microphone or speaker;
- a test produces an archive error or a corrupted session;
- the same failure repeats twice after a clean restart.

Do not tune thresholds on the test utterances and then report the same set as
held-out evidence. If a threshold changes, freeze it and rerun the full script.
