# Turnwave

[![Tests](https://github.com/Arudchayan/Turnwave/actions/workflows/tests.yml/badge.svg)](https://github.com/Arudchayan/Turnwave/actions/workflows/tests.yml)

Turnwave is a local speech-to-speech assistant for Windows and WSL2. The
public preview recommends **Standard LFM3-Omni-2B**, the **Balanced** pause,
and the **Interrupt** button.

[Quick start](#quick-start-standard-lfm3) · [Evaluation](docs/EVALUATION.md) ·
[Change log](CHANGELOG.md) · [Known limits](docs/KNOWN_LIMITATIONS.md)

## Quick start: Standard LFM3

1. Install Python 3.12 on Windows and WSL2 Ubuntu. In the repository, create a
   Python environment and install the app requirements.

   ```powershell
   python -m venv .venv
   .venv/Scripts/Activate.ps1
   python -m pip install -r requirements.txt
   ```

2. Install the [Standard LFM3 model and runner](#lfm3-backend-setup).
   Start the backend from the repository root.

   ```powershell
   wsl -d Ubuntu bash src/server_lfm3_wsl.sh
   ```

3. In a second PowerShell terminal with no cloud provider configured, activate
   the environment and start the app muted.

   ```powershell
   .venv/Scripts/Activate.ps1
   $env:LOCALVOICE_AGENT = ''
   $env:LOCALVOICE_SERVER = 'http://127.0.0.1:8080'
   $env:LOCALVOICE_BACKEND = 'lfm3'
   $env:LOCALVOICE_TRANSPORT = 'http'
   $env:LOCALVOICE_FLOOR_MODE = 'manual'
   $env:LOCALVOICE_START_MUTED = '1'
   python src/app.py
   ```

4. Open [Turnwave](http://127.0.0.1:8000/), select your headset, confirm
   **Local Direct** and **Balanced · 0.8 s**, then unmute. If Balanced
   splits your sentences, try **Patient · 1.12 s**. The soundcore R50i
   was used in the initial headset check.

Speak after a reply finishes, or press **Interrupt** and then speak.
Hands-free interruption and the Heretic research model are experimental.
Local Direct sends speech to the configured local backend. Optional cloud
conversation sends recognized text and recent context to the configured
provider. Search sends queries to the selected search provider. See
[data handling](docs/PRIVACY.md) for what is saved and sent.

## What we measured

| Check | Result | Scope |
| --- | --- | --- |
| Pause preset replay | A 650 ms inserted pause ended the turn early in 20/20 clips at 480 ms and 4/20 at 800 ms. The median final endpoint was 307.5 ms later. | Public read-speech fixture with inserted pauses. |
| Initial headset check | A paused utterance failed with Balanced and passed with Patient. Hands-free mode stopped for an acknowledgment and misunderstood a spoken correction. | One headset session. |
| Paired XSTest screen | Standard refused 24/32 unsafe prompts and Research refused 22/32. One blinded coding agent graded all 144 completions across 72 paired prompts. | The result did not establish reliable refusal reduction. |

[Methods and reproduction](docs/EVALUATION.md) · [Detailed results](docs/RESULTS.md) ·
[Real-device testing](docs/TESTING.md) · [Experiment ledger](EXPERIMENTS.md) ·
[Architecture](docs/ARCHITECTURE.md) · [Example audio](research/examples/README.md)

## LFM3 backend setup

Standard LFM3-Omni-2B uses a recent llama.cpp server. It uses a different
protocol, so it does not use the pinned LFM2.5 runner or its patches. Obtain the
Q4_K_M model and projector from [LiquidAI's GGUF release](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF)
and place `LFM3-Omni-2B-Q4_K_M.gguf` and
`mmproj-LFM3-Omni-2B-Q4_K_M.gguf` in `models/LFM3-Omni-2B-GGUF/`.
Install a compatible llama.cpp build (b11960 or newer) separately. The default launcher looks for the b12032 CUDA build in
`models/runner/llama-b12032-cuda/files/llama-b12032/`; set
`LOCALVOICE_LFM3_RUNNER` inside WSL to choose another build. CUDA builds also
need compatible runtime libraries. The launcher checks the matching b12032
CUDA 13.4 runtime under `models/runner/llama-b12032-cuda/runtime/`; alternatively
set `LOCALVOICE_CUDA_LIB` inside WSL to their directory. With those assets installed:

```powershell
wsl -d Ubuntu bash src/server_lfm3_wsl.sh
```

Use the [quick start](#quick-start-standard-lfm3) to launch the app against
this server. The LFM3 client uses a realtime WebSocket for speech conversation
and transcription, and streaming PCM for synthesis.
The tested configuration and its limits are recorded in [results](docs/RESULTS.md).

See [model-modification research](#optional-model-modification-research) for
separately prepared variants and in-app switching.

The launch script `src/server_lfm3_wsl.sh` configures llama.cpp inference and acoustic sampling optimizations
tuned for 4 GiB GPU memory limits:
- FlashAttention (`--flash-attn on`)
- Symmetric 8-bit quantized KV caching (`--cache-type-k q8_0 --cache-type-v q8_0`)
- Single parallel sequence slot (`-np 1`)
- Natural RVQ acoustic sampling (`--temp 0.7 --top-k 40 --top-p 0.9 --min-p 0.05 --repeat-penalty 1.05`)

A historical speaker check on an NVIDIA GeForce RTX 3050 Laptop GPU (4 GiB)
measured 3,451 MiB peak VRAM and reduced first audio from 0.596 s to 0.102 s.
That timing excludes microphone capture and turn detection. Launcher overrides
include `LOCALVOICE_FLASH_ATTN`, `LOCALVOICE_KV_TYPE`,
`LOCALVOICE_PARALLEL`, and `LOCALVOICE_TEMP`. Optional voice conditioning
uses `LOCALVOICE_VOICE` and `LOCALVOICE_SPEAKER_REF`.

#### Optional model-modification research

The original and a separately prepared research variant can be selected when
starting an LFM3 backend. Model weights and native runners are not included in
this repository.

For experiments, obtain the full-precision LFM3 Omni checkpoint separately,
inspect its custom Python code, and install `heretic-llm==2.0.0` in a separate
environment. The experimental adapter gives Heretic access to text logits and
hidden states while preserving the saved speech-model architecture:

```powershell
python -m venv .venv-heretic
.venv-heretic/Scripts/python -m pip install heretic-llm==2.0.0
.venv-heretic/Scripts/python scripts/heretic_lfm3.py --model-dir models/LFM3-Omni-2B-HF --trust-model-code -- --device-map cpu --batch-size 1 --study-checkpoint-dir outputs/maintenance/heretic-checkpoints
```

Options after `--` are passed to Heretic. Save any modified checkpoint in a
separate directory, then use a compatible llama.cpp converter and quantizer to
prepare model and projector GGUF files. Verify their compatibility and behavior
before use. The helper executes checkpoint code only with the explicit trust
flag; inspect that code first. See [Heretic](https://github.com/p-e-w/heretic),
the [LFM3 model card](https://huggingface.co/LiquidAI/LFM3-Omni-2B), and
[results](docs/RESULTS.md) for terms, the small experiments, and their limits.
The existing research checkpoints are not published or automatically selected.

For the WSL server, set `LOCALVOICE_LFM3_VARIANT=research`,
`LOCALVOICE_LFM3_RESEARCH_MODEL`, and `LOCALVOICE_LFM3_RESEARCH_PROJECTOR`
inside WSL before running `src/server_lfm3_wsl.sh`. Omit those variables when
starting the standard variant. For an independently installed Windows
llama-server and matching artifacts, the optional Windows launcher can select
the variant explicitly:

```powershell
./src/server_lfm3_windows.ps1 -Variant standard
# Stop the standard server first; supply your own research artifacts:
./src/server_lfm3_windows.ps1 -Variant research -Model path/to/research.gguf -Projector path/to/projector.gguf
```

Use `-Runner` to select an installed Windows server. Research mode requires
explicit model and projector paths. The launcher does not check model-projector
compatibility or validate the quality or behavior of the modified model.

To switch from the app, start the standard and research servers independently
on distinct loopback ports, with enough memory for both. For example, configure
the research WSL server as described above and set `LOCALVOICE_LFM3_PORT=18087`
before starting it. Then start the app with:

```powershell
$env:LOCALVOICE_BACKEND = 'lfm3'
$env:LOCALVOICE_LFM3_RESEARCH_SERVER = 'http://127.0.0.1:18087'
python src/app.py
```

The standard server defaults to port 8080 and may be changed with
`LOCALVOICE_SERVER`. Both URLs must be loopback HTTP addresses with explicit
ports. The app does not start or stop either server. It checks the identifier
reported by each server's `/v1/models` endpoint before use. Expected names
default to `LFM3-Omni-2B-Q4_K_M.gguf` and `research-q4.gguf`; override these
with `LOCALVOICE_LFM3_STANDARD_MODEL_ID` and `LOCALVOICE_LFM3_RESEARCH_MODEL_ID`
if the servers report different names. These are server-reported identifiers,
not independently verified file hashes.

The selector starts a new conversation when changed and cannot be used during
an active turn or recorded session. In local mode it changes the model for the
conversation; in cloud mode it changes local transcription and speech synthesis,
while the selected cloud reasoning model stays the same. This option requires
the LFM3 HTTP transport and does not support a separate ASR server. Both server
contexts are reserved by the app while it runs.

## Legacy LFM2.5 backend

- Windows with working input/output devices and WSL2 Ubuntu.
- Python 3.12 for the application. Node.js for browser-script tests.
- A compatible Liquid audio native runner and separately obtained model files.
- Suitable hardware for the selected runner; performance depends on hardware.

The native integration pins llama.cpp PR #18641 commit
`ec9d1fdd9cc18643c5e161b65b8053f6f6f34a4b`. Obtain the source checkout,
model assets and toolchains separately; review [their terms](docs/THIRD_PARTY.md).
For an optional CUDA build, activate CMake, compatible CUDA 13 `nvcc` and a host
C++ compiler in WSL, then run from the repository root:

```bash
bash src/build_cuda_server.sh
```

The helper expects an existing checkout at `~/liquid-build`, verifies the exact
revision and accepts clean source or the complete reviewed patch set. It does
not clone, fetch, reset or install prerequisites. WSL environment overrides are
`LOCALVOICE_RUNNER_SOURCE`, `LOCALVOICE_RUNNER_BUILD_DIR`,
`LOCALVOICE_BUILD_JOBS` and `CUDACXX`. The build contract has fake-command tests;
a clean-machine native build has not been validated for this release.

### Model files and runner

Place model files in `models/LFM2.5-Audio-1.5B-GGUF/`:

- `LFM2.5-Audio-1.5B-Q8_0.gguf` or `LFM2.5-Audio-1.5B-Q4_0.gguf`
- `mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf`
- `vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf`
- `tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf`

The launcher prefers a local runner under `~/liquid-build/build/bin` inside
WSL. `LOCALVOICE_GPU_RUNNER` can select another runner directory; a prebuilt
runner may instead be placed under `models/runner/llama-liquid-audio-ubuntu-x64`.
Keep the required shared libraries with the runner. Stop the server before
applying the complete five-patch set using `src/apply_runner_fixes.sh`; that
helper patches the same existing checkout and rebuilds the server.

From the repository root, start the LFM2.5 backend:

```powershell
wsl -d Ubuntu bash src/server_wsl.sh
```

After it prints `Server ready`, use a second terminal:

```powershell
.venv/Scripts/Activate.ps1
$env:LOCALVOICE_BACKEND = 'lfm2'
$env:LOCALVOICE_START_MUTED = "1"
python src/app.py
```

Open **http://127.0.0.1:8000**, select the input device and unmute when ready.
Headphones are the more predictable starting setup. Muted startup still opens
an input stream but does not submit microphone turns.
The “Pause before reply” control trades response speed for more time to pause
while speaking. Changes apply between utterances; longer waits cannot guarantee
that an unfinished sentence will remain together. The setting returns to the
default when the app restarts.

## Voice settings

`LOCALVOICE_SERVER` changes the backend URL (default local port 8080).
`LOCALVOICE_CHAT_SYSTEM` changes the chat instruction. Optional transcription
uses `LOCALVOICE_ASR_SERVER` and requires a genuinely separate model context.
For devices with long playback delays, the experimental
`LOCALVOICE_ECHO_PROFILE=long` keeps echo history longer after playback. It may
reduce self-interruptions but can delay quiet speech while echo is present; see
[the measured tradeoff](docs/RESULTS.md). Set `$env:LOCALVOICE_ECHO_PROFILE = 'long'`
in PowerShell and restart the app to change profiles.
`LOCALVOICE_FLOOR_MODE=manual` is the default. Use the **Interrupt** button
to stop a reply before speaking. Set `LOCALVOICE_FLOOR_MODE=reversible` to
test the experimental duck-and-resume policy or `LOCALVOICE_FLOOR_MODE=hard`
to compare immediate spoken cancellation. Restart the app after changing modes.
For LFM3 Omni, `LOCALVOICE_CONTEXT_RECOVERY=1` enables an experimental recovery
after an interrupted local reply. It keeps the preceding user audio in memory,
transcribes it locally on the next turn, and supplies it with an explicitly
possibly unheard assistant draft as context. This adds latency, may misrecognize
speech, and does not reconstruct what was actually heard. Resetting the
conversation or changing mode clears the saved context. It is off by default.
`LOCALVOICE_EOT_MODEL=smart-turn` enables the optional Smart Turn completion
sidecar and extends the silence wait to `LOCALVOICE_EOT_TIMEOUT_MS` (default
1000; 2000 lowers false cut-offs further at a longer worst-case wait).
`LOCALVOICE_EOT_FIRST_THRESHOLD` defaults to 0.9 and
`LOCALVOICE_EOT_THRESHOLD` (later queries) to 0.7. Install
`requirements-eot.txt` first. See [architecture](docs/ARCHITECTURE.md)
and the offline replay in [evaluation](docs/EVALUATION.md).
The app and backend test entry points reserve exclusive context ownership;
stop the app before running tests that use its backend, even while muted.

### Development agent (experimental)

Open **development agent** in the app to select an installed OpenCode model,
including DeepSeek V4.1 Flash Free, describe an improvement and start a run.
Install and authenticate the OpenCode CLI separately. Free-tier availability
and limits can change; OpenCode serves free models through its own client.
The selected model receives your request, repository files it reads, and a
small summary of recent diagnostic status and tool/check names; it does not
receive the transcript automatically. OpenCode and tests may create their own
local logs. Keep credentials out of the repository.

Each run starts from the currently committed revision, checks baseline tests,
and works in a detached worktree
under `outputs/self-improvement/`. It retains `request.json`, `opencode-*.jsonl`,
`baseline-pytest.log`, `pytest-*.log`, `review.diff`, `result.json` and the
worktree for review.
If tests fail, one repair attempt sends the last part of the test output to the
model and reruns the suite; each attempt retains its own log. Test runs omit `LOCALVOICE_*`, `OPENCODE_*`, Brave credentials and pytest
plugin overrides so the app’s active profile does not silently change the test
baseline. Other user environment and filesystem access still apply. The assistant can edit
source, tests, docs and README; it does not merge its changes. A policy blocks
shell commands and external directory access in OpenCode, but this is not an
OS sandbox: generated tests run with your user permissions and may access files
or the network. Only start runs you are comfortable testing on this computer,
and review both code and test changes before applying them.

### Optional cloud conversation and web search

Changing the conversation mode or model starts a new conversation.

By default Turnwave uses the local speech-to-speech model. To use an OpenCode Zen
chat-completions model instead, configure these environment variables before
starting `src/app.py`:

```powershell
$env:LOCALVOICE_AGENT = 'opencode-zen'
$env:LOCALVOICE_AGENT_MODEL = 'kimi-k3' # optional; choose a compatible chat-completions model
```

Go and Zen use `kimi-k3` when no model is specified. `LOCALVOICE_AGENT_MODEL`
sets the startup default and the default for a newly selected provider; the
interface can also select a model explicitly. Returning from local mode to the
same provider retains its last selected model for the current app session.

Set `OPENCODE_ZEN_API_KEY` in the application process environment using your
preferred secret manager. A missing Zen key or provider limit is
reported in the interface. Selecting a model starts a new conversation. When the cloud agent is enabled, web search defaults
to Brave if `BRAVE_SEARCH_API_KEY` is set, and otherwise to DuckDuckGo HTML
search without a search API key. The interface can switch between DuckDuckGo,
Bing RSS, Brave and off while an agent is selected; Brave requires a key in the
app environment. Set `LOCALVOICE_SEARCH=bing` or `LOCALVOICE_SEARCH=none`
before launch to choose the initial setting. Avoid saving any key in a script
or committing it to the repository.

```powershell
$env:LOCALVOICE_SEARCH = 'bing' # set before launching the app
```

DuckDuckGo may block automated requests. If it returns no usable results,
Turnwave marks its Wikipedia fallback as limited and warns that current web
information could not be verified. Site-restricted queries do not use that
fallback, because Wikipedia cannot honor the requested site restriction.
When a requested web search returns no usable evidence, the spoken reply says
that verification failed instead of repeating an unsupported model answer.
When only the Wikipedia fallback is available, the spoken reply identifies the
limitation, and the interface shows the source links.
Links stay with the answer they supported during the current conversation,
including after follow-up questions. Starting a new conversation clears them.
If a model emits one complete web-search request as text instead of a structured
tool call, Turnwave can execute that search and pass the results back to the
model. Other textual tool requests and ambiguous markup are reported as errors.
For answers after web search, a dotted version number absent from all returned
titles and snippets is replaced by an uncertainty message. This narrow check
does not verify other claims, and search snippets can be incomplete or stale.

The local voice server still performs transcription and speech synthesis. The
recognized text and recent conversation are sent to the selected provider; the model is
told whether the current input was transcribed speech or typed text, but does
not receive the raw audio. If configured,
search queries go to the configured search provider (Brave, DuckDuckGo or Bing,
and potentially Wikipedia as a DuckDuckGo fallback) and search results go to the model. The
interface shows the active mode, model, search availability, and links returned
by search. The model may choose whether to search. `LOCALVOICE_AGENT=opencode-go` selects Go
and uses `OPENCODE_GO_API_KEY`. OpenCode describes Go as intended for
coding-agent traffic, so general conversation may not be supported; choose Zen
for general conversation. See [OpenCode Go model endpoints](https://opencode.ai/v2/docs/console/go/)
and [Brave Search](https://api-dashboard.search.brave.com/app/documentation/web-search/get-started).
Keep keys in your process environment, never in tracked files. The cloud path
requires network access and provider accounts; it is not covered by the local
model's privacy guarantees. The interface reports missing configuration and
request failures. Agent context is kept in memory and reset with the conversation.
Provider terms vary by model and can change. As documented by OpenCode on
November 4, 2026, Go's `iceman-contributor-free` uses prompts and completions to
improve the model, and `beam-free` has 30-day data retention. Review the current
[OpenCode Go privacy terms](https://opencode.ai/v2/docs/console/go/#privacy)
before choosing a model. Availability of these free models can be limited.

### Jarvis voice intermediary and local capabilities

When an external agent backend is active (`LOCALVOICE_AGENT=opencode-go`, `opencode-zen`, or `custom` via `LOCALVOICE_CUSTOM_ENDPOINT`, defaulting to `http://127.0.0.1:8081/v1/chat/completions`),
Turnwave operates as a voice intermediary ("Jarvis" persona) bridging the user with
local tools, expert reasoning delegation, and search:

- **Built-in local tools**:
  - `get_time`: Queries remote IANA timezones (local clock, date, and weekday are pre-injected in the system prompt for zero-round-trip answers).
  - `get_weather`: Live conditions and forecasts via Open-Meteo.
  - `system_status`: Queries GPU model, VRAM utilization, temperature, and disk space via `nvidia-smi` and system utilities (with a 10-second cached snapshot injected into the system prompt).
  - `remember` / `recall`: Retains session and persistent user preferences or facts across turns.
- **Expert delegation (`consult_expert`)**: For complex programming, mathematical, or architectural questions, the intermediary delegates to a deep reasoning model (e.g. `kimi-k3`). The full structured technical brief is rendered in the web studio's Expert Notes cards, while the assistant speaks a concise 2–3 sentence takeaway.
- **Persona switching (`set_persona`)**: Dynamically switches voice style between `jarvis` (composed intermediary), `direct` (zero-fluff operator), `natural`, `concise`, and `storyteller`.
- **Typed conversation (`/api/prompt`)**: Type a question in the interface and receive a spoken answer. The selected local or cloud conversation backend handles the prompt and its context.
- **Sentence-pipelined TTS**: Multi-sentence replies (>110 characters) are chunked into sentence segments (first sentence capped at 120 characters, subsequent segments ~220 characters) and pipelined sequentially to the TTS runner, delivering initial spoken audio in 50–90 ms without waiting for the complete multi-sentence paragraph to synthesize.

## Optional LFM3-Audio-2B realtime path

LFM3-Audio-2B is a newer 2B local audio model with 32K context, 18 speech
languages, tool calling, and native streaming through llama.cpp's
OpenAI-compatible `/v1/realtime` WebSocket. It needs llama.cpp **b11960 or
newer**; the pinned LFM2.5 runner is not used for this path.

Place these files in `models/LFM3-Audio-2B-GGUF/`:

- `LFM3-Audio-2B-Q4_K_M.gguf`
- `mmproj-LFM3-Audio-2B-Q8_0.gguf`

Obtain a b11960+ `llama-server` and start it from WSL:

```powershell
wsl -d Ubuntu bash -c 'LOCALVOICE_MODEL_DIR="$PWD/models/LFM3-Audio-2B-GGUF" LM=LFM3-Audio-2B-Q4_K_M.gguf MM=mmproj-LFM3-Audio-2B-Q8_0.gguf bash src/server_lfm3_wsl.sh'
```

Then run the app with the realtime transport:

```powershell
python -m pip install -r requirements-realtime.txt
$env:LOCALVOICE_TRANSPORT = "realtime"
$env:LOCALVOICE_REALTIME = "ws://127.0.0.1:8080/v1/realtime"
python src/app.py
```

Set `LOCALVOICE_LFM3_RUNNER` in WSL if the default runner is unavailable. This
experimental transport supports local conversation only; cloud agents and a
separate ASR server require the HTTP transport. Clear `LOCALVOICE_AGENT` and
`LOCALVOICE_ASR_SERVER` before using the realtime path.

The realtime client keeps one conversation socket between turns and opens a
fresh one when the app resets context after an interruption. A local smoke test
validated audio-in, interleaved text/audio-out, and `response.cancel`; controlled
latency, long-session and real-device acceptance remain to be measured. LFM3
model assets are not distributed here; review
[their terms](docs/THIRD_PARTY.md).


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


## Research and reproducibility

[The research index](research/README.md) connects measured observations, charts,
decisions and examples. [Evaluation](docs/EVALUATION.md) explains how to run the
offline tests, regenerate charts and prepare content-benchmark inputs. Published
data reproduce reported counts and figures; exact historical neural reruns also
require assets and source inputs that are not all distributed here. The 24-case
content slice is not a full VoiceBench score or a leaderboard claim.

## Recording and privacy

Normal launches write private diagnostics under `outputs/logs/`. Audio archiving
is opt-in; see [data handling](docs/PRIVACY.md) and [recorded sessions](docs/RECORDED_SESSIONS.md).
Muted startup opens the input device but does not submit microphone turns.
Keep these loopback services local; they have no multi-user authentication.

The repository includes two synthetic reply examples. Session recordings and
raw experiment data are kept locally; see [data handling](docs/PRIVACY.md) for
storage details and [sharing results](docs/PUBLICATION.md) before contributing media.

## Repository map

- `src/`: application, audio pipeline and native runner helpers.
- `src/synth/`: optional evaluation and research tools with additional prerequisites.
- `tests/`: generated-fixture and simulated-device regressions.
- `patches/`: reviewed changes against the pinned external runner.
- `docs/`, `research/`: architecture, evidence, limitations and curated examples.

## License

Turnwave's application code and documentation are [MIT licensed](LICENSE).
Models and dependencies have their own [licenses](docs/THIRD_PARTY.md).

See [contributing](CONTRIBUTING.md), [security](SECURITY.md) and
[publication guidance](docs/PUBLICATION.md).
