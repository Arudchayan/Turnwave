# Results and lessons

These are bounded engineering results from different frozen snapshots. They are
not a whole-app acceptance score, an official leaderboard submission or evidence
that conversation is consistently smooth. Numerical observations and provenance
are in [results.json](../research/results.json); chart rows and two genuine
generated reply examples are public. Raw private logs and recordings are not.

| Question | Evidence | Result | Decision |
| --- | --- | --- | --- |
| Can completion race with the final playback block? | Forced queue/producer ordering with fake devices | 0/2,003 → 2,003/2,003 samples played; separate 3,443-sample control | Fix implemented |
| Does the semantic gate release short completed requests promptly? | One 16-case synthetic streaming diagnostic | 5/8 completed intents within 160 ms extra; 8/8 unfinished intents held at least 2 seconds | Timing intent failed |
| Does adding a fixed audio detector solve that tradeoff? | First-endpoint screen on the same 16 development inputs | Completed positives 5→7/8, unfinished held 8→4/8 | Reject the fixed OR rule |
| Does selective extra pre-roll improve fresh capture? | Six frozen paired CPU cases, 12 arms | Events and captured bytes identical in every pair | Do not adopt; benefit unproven |
| Does native generation follow content tasks? | Adapted 24-request VoiceBench slice | 18 normal completions, 6 length-limited | Diagnostic baseline only |
| Can LFM3 time-to-first-audio fit a 4 GiB laptop GPU? | FlashAttention on, Q8 KV cache, single sequence slot (`-np 1`) | Speaker-check TTFT: 0.596 s → 0.102 s; VRAM: 3,451 / 4,096 MiB | Adopted in server script |
| Can the intermediary coordinate tools and deep reasoning without high TTS delay? | Built-in local tools, expert UI cards, sentence-pipelined TTS | Tools resolve within 1 s; deep brief in UI; pipelined TTS streams first sentence | Integrated intermediary |

## Short-turn completion delay

![Additional capture-gate delay for 16 synthetic cases](assets/short-turn-delay.png)

The measurement starts at the **first inner raw VAD endpoint** and stops when the
capture gate releases. It excludes the detector's preceding silence wait, model
generation, output buffering and devices. Each row is one observation; there are
no repeat-run confidence intervals. Labels describe authored intent, not human
conversational ground truth.

“Why?”, “Really?” and “And you?” waited approximately 2.52 seconds for quiet fallback.
All eight unfinished prefixes were held at least two seconds. This was a helper
capture/endpoint diagnostic, with no LFM answer generation. At the first raw
endpoint those three completed requests already had correctly punctuated helper
transcripts; later text could change before fallback. Missing punctuation alone
does not explain the result. A separate helper transcription mapped “Mine” to
“Mug”; that is not evidence of LFM misunderstanding.

## Rejected: a simple audio-detector OR rule

![Completion gains and unfinished-prefix regressions](assets/completion-screen.png)

The screen combined Namo's completed decision with Turn-1-mini's last fully
observed 160 ms step at score >0.5. This is the publisher example threshold,
not a separate benchmark threshold. It rescued “Why?” and “And you?” but still
missed “Really?”. Four unfinished prefixes became positive: “Before you…”,
“Please put the…”, “Which one should…” and “And you should…”.

At these fixed snapshots, holding all prefixes requires a threshold at least
0.908043920994, while rescuing all three misses requires a threshold below
0.283335864544. That rules out a separating scalar threshold **on these points**;
it says nothing universal about temporal or context-aware policies. The 121 CPU
steps and isolated maximum step time are not an end-to-end responsiveness test.
Reserved voices and remaining held-out cases were not consumed to repeat a
known failing rule. See the [16 scored rows](../research/candidate-screen-cases.json).

## Not adopted: selective opening history

The private candidate can extend pre-roll when adjacent energy windows indicate
sustained onset. Earlier known synthetic controls recovered 160 ms and 180 ms of
voice in two cases, but the acoustic suite remained 5/14 eligible passes (18 total,
four invalid-source cases) for both baseline and candidate. Energy cannot identify
whether sustained content is near speech, noise or echo.

The fresh comparison used two public audiobook utterances from one speaker:
clean onset, reduced-gain onset, seeded noise, echo-only, synthetic double-talk,
and reset/mute/fresh-input lifecycle. All six pairs completed once on CPU, with
20 ms callback steps. All 27 source/runtime and five frozen artifact pins matched.
The candidate added and removed **zero capture samples**; events and capture
hashes were identical. Noise-only and echo-only produced no starts/captures;
mute/reset controls matched, with no muted source captured.

This supports bounded non-regression, not efficacy. Some source PCM preceded
capture, but nonzero PCM is not a word boundary; we do not claim lost words.
The run used actual capture/queue/frontend code without a backend, ASR, model
reply, microphone or speaker. Two earlier attempts stopped on dependency-import
guards before neural execution; those are preserved setup failures, not voice
quality failures. The candidate remains private. Full sanitized case intervals,
hashes and source/runtime pins are in `public_capture_comparison` in results.json.

## Native content baseline: 24 actual generated answers

This historical run made 12 IFEval and 12 OpenBookQA requests through the native
GGUF runner, resetting context for each request and limiting output to 512 tokens.
It used a Q8 language model with Q4 components on an RTX 3050 Laptop (4 GiB),
requesting 99 GPU layers. It is not the reference PyTorch configuration.

| Measure | Model-emitted text | Independent generated-audio ASR |
| --- | --- | --- |
| OpenBookQA strict parsed correct | 4/12 | 2/12 |
| IFEval strict prompt pass | 2/12 | 2/12 |
| IFEval strict instruction pass | 3/14 | 3/14 |

All 24 outputs were saved; 18 completed normally and six IFEval outputs hit the
length limit. Failed/length-limited requests remain in denominators and cannot
earn accepted grading credit. Random upstream multiple-choice fallback credit
is excluded. Base.en ASR is unadjudicated recognition evidence: text/ASR
disagreements and extraction failures do not by themselves prove unclear audio.

Median request-to-first-decoded-audio was **344.5 ms**. It excludes capture,
endpointing, playback and room latency. The 361.28 seconds of saved generated
audio were not played in this test. This selected slice is not a full VoiceBench
result and cannot rank this system against other implementations.

## Earlier development controls

An older combined semantic suite passed 7/7 synthetic development cases, including
two stop commands and a joined sentence. It made seven actual submissions, with
no generated-answer evaluation. A separate fresh attempt produced no valid score.

The [two synthetic reply examples](../research/examples/README.md) similarly show
one text-rubric pass and one ownership failure. They are selected illustrations,
not a two-case accuracy estimate or evidence that a particular voice caused failure.

## What remains open

False interruptions, paused-sentence behavior and reply quality remain unresolved.
Controlled real-device acceptance and a clean-machine native installation have
not been established.
See [evaluation boundaries and reproduction](EVALUATION.md).

## LFM3 local voice feasibility check

On 2 November 2026, a single Windows SAPI voice rendered “Hello, can you hear
me clearly?” as 16 kHz mono PCM. Each backend processed that same completed
file three times on an RTX 3050 Laptop GPU (4 GiB), with no microphone,
endpointing, playback or room acoustics in the timing. The input SHA-256 was
`14f325bee8c502810701f87e7683645b74031711268b180461843dc5680e1de3`.

| Backend | First decoded audio, seconds | Total generation, seconds | ASR exact phrase |
| --- | --- | --- | --- |
| LFM2.5 Q4_0, existing native runner | 0.212, 0.139, 0.168 | 3.795, 3.773, 3.879 | 3/3 |
| LFM3 Omni Q4_K_M, llama.cpp b12032 | 0.369, 0.420, 0.416 | 1.761, 1.736, 1.880 | 3/3 |

The first-audio measurement starts before the client call and ends at its first
decoded audio callback. It includes each client's connection setup; LFM3 used a
fresh realtime session each trial. Both prompts requested brief answers, but the
outputs differed: LFM2.5 generated 12.54 seconds of audio each time and LFM3
generated 4.80–5.52 seconds. These observations do not establish a general
speed or quality ranking. The model output text is not evidence of what the
generated speech said; neither reply was independently transcribed for this
check. Exact historical reruns also depend on the local SAPI voice.

The newer runner and weights were separate from the existing model: llama.cpp
b12032 (`dde879bc4`) and LiquidAI LFM3-Omni-2B-GGUF revision `3d29af2ad166f8299c2d820937e47dc0add20ffc`.
The Q4_K_M model and projector were SHA-256 verified against the publisher.
The LFM2.5 files came from revision `7d525f883a077e20afb782f2ff618edcae0e39e4`;
the local baseline used the selected native-abort runner built from
`ec9d1fdd9`, not a fresh clean build. `src/synth/voice_backend_probe.py`
reproduces the measurements given a synthetic 16 kHz mono WAV and each server.

In one separate LFM3 cancellation trial, the client stopped 16 ms after an
abort issued after first audio, then generated a reply on a fresh session.
One muted application speaker check wrote 80,640 source samples to the selected
output device with zero reported underflows and a 12.87 ms maximum recorded
write gap. These do not establish end-to-end interruption latency, audible
quality, or reliability across real rooms and devices.

An additional single app-level check fed the same completed synthetic fixture
to each backend with the microphone muted and wrote the replies to the same
selected output device. Submission to first device write was 0.186 s for LFM2.5
and 0.581 s for LFM3; the player reported zero underflows in both runs and
maximum write gaps of 12.59 ms and 12.34 ms respectively. The two generated
different-length replies, and device writes do not verify acoustic playback.
This is one observation per backend, not a real conversation or a statistical
comparison of perceived smoothness.

In a separate two-turn synthetic check, the input said “My name is Mira. Please
remember it” followed by “What is my name?” in the same LFM3 realtime session.
The model's first emitted text used “Myra” and its second answered “Your name
is Myra.” This demonstrates that the second text reply reflected the prior
response in this one case; it does not verify spelling, spoken audio, or memory
after interruption.

## LFM3 inference optimization on 4 GiB GPU

To operate within a strict 4 GiB VRAM envelope on an NVIDIA GeForce RTX 3050
Laptop GPU, the LFM3 `llama-server` runner script (`src/server_lfm3_wsl.sh`) applies:
`--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 -np 1 --ctx-size 4096 -ngl 99`.

In an application speaker-check test, this configuration was compared against the
unoptimized baseline:

| Configuration | FlashAttention | KV Cache Quantization | Slots (`-np`) | VRAM Peak | Speaker Check TTFT (`last_ttft`) |
| --- | --- | --- | --- | --- | --- |
| Default runner | Off | f16 | Default | Exceeded / tight | 0.596 s |
| Optimized runner | On | q8_0 (K & V) | 1 | 3,451 MiB | 0.102 s |

Peak VRAM remained stable at 3,451 MiB (within the 4,096 MiB physical boundary),
avoiding Windows shared GPU memory paging. Time-to-first-audio (`last_ttft`) on
the initial speaker check dropped from 0.596 s to 0.102 s (~83% latency reduction).
These observations represent single controlled speaker-check runs without
microphone capture, endpointing delays, or room acoustics.

## Adjustable pause between turns

The interface offers 480, 800 and 1,120 ms silence waits. In one CPU replay
through the Silero frontend with echo processing disabled and no holdoff, the
SAPI phrase above was trimmed to 100 ms around samples above 0.008 absolute
amplitude, resulting in a 2.196 s clip. It was repeated with either 650 or
850 ms of inserted silence, followed by 1.5 s of silence, in 20 ms steps.
At both inserted gaps, the 480 and 800 ms settings produced
two detected turns; the 1,120 ms setting produced one. For the 650 ms gap,
the final endpoints were at 5.46, 5.78 and 6.10 s, respectively: joining the
two clips came with 640 ms of additional final wait compared with 480 ms.
The inserted silence is not the entire VAD-measured silence because the clips
also contain quiet audio. This is one synthetic voice and does not establish
word retention, false-interruption rates or a generally safe setting. The
default remains 480 ms and the setting changes only between utterances.

## Synthetic interruption with device playback

A separate LFM3 test played generated speech through the selected Windows
output device while injecting a 16 kHz SAPI-rendered interruption into the
application's capture queue. The microphone itself was simulated; the test
did not measure sound from the room. Two runs used the same interruption,
“Wait, please explain that differently,” but independently generated replies.

| Silence wait | Duck after first nonquiet input | Cancellation after onset | Captured interruption |
| --- | --- | --- | --- |
| 480 ms | 0.162 s | 0.874 s | Split into two submitted turns |
| 1,120 ms | 0.129 s | 3.588 s | One submitted turn |

The last input sample above the test's 0.02 amplitude threshold occurred at
2.307 s. A separate ASR request to the same LFM3 backend transcribed the
480 ms captures as “Wait.” and “Please explain that differently.” For the
1,120 ms run it transcribed the complete phrase in one capture. Those are
machine transcripts, not independent human verification. An earlier 480 ms
trial also split the input; another retained it as one turn. This small,
variable experiment supports offering the longer pause as a choice, not
claiming it prevents fragmentation in real conversations. The smoke tool
checks for extra submitted turns and premature cancellation as well as timing.

An additional app-level agent check used a synthetic “Find an update about the
moon” WAV, LFM3 for local transcription and synthesis, and a real Windows
output device. The model and search HTTP responses were simulated: the agent
issued a mock search request and received a mock source, which appeared in app
state, and the player accepted 46,080 synthesized samples.

A subsequent live spoken-search check ran that same 2.54 s WAV (SHA-256
`212860194694fbb74374ac3872435f9eee7d479500114b0a6a758dc9565eb879`) through
local LFM3 ASR, a live OpenCode Go model (`beam-free`, `reasoning_effort=low`)
with DuckDuckGo HTML search (`LOCALVOICE_SEARCH=ddg`), and local LFM3 speech
synthesis to the active Windows output device. Local ASR transcribed “Find an
update about the moon.”; the agent called `web_search`, returned five HTTPS
sources in application state, and synthesized a 30.64 s spoken answer (735,360
source samples). Time from submission to first device write was 4.096 s (including
ASR, search, and cloud generation; TTS first audio was 0.226 s), with zero
reported playback underflows and a 13.95 ms maximum write gap across 34.81 s
total turn time. Paid OpenCode Go models on the tested account returned HTTP 429
(`GoUsageLimitError`), which the interface surfaces as a rate/usage limit error;
this single live run checks the end-to-end spoken search path, not factual
freshness or acoustic audibility in a room.

## Jarvis intermediary capabilities and pipelined synthesis

When operating with an external reasoning provider (OpenCode Go, Zen, or a custom local OpenAI-compatible endpoint via `LOCALVOICE_CUSTOM_ENDPOINT`), Turnwave
acts as a voice intermediary ("Jarvis" persona) coordinating local tool execution,
deep technical delegation, and speech synthesis:

### Local tool execution and persona management

The intermediary provides direct tool definitions to the chat-completions model:
- `get_time`: Queries current time in remote IANA timezones (local clock, date, and weekday are pre-injected in the system prompt for zero-round-trip answers).
- `get_weather`: Performs geocoding and current/forecast lookups via Open-Meteo.
- `system_status`: Inspects local hardware vitals via `nvidia-smi` (VRAM used/total, GPU utilization, core temperature) and disk capacity, with a 10-second cached snapshot injected into the system prompt so common status questions resolve in a single LLM pass.
- `remember` / `recall`: Retains facts and user preferences across turns (in memory or persistent JSON), searchable by keyword tokens.
- `set_persona`: Adjusts system instructions dynamically between `jarvis` (composed intermediary), `direct` (zero-fluff operator), `natural`, `concise`, and `storyteller`.

### Expert delegation (`consult_expert`)

For complex programming, mathematical, or architectural queries, the intermediary
calls `consult_expert`, delegating the request to a deep reasoning model
(e.g., `kimi-k3`). The resulting technical analysis (up to 4,000
characters) is formatted into structured cards within the web interface's
Expert Notes panel, while the voice client speaks a 2–3 sentence spoken summary.
This decouples visual technical depth from spoken conciseness.

### Turn-speed, HTTP Keep-Alive, and voice-quality optimizations

- **Single-frame WebSocket ASR upload & `TCP_NODELAY`**: `LFM3Client._append` sends up to 128 KB per WebSocket frame instead of 50× 3,200-byte frames and catches Python 3.12 `TimeoutError` during socket reads, reducing 5.0 s local ASR transcription latency from **568.7 ms to 317.0 ms** (44% faster) while eliminating premature 0.6 s socket-timeout aborts.
- **Persistent HTTP/1.1 Keep-Alive pooling**: `AgentClient` reuses a persistent `urllib3.PoolManager` connection pool across turns and tool-call iterations, avoiding repeated ~220 ms TLS handshakes.
- **Natural RVQ acoustic sampling & boundary interpolation**: Replacing greedy argmax (`--temp 0`, which sets `top_k: 1` on LFM3's 8-codebook Mimi RVQ audio tokens) with `--temp 0.7 --top-k 40 --top-p 0.9 --min-p 0.05 --repeat-penalty 1.05` eliminates robotic pitch flattening and limit-cycle buzz, while linear tail extrapolation in `Player.play_chunks` removes the 50 Hz sample-clamping step during 24 kHz → 48 kHz resampling.

### Hybrid text prompt and sentence-pipelined TTS

The interface exposes `/api/prompt` (`POST /api/prompt`), allowing typed prompts
to enter the turn cycle without microphone capture. Replies are processed through
the standard conversation context and played through the audio device.

For answers exceeding 110 characters, `Assistant._speak_answer` segments text
at punctuation boundaries with a 120-character cap on the opening sentence and ~220-character units thereafter. The first sentence is submitted
immediately to the TTS engine, allowing audio playback to begin in 50–90 ms while subsequent
sentences continue to stream. In synthetic unit tests, pipelining reduced initial
playback wait while preserving single-stream audio ordering across sentence boundaries.

### Exploratory LFM3 model modification

An isolated CPU compatibility probe used Heretic 2.0.0 with the LiquidAI
LFM3-Omni-2B checkpoint at revision `5797d07c125503bd4f4c0e7f915ae2d3d703e8de`
(model SHA-256 `76c0ec2c3e27832fb19f886d5eb0afe37f03c5c32c4a796a7ce08cef119d2763`).
The research adapter uses Transformers' text generation for the analysis and
normalizes inactive sampling parameters so the model can be saved. The probe used
three examples in each direction, two examples per score, rank 1 and one trial;
the refusal score was 2/2 before and after, with reported KL divergence 0.0004.
These figures are too small to establish an effect or model quality.

The merged checkpoint had identical tensor names and 24 changed tensors, all in
the language backbone; all other model tensors matched the original exactly.
The result converted with llama.cpp b12032, was quantized to Q4_K_M, and was
served on a separate CPU server using the original projector. Turnwave's client
produced 53,760 speech samples in one TTS call; a separate synthesized sentence
was transcribed correctly. This checks basic functionality only: no human
listening assessment, broad behavior evaluation, or runtime mode switch was tested.
See [Heretic](https://github.com/p-e-w/heretic) and the
[original model](https://huggingface.co/LiquidAI/LFM3-Omni-2B) for their terms.
