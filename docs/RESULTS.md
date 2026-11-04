# Results and lessons

## Capture-loop interruption replay

On November 4, 2026, the app capture loop was replayed with one generated
far-end clip, one public LibriSpeech dummy near-end clip, simulated direct and
80 ms delayed echo, and instantaneous playback gain changes. Each policy ran
eight cases: echo-only, a 220 ms speech fragment, a 900 ms sustained fragment,
and two 400 ms fragments separated by 500 ms, each at full and half near-end
gain. This was a clipped speech fragment, not an annotated acknowledgment.

The current policy at main `326e6c0` canceled the full-gain 220 ms fragment
at 3,180 ms on the audio clock, versus 1,900 ms for the draft reversible policy
at `a0d75d3`. Neither detected the half-gain fragment. Both canceled the
900 ms case at both gains: current at 3,080 and 3,140 ms; draft at 1,900 and
2,400 ms. Near speech began at 1,500 ms. Neither canceled either echo-only
case. The draft also split the full-gain paused case into two captures. These
are individual diagnostics, not rates or evidence of headset performance.

The simulation uses virtual devices, one speaker pair, a simplified acoustic
path, and instant gain changes; real playback uses gain ramps. The two policies
were tested in different source revisions; the experiment does not isolate the
policy from all other branch differences. It does not assess recognition,
semantic intent, human listening, or a real room. It cannot establish that a
threshold change would improve real backchannels or quiet interruptions.

With the optional public dataset dependencies installed, reproduce the current
policy with `python -m src.synth.duplex_barge_probe --public-human --out outputs/barge.json`.
The experimental branch also supports `--mode reversible`. For a supplied mono
PCM16 WAV use `--near` instead of `--public-human`. Output records normalized
fixture hashes and per-case capture and gain events. No microphone or backend is
opened. The public near clip is fetched from an unpinned dataset revision, so
compare its recorded hash when reproducing.

### Synthetic acknowledgments and short commands

A follow-up used Windows SAPI David Desktop and Zira Desktop to synthesize four
fixed utterances each: “mm hmm” and “yes” as intended acknowledgments, and
“stop” and “wait a moment” as intended commands. After trimming low-energy
frames and normalizing each utterance, the current main capture loop replayed
each at full and half gain: 16 trials total. Full gain canceled 3/4 intended
acknowledgments and 3/4 intended commands; half gain canceled 1/4 intended
acknowledgments and 0/4 intended commands. The actual trimmed utterances were
420–780 ms long. “Yes” can signal different intent in conversation; its label
here reflects this synthetic protocol, not acoustic proof of a backchannel.

These data use generated speech and one simulated far-end/echo path, without a
microphone or real speaker. The gains are relative to fixture normalization, not
calibrated speech sound levels. They are not a natural conversation accuracy
estimate. A single rerun of David's “mm hmm” reproduced both outcomes.

The draft reversible policy, tested at `a0d75d3` on the same 16 fixtures,
made the same cancel/no-cancel decisions in every case. For the three full-gain
commands it did cancel, cancellation occurred at 1,900–2,200 ms on the audio
clock instead of 3,020–3,080 ms for the current policy; near speech started at
1,500 ms. It also canceled the same three full-gain acknowledgments sooner.
The revisions differ beyond the floor policy, so this is not an isolated
causal comparison. No candidate is promoted on these results.

On Windows with the named voices installed, generate fixtures into a new folder
and replay each file, for example:

```powershell
./src/synth/generate_interrupt_fixtures.ps1 -OutDir outputs/interrupt-fixtures
python -m src.synth.duplex_barge_probe --near outputs/interrupt-fixtures/david-ack.wav --trim-silence --utterance --out outputs/david-ack.json
```

Windows voice versions can differ; retain generated fixture hashes and the
report's normalized audio hashes when comparing runs. Do not publish recordings
without reviewing their origin and privacy.

### VAD sensitivity screen

On the same 16 synthetic utterances, lowering the VAD probability threshold
from 0.5 to 0.3 and onset confirmation from three frames to two recovered the
previously missed full-gain Zira “stop.” The candidate still missed all four
half-gain commands and canceled the same acknowledgments. It reported no
cancel in two repetitions of the same echo-only signal. For detected full-gain
cases, cancellation happened 60–100 ms later than the baseline because the
capture endpoint was reached later. One rerun of the recovered command matched
the original screen at both gains. This is a small in-sample diagnostic, not
an estimate of recall, false-positive rate, or real-device latency; the two
settings changed together and their effects were not isolated. The candidate
was not adopted.

The same replay can be run with `--vad-threshold 0.3 --start-chunks 2` added to
the command above. These switches affect only the benchmark process.

### Echo-correlation gate screen

A frame trace of the half-gain synthesized Zira “stop” showed high speech
probabilities during the utterance while the onset echo-correlation score stayed
above the existing 0.6 gate. In a separate screen on the same 16 synthetic
utterances, changing that gate alone to 0.9 increased half-gain command
cancellations from 0/4 to 3/4, but increased half-gain acknowledgment
cancellations from 1/4 to 4/4. At full gain, commands rose from 3/4 to 4/4 and
acknowledgments from 3/4 to 4/4. One rerun of the half-gain David “stop”
matched the candidate result. The gate cannot distinguish intent, so this
candidate was not adopted.

A separate 50-case frontend screen compared both gates on the earlier synthetic
echo-tail fixture: both had 3/5 echo-only starts, detected 10/10 near-only
cases, and had 4/10 overlap starts before near speech. Those are not independent
natural-device samples: the screen uses one far/near pair and five artificial
delays. It does not establish that the relaxed gate is safe against other echo.
For the capture-loop probe, add `--echo-threshold 0.9` to the utterance command;
the benchmark scales the correlation score to emulate that gate, leaving the
production frontend unchanged. These are in-sample diagnostics, not a model of
all voices, devices, rooms, or acknowledgment intents.

### Local ASR screen for interruption intent

One diagnostic transcribed the eight synthesized short utterances above, first
clean and then after capture through the virtual echo-mixing loop with the
experimental 0.9 echo gate at full gain. LFM3 Omni Q4 on CPU transcribed all
eight clean inputs into recognizable intended words or vocalization. For the
echo-mixed captures, it transcribed Zira's “stop” as “Yesterday” and Zira's
“mm-hmm” as “M.M. Mr.”; the other six outputs resembled their intended
utterances. The capture loop recorded all eight inputs. For these short samples,
capture transcription took 0.78–1.78 seconds per request on that machine,
after capture endpointing. Timings do not include a second server's setup and
do not establish production latency.

A second run of the checked-in probe again transcribed the echo-mixed “stop”
as “Yesterday” and rendered the echo-mixed “mm-hmm” as “MMM, Mr.” The other
six outputs again resembled their intended utterances. These are repetitions of
the same fixtures and synthetic path, not independent speakers or rooms.

This screen does not assess human speech, other devices, low-gain missed
onsets, a turn-intent model, or whether “yes” is an acknowledgment in context.
The errors can produce both missed and false decisions, so no semantic gate was
added. To rerun, stop the app, start an isolated LFM3 backend and generate the
fixtures as above, then run:

```powershell
python -m src.synth.semantic_interrupt_probe --fixtures outputs/interrupt-fixtures --url http://127.0.0.1:18085 --out outputs/interrupt-asr.json
```

The probe requests the backend ownership lock and records transcripts and
fixture hashes. It cannot protect against unrelated clients that ignore that
lock. Use a dedicated backend and generated or otherwise non-private material.

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

## Current policy diagnostics

These are newer offline diagnostics, separate from the frozen historical
snapshots above. Methods and limits are in [evaluation](EVALUATION.md).

| Question | Evidence | Result |
| --- | --- | --- |
| Does reversible floor control preserve short backchannels? | Seven authored barge-in shapes; plus a real-frontend acoustic stress replay | Authored: hard 3 non-limitation false cancels, 0/3 backchannels preserved; reversible 0 false cancels, 3/3 preserved, 0 missed, median commit 140 ms. Acoustic stress: a 220 ms backchannel produced ~316 ms of VAD evidence and was committed by both modes. |
| Does Smart Turn improve the public-pause trade-off? | 400 English turns from `livekit/eot-bench-data`; conversation-grouped split, 466 held-out pauses | Fixed 480 ms: 63.92% false cut-offs, 100% recall. Two-threshold 0.9/0.7 re-query + 1 s default: 31.37% false cut-offs, 100% recall, p90 1 s; 2 s: 19.22%, p90 2 s. |
| Does the English EOT schedule transfer cross-lingually? | 400 German and 400 Spanish turns, English thresholds fixed | German: fixed 62.95% false cut-offs vs 45.40% (1 s) / 39.55% (2 s), 100% recall. Spanish: 67.25% vs 59.22% / 52.93%. Transfer is weak; per-language calibration needed. |
| Is Smart Turn fast enough on this CPU? | 19 warm 8 s queries | 55.47 ms mean, 47.76 ms median, 81.93 ms p90. |
| Does the LFM3 realtime transport work? | One public example WAV through `/v1/realtime` | Interleaved text and 24 kHz speech; 0.68 s first audio and 5.04 s reply on one warm local run. |
| Is LFM3 ASR usable? | 20 LibriSpeech dummy clips, Q4_K_M CPU | 5.74% corpus WER, 65% sentence error; clean read speech, small subset. |

The floor figures are policy/timing diagnostics. The EOT figures are
per-pause observations on public English conversational audio, not per-turn or
real-device acceptance. The LFM3 figure is a single smoke test, not a latency
benchmark. The mid-turn-pause false cancel is reduced but not eliminated; the
optional sidecar and floor mode remain opt-in.

## Local continuity after a context reset

On November 4, a paired LFM3 Omni Q4 diagnostic used synthesized speech for
“Give me three ways to relax after work” followed by “What about something
else?” The same model's ASR recovered both inputs exactly. With the conversation
socket retained, the follow-up gave additional suggestions; after resetting it,
the reply was “What else?” Two runs of this single prompt pair showed that same
pattern. This is a small diagnostic, not a general accuracy score or a live
interruption test. The app currently resets local context after cancellation
because the backend cannot truncate its context to what was actually heard.
Reproduce with the app stopped and an LFM3 server running:

```powershell
python -m src.synth.continuity_reset_probe --out outputs/continuity-reset.json
```

An experimental recovery transcribes the interrupted user's prior audio and
provides it, plus a generated draft labeled as potentially unheard, to a new
LFM3 session. A three-round follow-up comparison on this same prompt pair gave
three topical follow-ups with new suggestions under recovery; three resets
without recovery instead asked what else the user wanted. Median time from the
follow-up request to first audio was 2.641 s with recovery versus 1.303 s with
reset alone (and 0.289 s with the original context retained). The automated test
suite ran concurrently for part of this local GPU diagnostic, so timing is only
indicative. It used synthesized inputs and no actual interruption; the draft may
include text the user never heard. Recovery is opt-in pending broader checks.
In a subsequent three-round simulation that cancelled generation after one
second of generated audio, all three recovered replies stayed on the relaxation
topic, though each repeated a music suggestion. First-audio times were 3.995,
4.676 and 7.134 seconds after the follow-up request. This exposes a substantial
latency cost after a real cancellation; the experiment does not establish that
the recovery is ready for everyday use.

An alternative replayed the previous audio directly into a fresh session,
along with the interrupted draft marked as potentially unheard. In a three-pair,
fixed-order diagnostic after simulated cancellation, reset-only first-audio
times were 2.559, 6.844 and 2.812 s; direct-audio replay times were 3.131,
2.885 and 2.731 s. All three replay replies remained on topic, but two repeated
some suggestions; the three reset-only replies all asked “What else?” An earlier
unpaired replay run took 4.152, 10.694 and 9.812 s, showing substantial timing
variation. Both use one synthesized prompt pair and the same model for fixture
ASR. This does not establish a reliable speedup, so direct replay is a research
probe and is not used by the app. With the app stopped and the LFM3 server
running, reproduce the paired diagnostic with:

```powershell
python -m src.synth.audio_context_replay_probe --rounds 3 --out outputs/audio-context-replay.json
```

In a separate synthetic echo sweep using one far-end recording, 12 delay/gain
combinations produced no starts in either hard or reversible mode. The room and
headset path were not measured. The acoustic double-talk replay still cancelled
its short backchannel in both modes, so neither result establishes reliable echo
or backchannel handling on a real device.

## Delayed playback echo tail

A synthetic frontend comparison used one generated far-end clip, one public
LibriSpeech near-end clip, five direct echo delays (80–1,100 ms), and full- and
quarter-level near speech. Unlike the earlier acoustic replay, it stopped
feeding the render reference when playback ended. The standard 300 ms tail and
750 ms history detected echo-only speech in 3/5 cases. An experimental 1.5 s
tail with 1.6 s history detected none. Both detected all 10 near-only cases. In
the 10 mixed echo/speech cases, the standard profile started before near speech
in four; the long profile had no early starts but delayed one quiet overlap
onset by 850 ms. Each result uses one signal pair and a simple delayed gain,
not a measured headset, room impulse response or varied speakers. Timing of
speech onset includes the detector's confirmation time. Median wall time to
process each 5.5-second synthetic case was 0.263 s for standard and 0.462 s for
long on this machine; this is not a real-time CPU-utilization measurement.

The app keeps the standard profile by default. Set
`LOCALVOICE_ECHO_PROFILE=long` before launch to try the longer profile; it can
improve long-tail suppression while delaying quiet interruptions. Reproduce the
50-case probe with (the output records whether the public clip or the generated
fallback was used; the fallback is not a reproduction of these exact results):

```powershell
python -m src.synth.echo_tail_probe --out outputs/echo-tail-probe.json
```

## Typed local conversation smoke check

On November 4, typed prompts were sent directly to each local conversation
backend without opening a microphone or playing audio through a device. Both
LFM3 Omni Q4 and LFM2.5 Audio Q4 generated text and audio for “What is two plus
two?” and returned a correct text answer. In a separate two-turn name-recall
check, LFM3's generated text said “Your name is Alex,” while LFM2.5 said “I'm
Alex,” misattributing the name. This small check verifies the typed request
path and exposes a model-quality failure; it does not establish that generated
speech matches text or that conversation memory is reliable.

## Controlled spoken-search pipeline

On November 4, one device-free run generated a synthetic spoken request for
Python downloads, transcribed it with LFM3 Omni, and passed it through the app.
A deterministic local stand-in requested live Bing search and returned a fixed
answer only after receiving an official Python link. Bing supplied three source
links; the app attached them to the reply and generated 4.0 seconds of audio.
LFM3's separate ASR session transcribed that audio as the expected sentence.
The app turn took 4.464 seconds with a non-realtime audio sink. This checks the
integration path, not cloud-model tool choice or reasoning, microphone capture,
voice detection and endpointing, real speaker playback, or independent speech
recognition. Reproduce with the
app stopped and the LFM3 server running:

```powershell
python -m src.synth.spoken_search_probe --out outputs/spoken-search-probe.json
```

The probe contacts Bing with a fixed public query and starts a local-only mock
model server. It does not use the OpenCode key or record user audio.

## What remains open

On November 4, a live OpenCode Go `kimi-k3` search request returned HTTP 429,
so it did not exercise the model or spoken search end to end. An independent
Bing RSS query for Python downloads with a `site:python.org` restriction
returned three links on that domain. This checks provider reachability and
filtering for one query; it does not verify that snippets are current, that a
model uses them correctly, or that spoken search works on a device.

False interruptions, paused-sentence behavior and reply quality remain unresolved.
Controlled real-device acceptance and a clean-machine native installation have
not been established. The EOT held-out check is one deterministic English split;
multi-seed, multilingual and live-microphone validation remain open. Streaming
microphone append, heard-prefix memory across realtime reconnects, LFM3 tool calling,
and controlled long-session latency remain unexplored.
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

For answers exceeding 110 characters, the draft `Assistant._speak_answer`
groups sentences into separate TTS requests. The size limits are grouping
targets, not hard caps on an individual sentence. Synthesis for the first
group begins before the remaining groups are requested; tests exercise ordering
but do not establish audio quality or an end-to-end speed improvement.

An isolated LFM3 Omni Q4 CPU comparison on November 4 used one fixed
three-sentence text, in single/segmented/segmented/single order, with no
playback. First synthesized audio arrived at 1.287 and 0.616 seconds in the
single-call runs and 0.643 and 0.502 seconds in the segmented runs. Total
synthesis took 15.577 and 15.741 seconds versus 14.345 and 15.286 seconds,
respectively. The first single run was cold, and generated output lengths
varied. Two runs of one text do not establish a general speedup, naturalness,
audible gaps, content completeness, interruption behavior, or response time
including cloud generation and a physical device. The draft is not promoted
on this result. With a dedicated LFM3 server and the app stopped, reproduce
the protocol with:

```powershell
python -m src.synth.tts_segmentation_probe --url http://127.0.0.1:18086 --out outputs/tts-segmentation.json
```

The probe does not record or assess speech quality.

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

A second exploratory CPU study used 20 training prompts per direction, 10
evaluation prompts per score, rank 4, five trials, seed 42, and at most 32
generated tokens. The baseline refusal score was 9/10; the selected candidate
scored 7/10 with reported KL divergence 0.2227. The other four trials each
scored 9/10. These are in-sample optimization scores, not an independent test;
the small sample and short generation limit preclude a broad effectiveness or
quality claim. The saved model had 44 changed tensors, all in the language
backbone, and unchanged tensor names and remaining tensor values.

The selected candidate was converted and quantized to Q4_K_M (SHA-256
`55dd5a234f1343dd8429e64b83ab297ad9540fb0d01a6d877e99a07718990390`). On a separate
llama.cpp b12032 CPU server, using the original projector, a synthetic voice
turn produced the transcription “What is two plus two?” and the spoken reply
“Two plus two equals four.” This does not replace a human listening assessment,
a broader independent behavior evaluation, or tests of the remote agent.

An additional held-out text-only comparison used the next 10 prompts from the
same dataset's test split, with a 32-token limit and deterministic sampling on
the Q4_K_M llama.cpp CPU server. Heretic's keyword heuristic matched 10/10
responses from the original and 9/10 from the research candidate. This small
heuristic check is not a manual assessment and does not measure general model
quality or speech behavior. The two Q4 files were quantized separately.

### Search-provider check

In a local, text-only check, DuckDuckGo returned an automated-request challenge
instead of usable web results. The previous Wikipedia fallback returned unrelated
pages for a site-restricted query. The agent now reports the fallback limitation
and avoids that substitution when a site restriction is present. A separate
test of the optional Bing RSS provider returned four `python.org` sources for
“Python latest version site:python.org”; a differently worded query returned no
matching sources. These spot checks do not establish search reliability or a
complete live voice-to-search-to-speech flow.

A later isolated synthetic-speech probe used local LFM3 speech, OpenCode Go's
`iceman-contributor-free` model and Bing RSS. It completed transcription, live
tool calls, source collection and speech synthesis, but initially gave an
outdated Python version while citing the [official downloads page](https://www.python.org/downloads/),
which listed 3.15.2 at the time of the check. The agent now
blocks a dotted version number in a searched answer when that number does not
appear in the returned search titles or snippets. In a repeat of the probe it
spoke an uncertainty message instead. This limited string check can reject
valid claims and does not establish factual correctness or relevance of other
claims. The `beam-free` model produced empty final replies during two probes,
and `kimi-k3` returned a usage-limit error. No human microphone or speaker
assessment was performed in these checks.

The later review of the local LFM3 agent logs found two applicable failure
modes: models mistaking a speech transcript for typed input, and tool requests
arriving as spoken text. Turnwave now marks the current input modality in the
cloud prompt and rejects textual tool-call markup rather than presenting it as
an executed tool. Unit tests cover both paths; the live Go check at that point did not
complete because the provider request failed; later synthetic checks are reported below.

## Live synthetic spoken search on the merged app

On November 4, 2026, two trials of the same generated question, “Search the
web for Python downloads on python dot org,” passed through LFM3 Omni Q4 local
transcription, a live OpenCode Go `beam-free` model, Bing RSS search, and LFM3
synthesis. Both called the search tool, returned five source links including
the official Python downloads page, and generated reply audio. The first run
produced 211,200 samples and took 21.95 seconds; the included probe produced
318,720 samples and took 22.72 seconds. Times include generating the synthetic
input, and the fake playback consumes audio faster than real time. LFM3 ran on
CPU through an isolated local llama-server; model output and external services
are variable. Earlier requests included HTTP 429 and empty replies.

These are two repetitions of one synthetic prompt, not a general accuracy or
latency benchmark. No microphone or speaker was opened, and no human judged the
audio. The source check establishes that an official link was returned, not that
all claims are correct. Provider availability, terms and charges can change.
To rerun, stop the app, start a separately owned LFM3 server at the chosen URL,
install the optional `scipy` package, and set the Go API key in the process
environment. This makes live model and search requests and may consume quota:

```powershell
python -m src.synth.live_search_probe --url http://127.0.0.1:18085 --model beam-free --out outputs/live-search.json
```

The probe refuses an active app on port 8000 and uses the backend ownership lock;
it does not detect unrelated applications that ignore that lock. Use a dedicated
backend instance. Output contains the synthetic prompt, model reply, and source
URLs; do not use it with private material.
