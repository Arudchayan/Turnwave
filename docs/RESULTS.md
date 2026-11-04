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

## LFM3 local voice feasibility

On November 2, one synthetic spoken phrase was submitted three times to each
local backend on an RTX 3050 Laptop GPU. First decoded audio was 0.139–0.212 s
for LFM2.5 Audio Q4 and 0.369–0.420 s for LFM3 Omni Q4. Generated replies
were different lengths, and the test did not include microphone capture,
endpointing, room acoustics, independent output ASR, or human listening. The
measurements do not establish a general performance or quality ranking.

One interruption trial stopped the LFM3 client 16 ms after abort and then
generated a fresh-session response. That does not measure end-to-end device
interruption latency or verify conversation context after interruption.

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

## Adjustable pause between turns

The interface offers 480, 800 and 1,120 ms silence waits. In one CPU replay
through the Silero frontend with echo processing disabled and no holdoff, the
SAPI phrase “Hello, can you hear me clearly?” was trimmed to 100 ms around samples above 0.008 absolute
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
50-case probe with:

```powershell
python -m src.synth.echo_tail_probe --out outputs/echo-tail-probe.json
```
