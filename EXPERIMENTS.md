# Experiment ledger

This ledger records what was tested, what happened, and what the evidence does
not prove. Experiments below marked **draft** were run on the voice-agent
feature branch and are not yet available on the default branch. The
[fixed research snapshot](https://github.com/Arudchayan/Turnwave/blob/4114920c50f14efd3823f2ddd0b00acf8901e577/docs/RESULTS.md)
contains the detailed methods, controls and reproduction commands. Current
baseline studies are documented in [results](docs/RESULTS.md) and
[evaluation](docs/EVALUATION.md).

| Study and status | Comparison and observed result | Decision and limits |
| --- | --- | --- |
| Playback completion — baseline | Forced producer/consumer ordering: 0/2,003 versus 2,003/2,003 samples written. | Fix retained; fake devices do not establish how often the race occurred or what a listener heard. |
| Endpoint detector — baseline | Authored 16-case diagnostic: 5/8 complete prompts released within the timing target, and 8/8 unfinished prefixes were held. | Timing target failed; not an estimate of natural-conversation accuracy. |
| Adjustable pause — merged | One synthetic voice with 650 and 850 ms inserted gaps: 480 and 800 ms settings split both; 1,120 ms joined both. | User-adjustable; the joined 650 ms case added 640 ms to the final endpoint. One voice does not establish real-world safety. See [results](docs/RESULTS.md). |
| Reversible floor — draft | Three authored backchannels were preserved, but the acoustic double-talk replay still cancelled its short backchannel. | Remains opt-in; authored shapes and one synthetic acoustic pair do not prove a room or headset improvement. |
| Delayed echo — draft | In a 50-case synthetic comparison, echo-only starts fell from 3/5 to 0/5; both profiles detected 10/10 near-only controls. One quiet overlap was detected 850 ms after speech began. | Longer history remains opt-in; one far/near pair and simulated delay do not validate real devices. |
| End of turn — draft | One conversation-grouped held-out English split: fixed 480 ms gave 63.92% per-pause false cut-offs; a Smart Turn schedule with a 1 s timeout gave 31.37%. | Opt-in research; the result is per pause, from one split, and not a live microphone test. |
| Context after interruption — draft | Three simulated cancellation follow-ups with recovery remained on topic, but repeated a suggestion; first audio took 3.995–7.134 s. | Recovery remains opt-in; one synthesized prompt pair and no actual device playback. |
| Direct audio context — draft | Three paired simulated cancellations: replay stayed on topic, but repeated suggestions in two cases and timing varied across runs. | Research probe only; no reliable latency improvement established. |
| Local model modification — draft | A small held-out keyword heuristic matched 10/10 original and 9/10 modified model replies. | Compatibility and narrow behavior check only; no broad effectiveness or speech-quality claim. Model weights are not published here. |
| Controlled spoken search — draft | Synthetic speech passed through local transcription, live Bing search, a deterministic model stand-in, source attachment and local synthesis; output ASR recovered the sentence. | Pipeline check only. It does not evaluate a cloud model, microphone endpointing, human listening or independent ASR. |
| Typed local conversation — draft | Both local backends generated audio and correct text for one arithmetic prompt; in a follow-up the older backend attributed the user's name to itself. | Request-path smoke test with a documented failure; not a general model-quality score. |

Real-device voice quality, interruption behavior, natural pauses, and a complete
human spoken search remain acceptance work. A prior OpenCode Go probe returned
HTTP 429, so a successful current-provider run must be verified separately.

## Recording new experiments

- State the question, baseline and candidate, exact input set and sample count.
- Record environment, configuration, commit or artifact identity, and metrics.
- Include failures and exclusions, and distinguish generated, written and heard
  audio. Describe whether each stage is real, simulated or mocked.
- Link a reproducible command and durable non-sensitive evidence where possible.
- Make the decision explicit: adopted, opt-in, rejected, or inconclusive.
  Never publish credentials, raw private audio or personal session logs.
