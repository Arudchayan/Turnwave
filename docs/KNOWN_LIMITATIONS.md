# Known limitations

This is a research prototype, not a claim of universally reliable conversation.

- A natural pause can end a turn too early. Current silence endpointing defaults
  to 480 ms (adjustable to 800 or 1,120 ms between utterances), with a 20-second
  capture cap and pre-roll. Longer waits add reply latency and still cannot
  guarantee that an unfinished sentence stays together.
- Speaker echo, quiet speech, room acoustics and device routing affect genuine
  and false interruption detection. Headphones are more predictable.
- A clean output queue does not guarantee intelligible speech. Generated text
  can disagree with generated audio, and correct content is a separate question.
- Exact instructions, attribution and conversation memory can fail. Interruptions
  reset the local audio model context for recovery; in cloud-agent mode,
  recognized user requests are retained in memory across interruptions, while
  unfinished assistant replies are excluded.
- Cloud conversation and web search depend on external network reachability,
  provider quotas and search-result quality. OpenCode Go is described by the
  provider as intended for coding-agent traffic and enforces account usage
  limits; DuckDuckGo HTML search is a keyless fallback that can be rate-limited
  or challenged by upstream bot filters.
- First-audio timings exclude endpointing unless explicitly measured otherwise.
- Offline content benchmarks must grade generated speech separately from the
  model's text. Independent recognition can also be wrong, particularly on short
  or unclear clips; preserve disagreements for listening review. A completed-WAV
  content test bypasses microphone endpointing, echo and interruption behavior.
- Windows/WSL and the external native runner/toolchain are the intended setup.
  Hardware support and dependency installation need validation on each machine.
- Model weights and native binaries are not distributed here. Research scripts
  may require prior local artifacts that are intentionally not published.
- Local hash-verified copies protect against some accidental loss, not loss of
  both storage locations. Device-write success is not proof of acoustic playback.

Automated tests check application behavior using simulated faults and generated
fixtures. Real-device acceptance remains necessary. Private recordings, personal
session notes and historical transcripts are excluded from publication.
