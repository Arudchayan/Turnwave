# Known limitations

This is a research prototype, not a claim of universally reliable conversation.

- A natural pause can end a turn too early. Current silence endpointing is about
  480 ms, with a 20-second capture cap and pre-roll.
- Speaker echo, quiet speech, room acoustics and device routing affect genuine
  and false interruption detection. Headphones are more predictable.
- A clean output queue does not guarantee intelligible speech. Generated text
  can disagree with generated audio, and correct content is a separate question.
- Exact instructions, attribution and conversation memory can fail. Interruptions
  reset context for recovery and therefore do not preserve all prior memory.
- First-audio timings exclude endpointing unless explicitly measured otherwise.
- Windows/WSL and the external native runner/toolchain are the intended setup.
  Hardware support and dependency installation need validation on each machine.
- Model weights and native binaries are not distributed here. Research scripts
  may require prior local artifacts that are intentionally not published.
- Local hash-verified copies protect against some accidental loss, not loss of
  both storage locations. Device-write success is not proof of acoustic playback.

Automated tests check application behavior using simulated faults and generated
fixtures. Real-device acceptance remains necessary. Private recordings, personal
session notes and historical transcripts are excluded from publication.
