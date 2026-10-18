# Data handling

## Local inference

Normal inference runs against the configured local server. Installing models and
dependencies may contact external services. Configuring a remote endpoint sends
requests there instead.

The microphone stream opens at startup. `LOCALVOICE_START_MUTED=1` prevents
submitting microphone turns until unmuted. The browser transcript stays in memory.

## Saved data

Diagnostics under `outputs/logs/` contain timings, device/state information and
errors. Native server logs and error messages may contain request-related data;
treat them as private.

Setting `LOCALVOICE_RECORD_SESSION` enables per-turn archives: processed input
WAVs, exact request fields, generated audio, model text, timings, cancellation and
playback metadata. The separate physical recorder captures raw microphone and
system output, which can include audio from other applications. Record only with
participants' knowledge. See [recorded sessions](RECORDED_SESSIONS.md).

Recordings and experiment outputs are ignored by Git. They can also have backup
copies outside the repository. There is no automatic retention or deletion policy;
removing a file from one location does not remove backups or committed copies.

## Shared examples

The repository contains two generated replies to fictional synthetic prompts,
alongside numerical summaries and figures. No participant recordings, original
synthetic input WAVs, benchmark audio or raw session requests are included.
The [publication guide](PUBLICATION.md) describes content review and asset checks.
