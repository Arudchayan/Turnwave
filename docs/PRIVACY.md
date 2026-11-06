# Data handling

## Local inference

Normal inference runs against the configured local server. Installing models and
dependencies may contact external services. Configuring a remote endpoint sends
requests there instead.

The microphone stream opens at startup. `LOCALVOICE_START_MUTED=1` prevents
submitting microphone turns until unmuted. The browser transcript stays in memory.

## Optional cloud conversation

When cloud conversation is enabled, processed microphone audio is sent to the
configured voice server for transcription; the recognized text and recent
conversation are sent to OpenCode Go or Zen. An interrupted request remains
in memory, but an unfinished assistant reply is not added to later context.
The reply is sent to the voice
server for synthesis. If a Brave Search key or `LOCALVOICE_SEARCH=ddg` is
configured and the model calls search, the query is sent to Brave or DuckDuckGo
and returned snippets are sent to OpenCode Go or Zen. These services have their
own retention and privacy policies. The default mode does not make these cloud
requests. If the configured voice server is remote, audio and synthesis requests
go to that server in either mode.

API keys are read from the application process environment, not delivered to the
browser or included in diagnostics. Opt-in session archives contain the input
WAV, generated audio, result text and the local synthesis request for agent
turns; they do not capture the cloud request or search results as exact request
records. Treat the resulting transcript and audio as sensitive.

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
