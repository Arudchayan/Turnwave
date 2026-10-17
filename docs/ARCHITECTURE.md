# Architecture

LocalVoice wraps a turn-based speech model with a continuously listening app.
It is not a native full-duplex speech model.

1. `audio_io.py` captures the selected input, selects an active channel and
   resamples to 16 kHz mono. A bounded queue prevents unbounded input latency.
2. `duplex.py` combines echo reference processing, voice activity detection,
   pre-roll and silence-based endpointing.
3. `app.py` coordinates capture, cancellable requests, transcript updates and
   serialized speaker playback. Speaking during a reply can abort it and begin
   a replacement turn.
4. `client.py` sends WAV input to the Liquid audio runner, reads streaming
   text/audio events and decodes output at 24 kHz.
5. `audio_io.Player` feeds an early echo reference and separately counts source
   samples after successful device writes. Accepted writes do not prove audibility.
6. `session_archive.py` optionally retains exact requests/generated samples and
   separate playback metadata. Diagnostics are distinct from an audio archive.

An OS-backed backend lease prevents app and test processes from sharing the same
stateful model conversation. Interrupted turns reset model context because the
runner cannot truncate context to the portion of a reply actually heard.

The browser UI is embedded in `ui.py`. It talks to a loopback HTTP service in
`app.py`. No cloud API is part of the normal inference path.

Browser controls send explicit microphone state, suppress duplicate requests
per control and show request failures/timeouts beside existing warnings. An
unrelated pending control does not disable interruption. Stale state polls
cannot reverse a completed microphone intent.

Microphone indices belong to the app's PortAudio inventory. Automation should
match the full `mic_devices` descriptors returned by `/api/state` and submit that
app-local index to `/api/micdev`; an index from another process can identify a
different endpoint. `mic_identity` reports the opened input's cached name, host
and capture format only when the metadata is complete and uniquely matches the
catalog. Unknown or ambiguous identity is `null`. State polling does not query
audio hardware. The existing short `mics` labels remain available to the UI.
