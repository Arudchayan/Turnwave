# Data handling

Normal inference runs locally against the configured server. Model/dependency
installation may access external services; optional custom endpoints change
where requests go. Check configuration before use.

The microphone stream opens at startup. `LOCALVOICE_START_MUTED=1` prevents
submitting microphone turns until unmuted; it does not mean the input device
has never opened. The UI transcript is in memory. Normal app diagnostics under
`outputs/logs/` contain timing, device/state and error metadata; errors and native
server logs may include additional information and should be treated as private.

`LOCALVOICE_RECORD_SESSION` explicitly enables per-turn archives. They contain
processed input WAVs, exact request field values, generated audio, reported text,
timings and cancellation/playback metadata. The separate physical recorder saves
raw microphone and actual system output, potentially including unrelated sounds.
Use it only with participants' knowledge.

Data, outputs, models, credentials, local agent notes and recordings are ignored
by default and must not be committed. Saved experiments may also have backup
copies outside the repository. Deleting or ignoring one file does not remove
those copies or past Git objects. No automatic retention/deletion policy exists.

The publication checker detects selected risky filenames and text patterns. It
is a useful guard, not a complete secret detector, content classifier or privacy
guarantee. Review the actual publication snapshot and its history before pushing.
