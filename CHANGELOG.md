# Change log

Changes are listed when they reach the default branch. Work in an open pull
request is described separately so experimental behavior is not mistaken for a
released default. No version numbers are assigned here without a release tag.

## Unreleased

- Fixed cloud follow-up context when a device write completes just after an
  interruption, while keeping reset conversations separate.
- Kept web sources attached to their corresponding replies through follow-up
  turns, with safe links and reset behavior.
- Added a capture-loop interruption benchmark and documented its false-cancel
  and quiet-speech observations; interruption behavior is unchanged.
- Preserve a cloud reply draft as possibly unheard context when playback is
  interrupted after audio has been accepted, improving follow-up continuity.
- Added a reproducible synthetic spoken-search probe and recorded two live
  provider trials, with their device and reliability limits.
- Added conversation mode, model, and search controls. Changing mode or model
  starts a fresh conversation to avoid sending old context to another provider.

- Added an experimental opt-in longer echo tail for delayed playback feedback.
  Quiet interruptions can be delayed; real-device validation is still needed.

- Added optional cloud conversation with local speech processing, web search,
  source links, and preserved questions after interruptions. Live provider and
  real-device spoken-search validation remain outstanding.

- Added optional LFM3 Omni local voice-backend support with a separate WSL
  launcher; the existing LFM2.5 backend remains the default.
- Refined the conversation interface with responsive controls and clearer
  connection and processing status.
- Added an adjustable pause before replying, with quick, balanced, and patient
  settings. The default remains 480 ms; longer pauses add latency.
- Added a public change log and an experiment ledger with outcomes, decisions,
  reproduction references, and evidence limits.

## In development

The [voice and agent development pull request](https://github.com/Arudchayan/Turnwave/pull/1)
contains additional work on local voice, agent behavior, the interface, and
research tooling. Some of that draft work is **not yet on the default branch**.
Features remain experimental or opt-in and still need real-device validation. See [EXPERIMENTS.md](EXPERIMENTS.md) for measured outcomes and
[known limitations](docs/KNOWN_LIMITATIONS.md) for the current baseline.

## Baseline

- The existing default branch contains the local LFM2.5 audio prototype and its
  documented synthetic and native-model studies. See [results](docs/RESULTS.md)
  and [evaluation methods](docs/EVALUATION.md).

## Maintenance

For each merged pull request, add a concise user-facing entry to **Unreleased**.
For each experiment, record the comparison, denominator, environment, outcome,
limitations, and decision in the experiment ledger or a linked detailed report.
Do not include private recordings, transcripts, credentials, or local machine
paths. Move entries into a dated version section only when a release is tagged.
