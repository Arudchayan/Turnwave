# Two generated reply examples

These complete WAVs were saved during LFM2.5-Audio runs and copied byte for byte.
They were selected after outcomes were known, so they illustrate individual
outputs rather than estimate performance.

Both conversations began: “For this test, I bought a blue lamp yesterday. Please
remember what I bought and acknowledge briefly.” The following questions differed
in wording and synthetic input voice, so this pair cannot isolate why the
outputs differ.

| Field | Text-rubric pass | Text-rubric failure |
| --- | --- | --- |
| Recall question | What did the user buy yesterday? | What did I buy yesterday? |
| Input voice | Synthetic Alto (David +35% pitch) | David |
| Model-emitted text | The user bought a blue lamp yesterday. | I bought a blue lamp yesterday. |
| Independent machine transcription | The user bought a blue lamp yesterday | I got a blue lamp yesterday. |
| Text rubric | Facts, ownership and instructions pass | Ownership fails: assistant makes a first-person claim |
| Whole saved reply | [Listen: 2.30 s](pass-reply.wav) | [Listen: 2.38 s](failure-reply.wav) |

The rubric grades **model-emitted text**. A separate CPU base.en ASR system
transcribed “got” where the model text says “bought” in the second example.
No person verified which transcript matches the recording or evaluated spoken
intelligibility.

Both replies ended normally. WAV files are mono PCM16 at 24 kHz, copied byte for
byte with no trimming, normalization or re-encoding. [examples.json](examples.json)
records exact SHA-256 values, sample counts, source case/result hashes, model
asset and runner hashes, generation settings and ASR revision. Historical source
also had uncommitted tooling changes; its commit alone cannot reproduce the run.

Only authored fictional text and generated response audio are distributed here.
No human recording, original synthetic input WAV, embedded request, private log,
third-party benchmark clip or model asset is included. Read the separate
[model and dependency terms](../../docs/THIRD_PARTY.md).
