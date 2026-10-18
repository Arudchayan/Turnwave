# Third-party components

Turnwave's application code and documentation use the [MIT license](../LICENSE).
Models, native runners and dependencies retain their own licenses. The
llama.cpp notice for source included in the patches is preserved in
[third-party notices](../THIRD_PARTY_NOTICES.md).

| Component | Role | Source / terms to review |
| --- | --- | --- |
| LiquidAI LFM2.5-Audio-1.5B | Direct audio input/output model | [Model card](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B), [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B/blob/main/LICENSE) |
| llama.cpp Liquid audio integration | External native GGUF runner; local derived patches | [PR 18641](https://github.com/ggml-org/llama.cpp/pull/18641), [upstream license at pinned source](https://github.com/ggml-org/llama.cpp/blob/ec9d1fdd9cc18643c5e161b65b8053f6f6f34a4b/LICENSE) |
| pywebrtc-audio | Echo processing wrapper | [Project](https://pypi.org/project/pywebrtc-audio/) and installed distribution notices |
| Silero VAD | Voice activity detection | [Upstream](https://github.com/snakers4/silero-vad) |
| ONNX Runtime | CPU inference runtime | [Upstream](https://github.com/microsoft/onnxruntime) |
| sounddevice / PortAudio, NumPy | Device I/O and arrays | [sounddevice](https://python-sounddevice.readthedocs.io/), [PortAudio](https://www.portaudio.com/), [NumPy](https://numpy.org/) |

Models, native binaries and vendored dependency distributions are not included.
Obtain them separately and retain applicable notices. The native source pin is
`ec9d1fdd9cc18643c5e161b65b8053f6f6f34a4b`; model asset hashes used by the
illustrative replies are recorded in [examples.json](../research/examples/examples.json).

## Research evidence

The two response WAVs are genuine generated output from neutral, authored
fictional scenarios, included for research documentation with model attribution.
They contain no human recording. Microsoft SAPI-rendered input WAVs are not
included. Uses of model-generated audio are subject to the model's terms.

The capture comparison used two LibriSpeech utterances from the Hugging Face
`hf-internal-testing/librispeech_asr_dummy` fixture at revision
`5be91486e11a2d616f4ec5db8d3fd248585ac07a`. The source lineage is
[LibriSpeech / OpenSLR 12](https://www.openslr.org/12), which states CC BY 4.0;
the fixture card itself does not declare a license. Only numerical observations
and source identifiers are published here, not those human-read recordings.

The adapted content slice used [VoiceBench](https://github.com/MatthewCYM/VoiceBench)
inputs at revision `b02edcef1330480be3a11bd6f7434ac32f05ad08`, with IFEval and
OpenBookQA scoring concepts. Source dataset prompts and audio are not redistributed.
This is not an official full benchmark submission. Semantic detector studies used
[Namo](https://huggingface.co/videosdk-live/Namo-Turn-Detector-v1-English) and
[Turn-1-mini](https://huggingface.co/p99lab/turn-1-mini); output transcription used
[faster-whisper base.en](https://huggingface.co/Systran/faster-whisper-base.en).
Frozen revisions are in the public evidence JSON. Review each upstream's terms
before obtaining assets or rerunning studies.
