# Contributing

Use Python 3.12. Install `requirements-dev.txt`, then run:

```powershell
python -m pytest tests -q
python scripts/check_publication.py
```

Tests use generated fixtures, simulated devices and temporary local HTTP servers.
They should not require model weights, saved experiments or physical recording.
Node.js enables the browser-script checks; install it to avoid skipping those.

Keep capture, generation, device submission and heard audio distinct in reports.
Document failing controls and limits, not just passing totals. Changes to echo
handling or endpointing need held-out speech and quiet-playback controls.

Backend research tools under `src/synth/` are experimental command-line programs,
not automatic startup tasks. Some need externally supplied audio, prior local
experiments, WSL toolchains or a stopped application. Review their prerequisites
before use. Never run a test against a model context owned by a live app.

Do not commit recordings, transcripts, credentials, model weights, machine
configuration or private research notes. `.gitignore` does not remove existing
tracked files or history. See [publication guidance](docs/PUBLICATION.md).
