# Research evidence

This directory contains a small, reviewed evidence ledger and case summaries for Turnwave. It is designed to make the published counts and charts inspectable; it is not a leaderboard or a single assistant-quality score.

Start with [the evaluation guide](../docs/EVALUATION.md) for evidence classes, denominators, results, limitations and reproduction commands. The curated ledger is [`results.json`](results.json). Authored short-turn case details are in [`short-turn-cases.json`](short-turn-cases.json) and [`candidate-screen-cases.json`](candidate-screen-cases.json). Reviewed generated-response examples and their explicit asset manifest are described in [`examples/README.md`](examples/README.md).

The headline evidence includes deterministic playback-race controls, a selective-opening acoustic gate that failed, a narrow 7/7 combined development suite, rejected short-turn classifier rules, one six-pair CPU capture comparison with no baseline/candidate capture difference, and one fixed 24-request native content slice. Each result keeps its own denominator and limits. The capture candidate is not adopted; its recovery efficacy remains unproven.

The latest capture comparison used only the current capture path and an isolated private `OpeningBuffer` candidate. It was step-paced in 20 ms blocks on CPU, with no backend service, physical audio device or live application. Both arms produced identical detector events and capture bytes in all six cases. Two earlier setup attempts stopped before model or frontend construction; they are setup failures with zero model cases, not evidence of model failure.

## Reproduce the included evidence

From the repository root, the figure builder reads curated local JSON and does not perform inference:

```powershell
python scripts/verify_research.py
python -m pip install -r requirements-docs.txt
python scripts/build_research_figures.py
```

The verifier recomputes the frozen short-turn counts, checks the six-pair capture denominator and decision, and validates the two example WAV hashes and formats. The public unit and publication checks are:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python scripts/check_publication.py --worktree
```

The figure builder is a reporting tool. Reproducing counts and plots does not reproduce historical neural responses. Exact reruns also need the frozen stimuli, model/runtime assets, source revisions, configuration and hardware recorded in the historical protocol; those are not all distributed here. See [Evaluation](../docs/EVALUATION.md) before interpreting any measure.

The content-benchmark adapter's `prepare` and `export` commands validate and transform local files without inference. The explicit Python API is required for caller-supplied inference, and no official benchmark or judge is run by these commands. See the adapter section in the evaluation guide.

## Publication boundary

Only reviewed numerical summaries and the individually approved generated examples are included. Human recordings, local SAPI stimuli, third-party benchmark audio/prompts, raw private traces, model weights, native binaries and private Git history are excluded. The checked-in publication manifest authorizes only exact reviewed research assets; adding a chart or audio file requires adding its exact hash and provenance to that manifest.

## Figures

These charts summarize the frozen authored-case results. Counts are descriptive and are not population estimates.

![Observed extra time from first endpoint to release for each authored short-turn case](../docs/assets/short-turn-delay.svg)

![Complete-intent positive decisions and unfinished-prefix holds for the rejected fixed-endpoint audio OR screen](../docs/assets/completion-screen.svg)
