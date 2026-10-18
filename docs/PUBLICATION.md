# Publishing safely

The development repository can contain private historical commits even after
its current files are cleaned. Do not publish its branches, tags, bundles or a
mirror. `.gitignore` does not scrub old commits or detect sensitive words.

Use the separately prepared public repository for the initial public push. It contains reviewed current files and a new root commit, with an independently reviewed history. The published project is Turnwave;
confirm its GitHub destination before pushing. Check the exact folder, branch and remote before publishing. Never
copy the development `.git` directory into it.

```powershell
python scripts/check_publication.py --worktree
python scripts/check_publication.py
git status --short
git log --oneline --all
git ls-files
```

Review commit author/email metadata as well as file contents. The clean snapshot
uses a neutral contributor identity. Enable GitHub secret scanning/push protection
where available. Credentials already published require revocation independently
of Git cleanup. No credential found by a limited scan is a guarantee of absence.

Public source includes optional research tools and curated, sanitized numerical
results. Human recordings, model weights, raw requests/logs and internal handoffs
stay local. Exactly two reviewed complete generated reply WAVs are allowed under
`research/examples/`. They are fictional synthetic scenarios, not user recordings.
`research/publication-manifest.json` pins each allowed research asset by path,
size, type and SHA-256. Chart and JSON assets use the same closed allowlist.
Any changed bytes require review and a manifest update; arbitrary audio remains
blocked. The default checker reads index blobs; `--worktree` also checks tracked
and nonignored untracked working files before staging. Both scans are heuristic.
The maintainer explicitly chose to leave the original application code
unlicensed for now. Do not describe it as licensed open source. Third-party
model/runner/patch terms remain separate; no external license is replaced here.
