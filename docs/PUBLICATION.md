# Publishing safely

The development repository can contain private historical commits even after
its current files are cleaned. Do not publish its branches, tags, bundles or a
mirror. `.gitignore` does not scrub old commits or detect sensitive words.

Use the separately prepared `LocalVoice-Public` repository for the initial public
push. It contains reviewed current files and a new root commit, with no remote
configured. Check the exact folder, branch and remote before publishing. Never
copy the development `.git` directory into it.

```powershell
python scripts/check_publication.py
git status --short
git log --oneline --all
git ls-files
```

Review commit author/email metadata as well as file contents. The clean snapshot
uses a neutral contributor identity. Enable GitHub secret scanning/push protection
where available. Credentials already published require revocation independently
of Git cleanup. No credential found by a limited scan is a guarantee of absence.

Public source includes optional research tools, not private fixtures/results.
User audio, model weights, generated replies and internal handoffs stay local.
The maintainer explicitly chose to leave the original application code
unlicensed for now. Do not describe it as licensed open source. Third-party
model/runner/patch terms remain separate; no external license is replaced here.
