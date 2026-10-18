# Sharing results

Session archives can contain voices, transcripts and exact model requests.
Review recordings and metadata before sharing them, and obtain permission from
recorded participants. See [data handling](PRIVACY.md) for what the app stores.

## Check a contribution

```powershell
python scripts/check_publication.py --worktree
git diff
git diff --cached
python scripts/check_publication.py
```

The worktree check includes tracked and nonignored untracked files. The default
check reads staged Git blobs. These checks detect selected filenames, credentials
and personal paths; manual content review is still needed. Ignoring or deleting
a file does not remove copies from Git history or backups.

## Add a research asset

Published figures, summary data and the two synthetic reply examples are listed
in `research/publication-manifest.json`. Each entry records its path, size, type,
source and SHA-256. The checker accepts only the specific paths declared in
`scripts/check_publication.py` with matching manifest entries.

To add or change an asset:

1. Check the content, provenance and redistribution terms.
2. Remove personal information and unrelated session data.
3. Update the path allowlist and manifest, then run both publication checks.
4. Include the experiment method, result and limitations with the asset.

Raw sessions, model weights and native binaries remain outside this repository.
Application code and documentation are [MIT licensed](../LICENSE); retain
applicable [third-party notices](THIRD_PARTY.md) when sharing derived work.
