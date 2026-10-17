#!/bin/bash
# Apply the reviewed LocalVoice fixes to the pinned backend, then rebuild.
# Stop the server first. Does not fetch a moving branch or reset local edits.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE="${LOCALVOICE_RUNNER_SOURCE:-$HOME/liquid-build}"
PATCHES=("$ROOT/patches/liquid-turn-reset.patch"
         "$ROOT/patches/liquid-attachment-reset.patch"
         "$ROOT/patches/liquid-token-lifetime.patch"
         "$ROOT/patches/liquid-token-limit.patch"
         "$ROOT/patches/liquid-abort-lifecycle.patch")
REVISION="$(git -C "$SOURCE" rev-parse --short=9 HEAD)"
if [ "$REVISION" != ec9d1fdd9 ]; then
  echo "Expected runner ec9d1fdd9, found $REVISION; review patch compatibility first." >&2
  exit 1
fi
if pgrep -f "^$SOURCE/build/bin/llama-liquid-audio-server " >/dev/null; then
  echo "Stop the Liquid audio server before rebuilding its loaded libraries." >&2
  exit 1
fi
for PATCH in "${PATCHES[@]}"; do
  if git -C "$SOURCE" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "Already applied: $(basename "$PATCH")"
  else
    git -C "$SOURCE" apply --check "$PATCH"
    git -C "$SOURCE" apply "$PATCH"
  fi
done
# Avoid CMake scanning inherited Windows PATH entries through slow WSL mounts.
export PATH="$HOME/tools/cmake-3.30.5-linux-x86_64/bin:$HOME/miniforge/envs/cudaenv/bin:/usr/local/bin:/usr/bin:/bin"
cmake --build "$SOURCE/build" -j4 --target llama-liquid-audio-server
