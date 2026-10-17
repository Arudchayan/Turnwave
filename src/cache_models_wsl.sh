#!/bin/bash
# Put model weights on WSL ext4 to avoid slow mmap page faults through /mnt/e.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE="$ROOT/models/LFM2.5-Audio-1.5B-GGUF"
CACHE="$HOME/.cache/localvoice/models"
LM="${1:-}"
if [ -z "$LM" ]; then
  if [ -f "$SOURCE/LFM2.5-Audio-1.5B-Q8_0.gguf" ]; then
    LM=LFM2.5-Audio-1.5B-Q8_0.gguf
  else
    LM=LFM2.5-Audio-1.5B-Q4_0.gguf
  fi
fi
case "$LM" in LFM2.5-Audio-1.5B-Q4_0.gguf|LFM2.5-Audio-1.5B-Q8_0.gguf) ;; *) echo 'Unsupported model filename' >&2; exit 1;; esac
mkdir -p "$CACHE"
for name in "$LM" mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf; do
  if [ -f "$CACHE/$name" ] && cmp -s "$SOURCE/$name" "$CACHE/$name"; then
    echo "Verified cached $name"
  else
    echo "Caching $name on WSL ext4"
    cp -- "$SOURCE/$name" "$CACHE/$name.tmp.$$"
    cmp -- "$SOURCE/$name" "$CACHE/$name.tmp.$$"
    mv -- "$CACHE/$name.tmp.$$" "$CACHE/$name"
  fi
done
touch "$CACHE/.ready-$LM"
echo "Verified model cache: $CACHE"
