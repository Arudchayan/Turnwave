#!/bin/bash
# Launch llama-liquid-audio-server in WSL2 with GPU offload.
# Usage: bash src/server_wsl.sh   (from repo root, or via: wsl -d Ubuntu bash src/server_wsl.sh)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Preserve stdout/stderr under a unique name on every launch. The tee also
# keeps the terminal useful; no previous server session is truncated here.
mkdir -p "$ROOT/outputs/logs"
LOG_FILE="$ROOT/outputs/logs/server-$(date -u +%Y%m%dT%H%M%S)-$$.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Turnwave launch: $(date -u +%FT%TZ) pid=$$"
CKPT="$ROOT/models/LFM2.5-Audio-1.5B-GGUF"
if [ -z "${LM:-}" ]; then
  if [ -f "${LOCALVOICE_MODEL_DIR:-$CKPT}/LFM2.5-Audio-1.5B-Q8_0.gguf" ]; then
    LM=LFM2.5-Audio-1.5B-Q8_0.gguf
  else
    LM=LFM2.5-Audio-1.5B-Q4_0.gguf
  fi
fi
if [ -n "${LOCALVOICE_MODEL_DIR:-}" ]; then
  CKPT="$LOCALVOICE_MODEL_DIR"
elif [ -f "$HOME/.cache/localvoice/models/.ready-$LM" ]; then
  CKPT="$HOME/.cache/localvoice/models"
fi
RUN="$ROOT/models/runner/llama-liquid-audio-ubuntu-x64"
# Prefer the validated local GPU build when available. CUDA_BUILD=0 forces
# the prebuilt CPU runner; CUDA_BUILD=1 requires the local build.
GPU_RUN="${LOCALVOICE_GPU_RUNNER:-$HOME/liquid-build/build/bin}"
RUN_LIB=""
if [ -z "${LOCALVOICE_GPU_RUNNER:-}" ] && [ "${CUDA_BUILD:-auto}" != "0" ]; then
  # An activated local candidate must pass its saved binary/library hashes.
  # A damaged selection fails visibly instead of silently loading another build.
  SELECTION="$(python3 "$ROOT/src/runner_selection.py")" || exit 1
  if [ -n "$SELECTION" ]; then
    mapfile -t SELECTED_RUNNER <<< "$SELECTION"
    GPU_RUN="${SELECTED_RUNNER[0]}"
    RUN_LIB="${SELECTED_RUNNER[1]}"
  fi
fi
if [ "${CUDA_BUILD:-auto}" != "0" ] && [ -x "$GPU_RUN/llama-liquid-audio-server" ]; then
  RUN="$GPU_RUN"
elif [ "${CUDA_BUILD:-auto}" = "1" ]; then
  echo "CUDA runner missing at $GPU_RUN" >&2
  exit 1
fi
if [ ! -x "$RUN/llama-liquid-audio-server" ]; then
  echo "Audio server not found: $RUN/llama-liquid-audio-server" >&2
  exit 1
fi
for artifact in "$LM" mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf; do
  [ -f "$CKPT/$artifact" ] || { echo "Missing model file: $CKPT/$artifact" >&2; exit 1; }
done
echo "Turnwave server session: $(date -u +%FT%TZ) pid=$$ model=$LM"
echo "Model directory: $CKPT; runner: $RUN; log: $LOG_FILE"
export LD_LIBRARY_PATH="$RUN:${RUN_LIB:+$RUN_LIB:}${LD_LIBRARY_PATH}"
MMFLAG=""
[ "${MMOFFLOAD:-1}" = "0" ] && MMFLAG="--no-mmproj-offload"
exec "$RUN/llama-liquid-audio-server" \
  -m "$CKPT/$LM" \
  -mm "$CKPT/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf" \
  -mv "$CKPT/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf" \
  --tts-speaker-file "$CKPT/tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf" \
  -ngl ${NGL:-99} --fit off $MMFLAG --host 127.0.0.1 --port 8080 "$@"
