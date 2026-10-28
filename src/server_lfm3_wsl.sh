#!/bin/bash
# Launch llama.cpp's llama-server with LFM3-Audio-2B in WSL2.
# Requires llama.cpp b11960 or newer (the GGUF audio/realtime support line).
# Usage: bash src/server_lfm3_wsl.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/outputs/logs"
LOG_FILE="$ROOT/outputs/logs/lfm3-server-$(date -u +%Y%m%dT%H%M%S)-$$.log"
exec > >(tee -a "$LOG_FILE") 2>&1

CKPT="${LOCALVOICE_MODEL_DIR:-$ROOT/models/LFM3-Audio-2B-GGUF}"
RUN="${LOCALVOICE_LFM3_RUNNER:-$HOME/llama-b11960/build/bin}"
LM="${LM:-LFM3-Audio-2B-Q4_K_M.gguf}"
MM="${MM:-mmproj-LFM3-Audio-2B-Q8_0.gguf}"

for artifact in "$LM" "$MM"; do
  [ -f "$CKPT/$artifact" ] || { echo "Missing model file: $CKPT/$artifact" >&2; exit 1; }
done
[ -x "$RUN/llama-server" ] || { echo "llama-server not found: $RUN/llama-server" >&2; exit 1; }

echo "Turnwave LFM3 session: $(date -u +%FT%TZ) pid=$$ model=$LM"
echo "Model directory: $CKPT; runner: $RUN; log: $LOG_FILE"
MMFLAG=""
[ "${MMOFFLOAD:-1}" = "0" ] && MMFLAG="--no-mmproj-offload"
JINJAFLAG=""
[ "${JINJA:-1}" = "1" ] && JINJAFLAG="--jinja"

exec "$RUN/llama-server" \
  -m "$CKPT/$LM" \
  -mm "$CKPT/$MM" \
  -ngl "${NGL:-99}" \
  -c "${CTX:-8192}" \
  $JINJAFLAG $MMFLAG \
  --host 127.0.0.1 --port "${PORT:-8080}" "$@"
