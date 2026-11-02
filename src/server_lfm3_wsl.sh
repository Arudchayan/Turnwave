#!/bin/bash
# Start an independently installed LFM3-compatible llama-server in WSL.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${LOCALVOICE_LFM3_MODEL_DIR:-$ROOT/models/LFM3-Omni-2B-GGUF}"
RUNNER_DIR="${LOCALVOICE_LFM3_RUNNER:-$ROOT/models/runner/llama-b12032-cuda/files/llama-b12032}"
MODEL="$MODEL_DIR/LFM3-Omni-2B-Q4_K_M.gguf"
PROJECTOR="$MODEL_DIR/mmproj-LFM3-Omni-2B-Q4_K_M.gguf"
for file in "$MODEL" "$PROJECTOR" "$RUNNER_DIR/llama-server"; do
  [ -f "$file" ] || { echo "Missing LFM3 artifact: $file" >&2; exit 1; }
done
export LD_LIBRARY_PATH="$RUNNER_DIR${LOCALVOICE_CUDA_LIB:+:$LOCALVOICE_CUDA_LIB}:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$RUNNER_DIR/llama-server" -m "$MODEL" -mm "$PROJECTOR" \
  --reasoning off --temp 0 -ngl "${NGL:-99}" --ctx-size "${LOCALVOICE_CONTEXT_SIZE:-4096}" \
  --host 127.0.0.1 --port "${LOCALVOICE_LFM3_PORT:-8080}" "$@"
