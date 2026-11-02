#!/bin/bash
# Start an independently installed LFM3-compatible llama-server in WSL.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${LOCALVOICE_LFM3_MODEL_DIR:-$ROOT/models/LFM3-Omni-2B-GGUF}"
RUNNER_DIR="${LOCALVOICE_LFM3_RUNNER:-$ROOT/models/runner/llama-b12032-cuda/files/llama-b12032}"
RUNTIME_DIR="${LOCALVOICE_CUDA_LIB:-$ROOT/models/runner/llama-b12032-cuda/runtime/cudart-llama-b12032-bin-ubuntu-cuda-13.4-x64}"
case "${LOCALVOICE_LFM3_VARIANT:-standard}" in
  standard)
    MODEL="$MODEL_DIR/LFM3-Omni-2B-Q4_K_M.gguf"
    PROJECTOR="$MODEL_DIR/mmproj-LFM3-Omni-2B-Q4_K_M.gguf"
    ;;
  research)
    MODEL="${LOCALVOICE_LFM3_RESEARCH_MODEL:?Set LOCALVOICE_LFM3_RESEARCH_MODEL to the research GGUF path}"
    PROJECTOR="${LOCALVOICE_LFM3_RESEARCH_PROJECTOR:?Set LOCALVOICE_LFM3_RESEARCH_PROJECTOR to its matching projector GGUF path}"
    ;;
  *) echo "Unknown LFM3 variant; use standard or research" >&2; exit 1 ;;
esac
for file in "$MODEL" "$PROJECTOR" "$RUNNER_DIR/llama-server"; do
  [ -f "$file" ] || { echo "Missing LFM3 artifact: $file" >&2; exit 1; }
done
export LD_LIBRARY_PATH="$RUNNER_DIR:$RUNTIME_DIR:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$RUNNER_DIR/llama-server" -m "$MODEL" -mm "$PROJECTOR" \
  --reasoning off --temp 0 -ngl "${NGL:-99}" -np "${LOCALVOICE_PARALLEL:-1}" \
  --flash-attn "${LOCALVOICE_FLASH_ATTN:-on}" --cache-type-k "${LOCALVOICE_KV_TYPE:-q8_0}" --cache-type-v "${LOCALVOICE_KV_TYPE:-q8_0}" \
  --ctx-size "${LOCALVOICE_CONTEXT_SIZE:-4096}" \
  --host 127.0.0.1 --port "${LOCALVOICE_LFM3_PORT:-8080}" "$@"
