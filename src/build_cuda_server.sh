#!/usr/bin/env bash
# Build the pinned Liquid audio runner from an existing local source checkout.
# This script never clones, fetches, checks out, resets, cleans, installs, or downloads.
# Run from the Turnwave repository root in WSL:
#   bash src/build_cuda_server.sh
set -euo pipefail

readonly EXPECTED_REVISION="ec9d1fdd9cc18643c5e161b65b8053f6f6f34a4b"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly SOURCE="${LOCALVOICE_RUNNER_SOURCE:-${HOME}/liquid-build}"
readonly BUILD_DIR="${LOCALVOICE_RUNNER_BUILD_DIR:-${SOURCE}/build}"
readonly PATCHES=(
  "${REPO_ROOT}/patches/liquid-turn-reset.patch"
  "${REPO_ROOT}/patches/liquid-attachment-reset.patch"
  "${REPO_ROOT}/patches/liquid-token-lifetime.patch"
  "${REPO_ROOT}/patches/liquid-token-limit.patch"
  "${REPO_ROOT}/patches/liquid-abort-lifecycle.patch"
)

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[[ -d "${SOURCE}" ]] || fail "runner source checkout not found: ${SOURCE}. Prepare an existing local llama.cpp PR #18641 checkout at ${EXPECTED_REVISION}; this helper does not clone or fetch. Set LOCALVOICE_RUNNER_SOURCE to its path."
command -v git >/dev/null 2>&1 || fail "git is required to verify the existing runner checkout."

SOURCE_TOP="$(git -C "${SOURCE}" rev-parse --show-toplevel 2>/dev/null)" \
  || fail "${SOURCE} is not an existing Git checkout. No source was changed."
SOURCE_REAL="$(cd -- "${SOURCE}" && pwd -P)"
SOURCE_TOP_REAL="$(cd -- "${SOURCE_TOP}" && pwd -P)"
[[ "${SOURCE_REAL}" == "${SOURCE_TOP_REAL}" ]] \
  || fail "LOCALVOICE_RUNNER_SOURCE must name the checkout root, not a subdirectory."

ACTUAL_REVISION="$(git -C "${SOURCE}" rev-parse --verify 'HEAD^{commit}' 2>/dev/null)" \
  || fail "could not resolve HEAD in ${SOURCE}; no source was changed."
[[ "${ACTUAL_REVISION}" == "${EXPECTED_REVISION}" ]] \
  || fail "runner checkout is ${ACTUAL_REVISION}; expected exact commit ${EXPECTED_REVISION}. The helper will not fetch, switch, or reset an existing checkout."

# Accept either the clean upstream checkout or the full reviewed Turnwave
# patch set applied by src/apply_runner_fixes.sh. Refuse partial/unknown patch
# states. All checks are read-only; the user's worktree is never reset or cleaned.
PATCHES_APPLIED=0
PATCHES_UNAPPLIED=0
for patch in "${PATCHES[@]}"; do
  [[ -f "${patch}" ]] || fail "required reviewed patch is missing: ${patch}"
  if git -C "${SOURCE}" apply --reverse --check "${patch}" >/dev/null 2>&1; then
    PATCHES_APPLIED=$((PATCHES_APPLIED + 1))
  elif git -C "${SOURCE}" apply --check "${patch}" >/dev/null 2>&1; then
    PATCHES_UNAPPLIED=$((PATCHES_UNAPPLIED + 1))
  else
    fail "patch state is not cleanly applied or applicable: ${patch}. Inspect the existing checkout; it was left untouched."
  fi
done
if (( PATCHES_APPLIED != 0 && PATCHES_UNAPPLIED != 0 )); then
  fail "only ${PATCHES_APPLIED}/${#PATCHES[@]} reviewed patches are applied. Apply or inspect the complete patch set with src/apply_runner_fixes.sh; the checkout was left untouched."
fi

if [[ -n "$(git -C "${SOURCE}" status --porcelain --untracked-files=normal)" ]]; then
  printf '%s\n' "NOTICE: the pinned checkout has existing local changes. They will be preserved; this helper performs no fetch, checkout, reset, clean, or patch application." >&2
fi

command -v cmake >/dev/null 2>&1 \
  || fail "CMake is required but was not found. Install/activate the required CMake toolchain yourself; this helper does not download or install tools."
command -v nvcc >/dev/null 2>&1 \
  || fail "CUDA Toolkit nvcc is required but was not found. Install/activate a compatible CUDA 13 toolkit yourself; this helper does not download or install it."
if ! command -v c++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
  fail "A host C++ compiler (c++, g++, or clang++) is required. Install/activate the compiler toolchain yourself."
fi

if [[ -n "${CUDACXX:-}" ]]; then
  CUDA_COMPILER="$(command -v "${CUDACXX}" 2>/dev/null)" \
    || fail "CUDACXX does not resolve to an executable: ${CUDACXX}"
else
  CUDA_COMPILER="$(command -v nvcc)"
fi
[[ -x "${CUDA_COMPILER}" ]] || fail "CUDA compiler is not executable: ${CUDA_COMPILER}"
[[ "${BUILD_DIR}" != "${SOURCE_TOP}" ]] \
  || fail "build directory must not be the source checkout root. Set LOCALVOICE_RUNNER_BUILD_DIR to a separate directory."

JOBS="${LOCALVOICE_BUILD_JOBS:-4}"
[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || fail "LOCALVOICE_BUILD_JOBS must be a positive integer."

if (( PATCHES_APPLIED == ${#PATCHES[@]} )); then
  printf 'Using pinned runner %s with all %s reviewed Turnwave patches already applied.\n' \
    "${EXPECTED_REVISION}" "${#PATCHES[@]}"
else
  printf 'Using pinned runner %s without Turnwave patches.\n' "${EXPECTED_REVISION}"
fi
printf 'Toolchain: %s; CUDA compiler: %s; build directory: %s\n' \
  "$(cmake --version | sed -n '1p')" "${CUDA_COMPILER}" "${BUILD_DIR}"

cmake -S "${SOURCE}" -B "${BUILD_DIR}" \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="${CUDA_COMPILER}"
cmake --build "${BUILD_DIR}" -j"${JOBS}" \
  --target llama-liquid-audio-server llama-liquid-audio-cli

[[ -x "${BUILD_DIR}/bin/llama-liquid-audio-server" ]] \
  || fail "build completed without the expected executable: ${BUILD_DIR}/bin/llama-liquid-audio-server"
[[ -x "${BUILD_DIR}/bin/llama-liquid-audio-cli" ]] \
  || fail "build completed without the expected executable: ${BUILD_DIR}/bin/llama-liquid-audio-cli"
printf 'BUILD_DONE revision=%s patches_applied=%s build_dir=%s\n' \
  "${EXPECTED_REVISION}" "${PATCHES_APPLIED}" "${BUILD_DIR}"
