#!/bin/bash
# Rootless CUDA build of llama-liquid-audio-server (PR #18641) inside WSL2.
# Toolchain via Miniforge (rootless) + conda-forge cuda-nvcc/cudart/cublas.
# (The pip wheels were a dead end: nvidia-cuda-nvcc ships ptxas but NO nvcc.)
# Run INSIDE WSL:  bash src/fixup_cuda_rootless.sh
set -x
export PATH="$HOME/tools/cmake-3.30.5-linux-x86_64/bin:$HOME/miniforge/bin:$PATH"
cd $HOME

# 1. cmake (already fetched on earlier attempt; keep if present)
if [ ! -x $HOME/tools/cmake-3.30.5-linux-x86_64/bin/cmake ]; then
  curl -sL -o cmake.tgz https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-linux-x86_64.tar.gz
  tar xzf cmake.tgz -C $HOME/tools && rm cmake.tgz
fi

# 2. miniforge (rootless conda)
if [ ! -x $HOME/miniforge/bin/conda ]; then
  curl -sL -o miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash miniforge.sh -b -p $HOME/miniforge && rm -f miniforge.sh
fi

# 3. CUDA compiler + dev libs (rootless, user env)
if [ ! -x $HOME/miniforge/envs/cudaenv/bin/nvcc ]; then
  $HOME/miniforge/bin/conda create -y -n cudaenv -c conda-forge \
    cuda-nvcc cuda-cudart-dev libcublas-dev
fi
PREFIX=$HOME/miniforge/envs/cudaenv
$PREFIX/bin/nvcc --version | tail -1
# driver API stub from WSL (conda env has none)
mkdir -p $PREFIX/lib
[ -e /usr/lib/wsl/lib/libcuda.so ] && ln -sf /usr/lib/wsl/lib/libcuda.so* $PREFIX/lib/ 2>/dev/null
ls $PREFIX/lib/libcuda* 2>/dev/null

# 4. llama.cpp @ PR 18641 (already cloned on earlier attempt)
cd $HOME/liquid-build
git fetch origin pull/18641/head:pr-18641 2>/dev/null || true
git checkout -q pr-18641 2>/dev/null || true
git log --oneline -1

# 5. build (CUDA, no curl dependency)
export CUDACXX=$PREFIX/bin/nvcc
export PATH="$PREFIX/bin:$PATH"
rm -rf build
cmake -S . -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT=$PREFIX -DCMAKE_CUDA_COMPILER=$PREFIX/bin/nvcc
cmake --build build -j$(nproc) --target llama-liquid-audio-server llama-liquid-audio-cli
ls -la build/bin/ | grep -i liquid
echo BUILD_DONE
