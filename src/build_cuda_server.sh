#!/bin/bash
# Rootless CUDA build of llama-liquid-audio-server (PR #18641) inside WSL2.
# Run INSIDE WSL:  bash src/build_cuda_server.sh
# No sudo needed. Produces ~/liquid-build/build/bin/llama-liquid-audio-server
set -x
export DEBIAN_FRONTEND=noninteractive
T=$HOME/tools; mkdir -p $T; cd $HOME

# 1. cmake (binary tarball, rootless)
if ! command -v cmake >/dev/null; then
  curl -sL -o cmake.tgz https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-linux-x86_64.tar.gz
  tar xzf cmake.tgz -C $T && rm cmake.tgz
  export PATH="$T/cmake-3.30.5-linux-x86_64/bin:$PATH"
fi

# 2. CUDA toolkit (runfile, user-local, no driver)
if [ ! -x $HOME/cuda-13/bin/nvcc ]; then
  curl -sL -o cuda.run https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_580.65.06_linux.run
  sh cuda.run --silent --toolkit --toolkitpath=$HOME/cuda-13 --no-opengl-libs --no-man-page --no-drm --no-getting-started
  rm -f cuda.run
fi
export PATH="$HOME/cuda-13/bin:$PATH"
export CUDACXX="$HOME/cuda-13/bin/nvcc"
nvcc --version | tail -1

# 3. llama.cpp @ PR 18641
if [ ! -d $HOME/liquid-build ]; then
  git clone --quiet https://github.com/ggml-org/llama.cpp.git $HOME/liquid-build
fi
cd $HOME/liquid-build
git fetch origin pull/18641/head:pr-18641 2>/dev/null || true
git checkout -q pr-18641 2>/dev/null || git checkout -q -b pr-18641 FETCH_HEAD || true
git log --oneline -1

# 4. build (CUDA, no curl dependency)
cmake -S . -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target llama-liquid-audio-server llama-liquid-audio-cli
ls -la build/bin/ | grep -i liquid
echo BUILD_DONE
