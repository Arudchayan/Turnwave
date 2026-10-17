#!/bin/bash
# Configure (CUDA 13 cccl-in-subdir fix) + detached incremental build.
# CUDA 13 moved CCCL under include/cccl/, the May-2026 PR branch predates it:
# add -I.../cccl so <cuda/iterator> and <cub/cub.cuh> resolve. PLUS --include
# cuda/iterator: <cub/cub.cuh> pulls cuda/std headers defining namespace cuda
# WITHOUT the iterator members, and nothing else includes <cuda/iterator>
# (proven by minimal nvcc probes: cub-first breaks, iterator-first works).
ENV="$HOME/miniforge/envs/cudaenv"
CCCL=$ENV/targets/x86_64-linux/include/cccl
export PATH="$HOME/tools/cmake-3.30.5-linux-x86_64/bin:$ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDACXX=$ENV/bin/nvcc
cd "$HOME/liquid-build" || exit 1
cmake -S . -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT=$ENV -DCMAKE_CUDA_COMPILER=$ENV/bin/nvcc \
  -DCMAKE_CUDA_FLAGS="-I$CCCL -include cuda/iterator" -DCMAKE_CXX_FLAGS="-I$CCCL"
if [ "${FOREGROUND}" = "1" ]; then
  cmake --build build -j12 --target llama-liquid-audio-server llama-liquid-audio-cli
else
  setsid cmake --build build -j12 --target llama-liquid-audio-server llama-liquid-audio-cli > "$HOME/build2.log" 2>&1 < /dev/null &
  echo "build detached, log ~/build2.log"
fi
