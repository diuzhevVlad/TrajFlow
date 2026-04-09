#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_DIR="${ROOT_DIR}/cuda_toolkits/cuda-12.1"

export CUDA_HOME="${CUDA_DIR}"
export PATH="${CUDA_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_DIR}/lib64:${CUDA_DIR}/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="9.0"

printf 'Using local CUDA toolkit at %s\n' "${CUDA_DIR}"
