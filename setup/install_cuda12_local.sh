#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_DIR="${ROOT_DIR}/cuda_toolkits/cuda-12.1"
TMP_DIR="${ROOT_DIR}/tmp"
INSTALLER="${ROOT_DIR}/cuda_12.1.1_530.30.02_linux.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run"

mkdir -p "${CUDA_DIR}" "${TMP_DIR}"

if [ ! -f "${INSTALLER}" ]; then
    wget "${CUDA_URL}" -O "${INSTALLER}"
fi

bash "${INSTALLER}" --silent --tmpdir="${TMP_DIR}" --toolkit --toolkitpath="${CUDA_DIR}"
rm -rf "${TMP_DIR}"

printf '\nCUDA toolkit installed at: %s\n' "${CUDA_DIR}"
printf 'Run the following in your shell before building TrajFlow:\n\n'
printf 'export CUDA_HOME="%s"\n' "${CUDA_DIR}"
printf 'export PATH="%s/bin:$PATH"\n' "${CUDA_DIR}"
printf 'export LD_LIBRARY_PATH="%s/lib64:%s/extras/CUPTI/lib64:$LD_LIBRARY_PATH"\n' "${CUDA_DIR}" "${CUDA_DIR}"
printf 'export TORCH_CUDA_ARCH_LIST="9.0"\n'
