#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-trajflow:cu121}"
WAYMO_SCENARIO_DIR="${WAYMO_SCENARIO_DIR:-/data2/datasets/Waymo/waymo_motion_sc}"
SHM_SIZE="${SHM_SIZE:-16g}"
GPU_ARGS="${GPU_ARGS:---gpus all}"

mkdir -p "${ROOT_DIR}/data/waymo" "${ROOT_DIR}/output"

if [ ! -d "${WAYMO_SCENARIO_DIR}" ]; then
    printf 'Waymo scenario directory not found: %s\n' "${WAYMO_SCENARIO_DIR}" >&2
    exit 1
fi

if [ ! -f "${ROOT_DIR}/data/waymo/cluster_64_center_dict.pkl" ]; then
    printf 'Missing %s/data/waymo/cluster_64_center_dict.pkl\n' "${ROOT_DIR}" >&2
    printf 'Download it with:\n' >&2
    printf '  curl -L https://raw.githubusercontent.com/sshaoshuai/MTR/master/data/waymo/cluster_64_center_dict.pkl -o %s/data/waymo/cluster_64_center_dict.pkl\n' "${ROOT_DIR}" >&2
    exit 1
fi

if [ "$#" -eq 0 ]; then
    set -- bash
fi

docker_tty_args=()
if [ -t 0 ] && [ -t 1 ]; then
    docker_tty_args=(-it)
fi

exec docker run --rm "${docker_tty_args[@]}" ${GPU_ARGS} --shm-size="${SHM_SIZE}" \
    -v "${ROOT_DIR}/data/waymo:/workspace/TrajFlow/data/waymo" \
    -v "${WAYMO_SCENARIO_DIR}:/workspace/TrajFlow/data/waymo/scenario:ro" \
    -v "${ROOT_DIR}/output:/workspace/TrajFlow/output" \
    "${IMAGE_NAME}" "$@"
