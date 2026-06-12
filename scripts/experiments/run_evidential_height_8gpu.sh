#!/usr/bin/env bash
# =============================================================================
# Launch 8 parallel single-GPU trainings: evidential U-Net, mean_height only.
#
# One job per physical GPU (0–7), each with its own seed and run directory.
# Jobs start simultaneously (background) and the script waits for all to finish.
#
# GPU memory (auto batch size):
#   P100 16GB  (GPUs 0,1,2,3,6) → batch_size=16
#   V100/V100S 32GB (GPUs 4,5,7) → batch_size=32
#
# Usage:
#   bash scripts/experiments/run_evidential_height_8gpu.sh
#
# Optional env overrides:
#   SEEDS="42 123 456 80 100 200 300 400"
#   GPUS="0 1 2 3 4 5 6 7"
#   IMAGE=evidential:latest
#   REPO_ROOT=/data/ammar/evidential
#   DATA_ZARR=/data/ammar/4g.zarr
#   NUM_WORKERS=4
#   SHM_SIZE=8g
#   DRY_RUN=1          # print commands only
# =============================================================================
set -euo pipefail

IMAGE="${IMAGE:-evidential:latest}"
REPO_ROOT="${REPO_ROOT:-/data/ammar/evidential}"
DATA_ZARR="${DATA_ZARR:-/data/ammar/4g.zarr}"
ARTIFACTS_HOST="${ARTIFACTS_HOST:-/data/ammar/evidential/artifacts}"
CONFIG="${CONFIG:-configs/experiments/singletask_height_unet_evidential.yaml}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SHM_SIZE="${SHM_SIZE:-8g}"
DRY_RUN="${DRY_RUN:-0}"

# 8 seeds (override with space-separated SEEDS env)
if [[ -n "${SEEDS:-}" ]]; then
    read -r -a SEED_ARR <<< "${SEEDS}"
else
    SEED_ARR=(42 80 100 123 456 789 1337 2024)
fi

if [[ -n "${GPUS:-}" ]]; then
    read -r -a GPU_ARR <<< "${GPUS}"
else
    GPU_ARR=(0 1 2 3 4 5 6 7)
fi

if [[ ${#SEED_ARR[@]} -ne ${#GPU_ARR[@]} ]]; then
    echo "ERROR: need same number of seeds (${#SEED_ARR[@]}) and GPUs (${#GPU_ARR[@]})." >&2
    exit 1
fi

N_JOBS=${#SEED_ARR[@]}
RUN_ROOT="${RUN_ROOT:-/workspace/artifacts/experiments/singletask_height_unet_evidential}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/evidential_height_8gpu}"

mkdir -p "${LOG_ROOT}"

# Query GPU names once (host nvidia-smi)
batch_for_gpu() {
    local gpu_id="$1"
    local name
    name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${gpu_id}" 2>/dev/null | head -1 || true)"
    if [[ "${name}" == *"V100"* ]]; then
        echo 32
    else
        echo 16
    fi
}

SEP="──────────────────────────────────────────────────────────────"
echo "${SEP}"
echo "  Evidential height — ${N_JOBS} parallel jobs (1 GPU each)"
echo "  CONFIG=${CONFIG}"
echo "  IMAGE=${IMAGE}"
echo "  SEEDS: ${SEED_ARR[*]}"
echo "  GPUS:  ${GPU_ARR[*]}"
echo "  LOG_ROOT (host): ${LOG_ROOT}"
echo "${SEP}"

PIDS=()
TAGS=()

for i in "${!SEED_ARR[@]}"; do
    SEED="${SEED_ARR[$i]}"
    GPU_HOST="${GPU_ARR[$i]}"
    BATCH="$(batch_for_gpu "${GPU_HOST}")"
    RUN_DIR="${RUN_ROOT}/seed_${SEED}"
    LOG_FILE="${LOG_ROOT}/seed_${SEED}_gpu${GPU_HOST}.log"
    TAG="seed_${SEED}_gpu${GPU_HOST}"

    CMD=(
        docker run --rm
        --gpus '"device='"${GPU_HOST}"'"'
        --shm-size "${SHM_SIZE}"
        -v "${REPO_ROOT}:/workspace"
        -v "${DATA_ZARR}:/data/4g.zarr:ro"
        -v "${ARTIFACTS_HOST}:/workspace/artifacts"
        -w /workspace
        -e "CUDA_VISIBLE_DEVICES=0"
        "${IMAGE}"
        python scripts/train.py
        --config "${CONFIG}"
        --seed "${SEED}"
        --run-dir "${RUN_DIR}"
        --batch-size "${BATCH}"
        --num-workers "${NUM_WORKERS}"
    )

    echo "[${TAG}] GPU ${GPU_HOST}  batch=${BATCH}  run_dir=${RUN_DIR}"
    echo "  log → ${LOG_FILE}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "  DRY_RUN: ${CMD[*]}"
        continue
    fi

    (
        echo "=== start $(date -Is)  seed=${SEED}  gpu=${GPU_HOST}  batch=${BATCH} ==="
        "${CMD[@]}"
        _ec=$?
        echo "=== end $(date -Is)  exit=${_ec} ==="
        exit "${_ec}"
    ) > "${LOG_FILE}" 2>&1 &

    PIDS+=($!)
    TAGS+=("${TAG}")
done

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN complete — no jobs started."
    exit 0
fi

echo ""
echo "All ${N_JOBS} jobs launched. Waiting..."
FAIL=0
for j in "${!PIDS[@]}"; do
    pid="${PIDS[$j]}"
    tag="${TAGS[$j]}"
    if wait "${pid}"; then
        echo "  OK   ${tag}  (pid ${pid})"
    else
        echo "  FAIL ${tag}  (pid ${pid}) — see ${LOG_ROOT}/${tag//_/-}.log" >&2
        FAIL=1
    fi
done

# Fix log path in fail message — logs are seed_${SEED}_gpu${GPU}.log
echo "${SEP}"
if [[ "${FAIL}" -eq 0 ]]; then
    echo "  All ${N_JOBS} trainings finished successfully."
    echo "  Checkpoints under: ${ARTIFACTS_HOST}/experiments/singletask_height_unet_evidential/seed_*/checkpoints/"
else
    echo "  One or more jobs failed. Inspect logs in: ${LOG_ROOT}"
    exit 1
fi
echo "${SEP}"
