#!/usr/bin/env bash
# Pipeline: 05102026_ACT
# Full collect → validate → train → infer pipeline for TurboPi figure-8 ACT.
#
# Requirements implemented:
#   • 128x128 images, 32 parallel envs, 16 episodes per intent (32 total)
#   • 3 laps per episode, 32 data points per episode (uniform subsampled)
#   • ACT chunk size 8, 45 training epochs
#   • FULL DYNAMIC PHYSICS: gravity + collision for both collection and inference
#   • Wheels grounded via START_HEIGHT=0.040 + road collision platform
#   • Physics validation after collection (trajectory overlay PNG)
#   • Expert path waypoint overlay visible in vec scene
#   • Training/validation loss curves saved per epoch
#   • Gaussian blur + color/crop augmentations in training
#   • Inference: 30s, 640x360 video, one per intent (chase + isometric views)
# Expected wall time: ≈ 8–9 min on a single GPU machine

set -euo pipefail

PIPELINE_NAME="05102026_ACT"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/workspace/isaaclab}"
ISAACLAB_PY="${ISAACLAB_ROOT}/isaaclab.sh"

export PYTHONPATH="\
${ISAACLAB_ROOT}/source/isaaclab:\
${ISAACLAB_ROOT}/source/isaaclab_assets:\
${ISAACLAB_ROOT}/source/isaaclab_rl:\
${ISAACLAB_ROOT}/source/isaaclab_tasks:\
${ISAACLAB_ROOT}/source/isaaclab_mimic:\
${REPO_ROOT}/scripts:\
${REPO_ROOT}\
${PYTHONPATH:+:${PYTHONPATH}}"

# ── Configuration ────────────────────────────────────────────────────────────
NUM_ENVS="${NUM_ENVS:-32}"
EPISODES_PER_INTENT="${EPISODES_PER_INTENT:-16}"
LAPS="${LAPS:-3}"
IMAGE_WIDTH="${IMAGE_WIDTH:-128}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-128}"
CHUNK_SIZE="${CHUNK_SIZE:-8}"
FRAMES_PER_EPISODE="${FRAMES_PER_EPISODE:-32}"
EPOCHS="${EPOCHS:-45}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SETTLE_STEPS="${SETTLE_STEPS:-4}"
INFERENCE_DURATION="${INFERENCE_DURATION:-30}"
INFERENCE_VIDEO_WIDTH="${INFERENCE_VIDEO_WIDTH:-640}"
INFERENCE_VIDEO_HEIGHT="${INFERENCE_VIDEO_HEIGHT:-360}"
INFERENCE_SETTLE_STEPS="${INFERENCE_SETTLE_STEPS:-60}"
SEED="${SEED:-42}"

DATA_DIR="${REPO_ROOT}/data/${PIPELINE_NAME}"
RUN_DIR="${REPO_ROOT}/runs/${PIPELINE_NAME}"
INFERENCE_DIR="${REPO_ROOT}/inference_videos/${PIPELINE_NAME}"
LOG_DIR="${REPO_ROOT}/logs/${PIPELINE_NAME}"
SESSION_NAME="${PIPELINE_NAME}_$(date -u +%Y%m%d_%H%M%S)"

mkdir -p "${DATA_DIR}" "${RUN_DIR}" "${INFERENCE_DIR}" "${LOG_DIR}"

log() { echo "[${PIPELINE_NAME}] $*" | tee -a "${LOG_DIR}/pipeline.log"; }
elapsed() { echo "$(($(date +%s) - T0))s"; }

T0=$(date +%s)
log "========================================================"
log "Pipeline ${PIPELINE_NAME} starting"
log "  envs=${NUM_ENVS}  episodes=${EPISODES_PER_INTENT}/intent  laps=${LAPS}"
log "  image=${IMAGE_WIDTH}x${IMAGE_HEIGHT}  chunk=${CHUNK_SIZE}  frames/ep=${FRAMES_PER_EPISODE}"
log "  epochs=${EPOCHS}  physics=dynamic  settle_steps=${SETTLE_STEPS}"
log "  inference=${INFERENCE_DURATION}s @ ${INFERENCE_VIDEO_WIDTH}x${INFERENCE_VIDEO_HEIGHT}  settle=${INFERENCE_SETTLE_STEPS}"
log "  data → ${DATA_DIR}"
log "  runs → ${RUN_DIR}"
log "  inference → ${INFERENCE_DIR}"
log "========================================================"

# ── Step 1: Vectorised data collection (dynamic physics) ─────────────────────
log "Step 1/4: Collecting data (${NUM_ENVS} envs, $((EPISODES_PER_INTENT * 2)) episodes, dynamic physics)…"
COLLECT_LOG="${LOG_DIR}/collect.log"

"${ISAACLAB_PY}" -p "${REPO_ROOT}/scripts/record_turbopi_figure8_act_vec.py" \
    --headless \
    --dynamic_physics \
    --num_envs      "${NUM_ENVS}" \
    --num_episodes  "$((EPISODES_PER_INTENT * 2))" \
    --laps          "${LAPS}" \
    --image_width   "${IMAGE_WIDTH}" \
    --image_height  "${IMAGE_HEIGHT}" \
    --output_dir    "${DATA_DIR}" \
    --session_name  "${SESSION_NAME}" \
    --dataset_name  "${PIPELINE_NAME}" \
    --action_noise_std 0.03 \
    --settle_steps  "${SETTLE_STEPS}" \
    --seed          "${SEED}" \
    2>&1 | tee "${COLLECT_LOG}"

log "Step 1 done ($(elapsed) elapsed). Checking episodes…"
EPISODE_COUNT=$(find "${DATA_DIR}" -name "episode_info.json" 2>/dev/null | wc -l)
log "  Found ${EPISODE_COUNT} episodes"
if [ "${EPISODE_COUNT}" -lt 2 ]; then
    log "ERROR: Not enough episodes collected. Aborting."
    exit 1
fi

# ── Step 1.5: Post-collection physics & trajectory validation ─────────────────
log "Step 1.5/4: Validating collected trajectories (track overlay plot)…"
"${ISAACLAB_PY}" -p "${REPO_ROOT}/scripts/validate_figure8_collection.py" \
    --data_dir "${DATA_DIR}" \
    --out      "${DATA_DIR}/validation_tracks.png" \
    2>&1 | tee -a "${COLLECT_LOG}" || {
    log "WARNING: Trajectory validation reported issues — check ${DATA_DIR}/validation_tracks.png"
}
log "  Validation plot → ${DATA_DIR}/validation_tracks.png"

# ── Step 2: Training ─────────────────────────────────────────────────────────
log "Step 2/4: Training ACT (epochs=${EPOCHS}, chunk=${CHUNK_SIZE}, img=${IMAGE_WIDTH}x${IMAGE_HEIGHT})…"
TRAIN_LOG="${LOG_DIR}/train.log"

"${ISAACLAB_PY}" -p "${REPO_ROOT}/train_turbopi_mountain_act.py" \
    --episodes-dir      "${DATA_DIR}" \
    --run-dir           "${RUN_DIR}" \
    --epochs            "${EPOCHS}" \
    --batch-size        "${BATCH_SIZE}" \
    --chunk-size        "${CHUNK_SIZE}" \
    --image-size        "${IMAGE_WIDTH}" \
    --frames-per-episode "${FRAMES_PER_EPISODE}" \
    --num-workers       0 \
    --seed              "${SEED}" \
    2>&1 | tee "${TRAIN_LOG}"

log "Step 2 done ($(elapsed) elapsed). Locating best checkpoint…"
BEST_CKPT=$(find "${RUN_DIR}" -name "best.pt" 2>/dev/null | sort | tail -1)
if [ -z "${BEST_CKPT}" ]; then
    log "ERROR: No best.pt checkpoint found. Aborting."
    exit 1
fi
log "  Checkpoint: ${BEST_CKPT}"

# ── Step 3: Inference — one video per intent (dynamic physics) ────────────────
log "Step 3/4: Running inference (${INFERENCE_DURATION}s, ${INFERENCE_VIDEO_WIDTH}x${INFERENCE_VIDEO_HEIGHT}, dynamic)…"

for TASK in go_left go_right; do
    INF_LOG="${LOG_DIR}/infer_${TASK}.log"
    log "  Inference: task=${TASK}"
    "${ISAACLAB_PY}" -p "${REPO_ROOT}/scripts/drive_turbopi_mountain_act.py" \
        --headless \
        --checkpoint        "${BEST_CKPT}" \
        --task              "${TASK}" \
        --duration          "${INFERENCE_DURATION}" \
        --control_mode      dynamic \
        --video_output_dir  "${INFERENCE_DIR}" \
        --video_width       "${INFERENCE_VIDEO_WIDTH}" \
        --video_height      "${INFERENCE_VIDEO_HEIGHT}" \
        --video_views       chase,isometric \
        --video_fps         30 \
        --settle_steps      "${INFERENCE_SETTLE_STEPS}" \
        2>&1 | tee "${INF_LOG}"
    log "  Inference ${TASK} done ($(elapsed) elapsed)"
done

# ── Summary ──────────────────────────────────────────────────────────────────
TOTAL=$(($(date +%s) - T0))
log "========================================================"
log "Pipeline ${PIPELINE_NAME} COMPLETE in ${TOTAL}s"
log "  Episodes collected  : ${EPISODE_COUNT}"
log "  Best checkpoint     : ${BEST_CKPT}"
log "  Validation plot     : ${DATA_DIR}/validation_tracks.png"
log "  Loss curve          : $(find "${RUN_DIR}" -name "loss_curve.png" | head -1)"
log "  Inference videos    : ${INFERENCE_DIR}"
ls -lh "${INFERENCE_DIR}"/*.mp4 2>/dev/null | tee -a "${LOG_DIR}/pipeline.log" || true
log "========================================================"
