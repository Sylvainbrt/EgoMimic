#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DEFAULT_EMIMIC_PYTHON="/data/sybeuret/miniconda3/envs/emimic/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$DEFAULT_EMIMIC_PYTHON" ]]; then
    PYTHON_BIN="$DEFAULT_EMIMIC_PYTHON"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "ERROR: No python interpreter found. Set PYTHON_BIN explicitly." >&2
    exit 1
  fi
fi

ROBOT_DATA="${ROBOT_DATA:-/data/sybeuret/remote_converted_lerobot_data/pick_and_place_220526.hdf5}"
HUMAN_MASKED_DATA="${HUMAN_MASKED_DATA:-/data/sybeuret/aria_gen2_data/converted/pick_and_place_280526_masked_left.hdf5}"
HUMAN_UNMASKED_DATA="${HUMAN_UNMASKED_DATA:-$HUMAN_MASKED_DATA}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/data/sybeuret/output_trainings_egomimic/pap_full_sequence}"
RUN_TAG="${RUN_TAG:-pap_full}"

DEBUG_MODE="${DEBUG_MODE:-0}"
NO_WANDB="${NO_WANDB:-0}"
DRY_RUN="${DRY_RUN:-0}"
USE_MIX_SCHEDULE="${USE_MIX_SCHEDULE:-0}"
MIX_SCHEDULE_PROFILE="${MIX_SCHEDULE_PROFILE:-default}"

BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-500}"
NUM_DATA_WORKERS="${NUM_DATA_WORKERS:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
NUM_NODES="${NUM_NODES:-1}"

START_AT="${START_AT:-1}"
END_AT="${END_AT:-3}"

RUN_ROBOT_ONLY="${RUN_ROBOT_ONLY:-1}"
RUN_ROBOT_HUMAN_MASKED="${RUN_ROBOT_HUMAN_MASKED:-1}"
RUN_ROBOT_HUMAN_UNMASKED="${RUN_ROBOT_HUMAN_UNMASKED:-1}"

ROBOT_ONLY_CONFIG="egomimic/configs/act.json"
MASKED_CONFIG="egomimic/configs/egomimic_lerobot_pap.json"
UNMASKED_CONFIG="egomimic/configs/egomimic_lerobot_pap_nomask.json"
PAIRED_RUN_SUFFIX=""

if [[ "$USE_MIX_SCHEDULE" == "1" ]]; then
  MASKED_CONFIG="egomimic/configs/egomimic_lerobot_pap_sched.json"
  UNMASKED_CONFIG="egomimic/configs/egomimic_lerobot_pap_nomask_sched.json"
  PAIRED_RUN_SUFFIX="_mix_sched"
fi

banner() {
  printf '\n============================================================\n'
  printf '%s\n' "$1"
  printf '============================================================\n'
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  local label="$2"
  [[ -f "$path" ]] || die "$label not found: $path"
}

is_enabled() {
  [[ "$1" == "1" ]]
}

in_range() {
  local step="$1"
  [[ "$step" -ge "$START_AT" && "$step" -le "$END_AT" ]]
}

should_run() {
  local step="$1"
  local enabled="$2"
  in_range "$step" && is_enabled "$enabled"
}

run_training() {
  local step="$1"
  local label="$2"
  local config_rel="$3"
  local robot_dataset="$4"
  local human_dataset="$5"
  local output_subdir="$6"
  local description="$7"

  if ! in_range "$step"; then
    return
  fi

  banner "Starting training ${step}: ${label}"

  local -a cmd=(
    "$PYTHON_BIN"
    "egomimic/scripts/pl_train.py"
    "--config" "$config_rel"
    "--dataset" "$robot_dataset"
    "--output_dir" "${OUTPUT_ROOT}/${output_subdir}"
    "--name" "$label"
    "--description" "$description"
    "--batch-size" "$BATCH_SIZE"
    "--num-epochs" "$NUM_EPOCHS"
    "--num-data-workers" "$NUM_DATA_WORKERS"
    "--gpus-per-node" "$GPUS_PER_NODE"
    "--num-nodes" "$NUM_NODES"
  )

  if [[ -n "$human_dataset" ]]; then
    cmd+=("--dataset_2" "$human_dataset")
  fi

  if [[ "$DEBUG_MODE" == "1" ]]; then
    cmd+=("--debug")
  fi

  if [[ "$NO_WANDB" == "1" ]]; then
    cmd+=("--no-wandb")
  fi

  if [[ "$USE_MIX_SCHEDULE" == "1" && "$config_rel" != "$ROBOT_ONLY_CONFIG" ]]; then
    cmd+=("--mix-schedule-profile" "$MIX_SCHEDULE_PROFILE")
  fi

  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1, skipping execution."
    return
  fi

  (
    cd "$REPO_ROOT"
    "${cmd[@]}"
  )
}

banner "EgoMimic sequential launcher"
echo "Repo root: $REPO_ROOT"
echo "Python: $PYTHON_BIN"
echo "Robot data: $ROBOT_DATA"
echo "Human masked data: $HUMAN_MASKED_DATA"
echo "Human unmasked data: $HUMAN_UNMASKED_DATA"
echo "Output root: $OUTPUT_ROOT"
echo "Debug mode: $DEBUG_MODE"
echo "Use mix schedule: $USE_MIX_SCHEDULE"
echo "Mix schedule profile: $MIX_SCHEDULE_PROFILE"
echo "Batch size override: $BATCH_SIZE"
echo "Num epochs override: $NUM_EPOCHS"
echo "Num data workers override: $NUM_DATA_WORKERS"
echo "Run range: $START_AT -> $END_AT"

[[ "$START_AT" =~ ^[0-9]+$ ]] || die "START_AT must be an integer"
[[ "$END_AT" =~ ^[0-9]+$ ]] || die "END_AT must be an integer"
[[ "$START_AT" -le "$END_AT" ]] || die "START_AT must be <= END_AT"

require_file "$REPO_ROOT/$ROBOT_ONLY_CONFIG" "Robot-only config"
require_file "$REPO_ROOT/$MASKED_CONFIG" "Masked paired config"
require_file "$REPO_ROOT/$UNMASKED_CONFIG" "Unmasked paired config"
require_file "$ROBOT_DATA" "Robot HDF5"

if should_run 2 "$RUN_ROBOT_HUMAN_MASKED" || should_run 3 "$RUN_ROBOT_HUMAN_UNMASKED"; then
  require_file "$HUMAN_MASKED_DATA" "Masked human HDF5"
fi

if should_run 3 "$RUN_ROBOT_HUMAN_UNMASKED"; then
  require_file "$HUMAN_UNMASKED_DATA" "Human HDF5 for unmasked run"
fi

if should_run 1 "$RUN_ROBOT_ONLY"; then
  run_training \
    1 \
    "pap_robot_only" \
    "$ROBOT_ONLY_CONFIG" \
    "$ROBOT_DATA" \
    "" \
    "robot_only" \
    "${RUN_TAG}_robot_only"
fi

if should_run 2 "$RUN_ROBOT_HUMAN_MASKED"; then
  run_training \
    2 \
    "pap_robot_human_masked${PAIRED_RUN_SUFFIX}" \
    "$MASKED_CONFIG" \
    "$ROBOT_DATA" \
    "$HUMAN_MASKED_DATA" \
    "robot_human_masked${PAIRED_RUN_SUFFIX}" \
    "${RUN_TAG}_robot_human_masked${PAIRED_RUN_SUFFIX}"
fi

if should_run 3 "$RUN_ROBOT_HUMAN_UNMASKED"; then
  run_training \
    3 \
    "pap_robot_human_unmasked${PAIRED_RUN_SUFFIX}" \
    "$UNMASKED_CONFIG" \
    "$ROBOT_DATA" \
    "$HUMAN_UNMASKED_DATA" \
    "robot_human_unmasked${PAIRED_RUN_SUFFIX}" \
    "${RUN_TAG}_robot_human_unmasked${PAIRED_RUN_SUFFIX}"
fi

banner "Launcher finished"
