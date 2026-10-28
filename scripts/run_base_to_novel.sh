#!/usr/bin/env bash
set -euo pipefail

# VGAP_MODE selects the ADR-0001 variant (text = original BiomedCCPL, none, image).
# RUN_TAG names the output folder; set it when two runs share a mode (e.g. R2 vs R3).
# Extra arguments after GPU_ID are passed to train.py as config overrides, for
# both the base training run and the novel evaluation.
if [[ $# -lt 2 ]]; then
  echo "Usage: [VGAP_MODE=text|none|image] [RUN_TAG=name] $0 DATA_ROOT DATASET [GPU_ID [KEY VALUE ...]]" >&2
  exit 2
fi

DATA_ROOT=$1
DATASET=$2
GPU_ID=${3:-0}
shift $(( $# < 3 ? $# : 3 ))
VGAP_MODE=${VGAP_MODE:-text}
RUN_TAG=${RUN_TAG:-${VGAP_MODE}}
CONFIG="configs/trainers/BiomedCCPL/base_to_novel/${DATASET}.yaml"

if [[ "${DATASET}" == "BUSI" ]]; then
  echo "BUSI is excluded from base-to-novel evaluation (only three classes)." >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Unknown dataset or missing config: ${CONFIG}" >&2
  exit 2
fi

ROOT_OUTPUT="output/base_to_novel/${RUN_TAG}/${DATASET}"
for SEED in 1 2 3; do
  BASE_OUTPUT="${ROOT_OUTPUT}/base/seed${SEED}"
  NOVEL_OUTPUT="${ROOT_OUTPUT}/novel/seed${SEED}"

  CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --root "${DATA_ROOT}" \
    --seed "${SEED}" \
    --output-dir "${BASE_OUTPUT}" \
    --config-file "${CONFIG}" \
    DATASET.NAME "${DATASET}" \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.BIOMEDCCPL.VGAP.MODE "${VGAP_MODE}" \
    ${@+"$@"}

  CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --root "${DATA_ROOT}" \
    --seed "${SEED}" \
    --output-dir "${NOVEL_OUTPUT}" \
    --config-file "${CONFIG}" \
    --model-dir "${BASE_OUTPUT}" \
    --load-epoch 50 \
    --eval-only \
    DATASET.NAME "${DATASET}" \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES new \
    TRAINER.BIOMEDCCPL.VGAP.MODE "${VGAP_MODE}" \
    ${@+"$@"}
done

python tools/parse_test_res.py "${ROOT_OUTPUT}/base"
python tools/parse_test_res.py "${ROOT_OUTPUT}/novel"
