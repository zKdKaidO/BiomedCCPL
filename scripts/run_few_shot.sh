#!/usr/bin/env bash
set -euo pipefail

# VGAP_MODE selects the ADR-0001 variant (text = original BiomedCCPL, none, image).
# RUN_TAG names the output folder; set it when two runs share a mode (e.g. R2 vs R3).
# Extra arguments after GPU_ID are passed to train.py as config overrides.
if [[ $# -lt 3 ]]; then
  echo "Usage: [VGAP_MODE=text|none|image] [RUN_TAG=name] $0 DATA_ROOT DATASET SHOTS [GPU_ID [KEY VALUE ...]]" >&2
  exit 2
fi

DATA_ROOT=$1
DATASET=$2
SHOTS=$3
GPU_ID=${4:-0}
shift $(( $# < 4 ? $# : 4 ))
VGAP_MODE=${VGAP_MODE:-text}
RUN_TAG=${RUN_TAG:-${VGAP_MODE}}

case "${SHOTS}" in
  1|2|4|8|16) ;;
  *) echo "SHOTS must be one of: 1, 2, 4, 8, 16" >&2; exit 2 ;;
esac

CONFIG="configs/trainers/BiomedCCPL/few_shot/${SHOTS}/${DATASET}.yaml"
if [[ ! -f "${CONFIG}" ]]; then
  echo "Unknown dataset or missing config: ${CONFIG}" >&2
  exit 2
fi

ROOT_OUTPUT="output/few_shot/${RUN_TAG}/${DATASET}/${SHOTS}shot"
for SEED in 1 2 3; do
  OUTPUT="${ROOT_OUTPUT}/seed${SEED}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --root "${DATA_ROOT}" \
    --seed "${SEED}" \
    --output-dir "${OUTPUT}" \
    --config-file "${CONFIG}" \
    DATASET.NAME "${DATASET}" \
    DATASET.NUM_SHOTS "${SHOTS}" \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.BIOMEDCCPL.VGAP.MODE "${VGAP_MODE}" \
    ${@+"$@"}
done

python tools/parse_test_res.py "${ROOT_OUTPUT}"
