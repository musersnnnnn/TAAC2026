#!/usr/bin/env bash
set -euo pipefail

# Smoke test using the official HuggingFace 1k-row sample.
# This is intended for local debugging only.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

DATA_DIR="${1:-${SCRIPT_DIR}/data_sample_1000}"
SMOKE_OUT_DIR="${SCRIPT_DIR}/outputs/smoke_hf_sample"

python3 -u "${SCRIPT_DIR}/tools/prepare_hf_sample.py" --out_dir "${DATA_DIR}"

TRAIN_CKPT_PATH="${TRAIN_CKPT_PATH:-${SMOKE_OUT_DIR}/ckpt}" \
TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-${SMOKE_OUT_DIR}/log}" \
TRAIN_TF_EVENTS_PATH="${TRAIN_TF_EVENTS_PATH:-${SMOKE_OUT_DIR}/events}" \
bash "${SCRIPT_DIR}/run.sh" \
  --data_dir "${DATA_DIR}" \
  --schema_path "${DATA_DIR}/schema.json" \
  --device cpu \
  --num_workers 0 \
  --buffer_batches 1 \
  --batch_size 64 \
  --num_epochs 1 \
  --eval_every_n_steps 50 \
  --valid_target_min_samples 1 \
  --sample_weight_mode none \
  --rank_loss_weight 0 \
  --tail_aux_loss_weight 0 \
  --tail_residual_scale 0 \
  --dense_ema_decay 0 \
  --dense_ema_start_step 0 \
  --d_model 32 \
  --emb_dim 32 \
  --num_heads 4 \
  --rank_mixer_mode ffn_only \
  --seq_max_lens seq_a:64,seq_b:64,seq_c:64,seq_d:64
