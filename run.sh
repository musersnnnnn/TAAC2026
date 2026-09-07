#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: v9 raw-dense stats + gated tail residual calibration ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 1 \
    --user_dense_tokens 5 \
    --item_dense_tokens 1 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --dense_weight_decay 0.01 \
    --valid_metric online_proxy_auc \
    --valid_target_hours 7,8,9 \
    --valid_target_weekdays 0 \
    --valid_target_min_samples 1000 \
    --sample_weight_mode online_proxy \
    --rank_loss_weight 0.05 \
    --rank_loss_temperature 1.0 \
    --rank_loss_max_pairs 8192 \
    --tail_residual_scale 0.20 \
    --tail_aux_loss_weight 0.12 \
    --tail_residual_l2 0.0005 \
    --tail_high_score_quantile 0.80 \
    --label_smoothing 0.003 \
    --logit_l2 0.0002 \
    --dense_ema_decay 0.999 \
    --dense_ema_start_step 7248 \
    --patience 3 \
    --num_workers 8 \
    "$@"

# Keeps the stable v9 backbone/features, then adds a bounded residual calibration head:
# - original user_dense values are kept unchanged; the 32 engineered stats are appended;
# - low-cardinality sparse current-time tokens in addition to dense sin/cos time;
# - time-delta item-history match summaries, so match recency does not depend on sequence order;
# - light in-batch pairwise AUC loss + tiny logit penalty to reduce high-confidence tail misranking.
# - gated tail residual starts from exactly zero and is trained with a small extra loss on
#   target-time, long-history, and high-score samples;
# - dense EMA starts after epoch 2 (3624 steps/epoch in the observed logs);
#   sparse embeddings are not averaged because they are deliberately reset each epoch.
# Checkpoint selection uses online_proxy_auc to avoid over-trusting target_auc alone.
