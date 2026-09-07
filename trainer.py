"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored).

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

import os
import glob
import shutil
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping
from model import ModelInput


def _parse_int_set(value: Any, default: List[int]) -> List[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        if not value.strip():
            return list(default)
        return [int(part.strip()) for part in value.split(',') if part.strip()]
    return [int(v) for v in value]


def _format_auc(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def _safe_binary_auc(labels: np.ndarray, probs: np.ndarray) -> Optional[float]:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, probs))


def _hour_weekday_from_timestamp(timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ts = timestamps.astype(np.int64, copy=False)
    if ts.size > 0 and int(np.nanmax(ts)) > 10**12:
        ts = ts // 1000
    hours = ((ts // 3600 + 8) % 24).astype(np.int64)
    days = (ts + 8 * 3600) // 86400
    weekdays = ((days + 3) % 7).astype(np.int64)
    return hours, weekdays

class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification.

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss.
    Metrics: BinaryAUROC + binary logloss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        logit_l2: float = 0.0,
        rank_loss_weight: float = 0.0,
        rank_loss_temperature: float = 1.0,
        rank_loss_max_pairs: int = 8192,
        sample_weight_mode: str = 'none',
        tail_aux_loss_weight: float = 0.0,
        tail_residual_l2: float = 0.0,
        tail_high_score_quantile: float = 0.80,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        dense_weight_decay: float = 0.01,
        dense_ema_decay: float = 0.0,
        dense_ema_start_step: int = 0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        valid_metric: str = 'target_auc',
        valid_target_hours: Optional[Any] = None,
        valid_target_weekdays: Optional[Any] = None,
        valid_target_min_samples: int = 1000,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        # schema_path is copied alongside every checkpoint so that infer.py can
        # rebuild the exact same feature schema the model was trained with.
        self.schema_path: Optional[str] = schema_path
        # ns_groups_path is optional; copied next to schema.json when provided
        # and points at an existing file. Keeping the JSON inside the ckpt dir
        # makes the checkpoint self-contained for evaluation environments that
        # do not ship ns_groups.json separately.
        self.ns_groups_path: Optional[str] = ns_groups_path

        # Dual optimizer: Adagrad for sparse Embeddings, AdamW for dense params.
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        self.dense_param_names: List[str] = []
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            sparse_ptrs = {p.data_ptr() for p in sparse_params}
            dense_decay_params = []
            dense_no_decay_params = []
            dense_param_names = []
            for name, p in model.named_parameters():
                if not p.requires_grad or p.data_ptr() in sparse_ptrs:
                    continue
                dense_param_names.append(name)
                lname = name.lower()
                if p.ndim <= 1 or name.endswith('.bias') or 'norm' in lname:
                    dense_no_decay_params.append(p)
                else:
                    dense_decay_params.append(p)
            self.dense_param_names = dense_param_names
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_decay_params) + sum(p.numel() for p in dense_no_decay_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(
                f"Dense params: decay={len(dense_decay_params)} tensors, no_decay={len(dense_no_decay_params)} tensors, "
                f"{dense_param_count:,} parameters (AdamW lr={lr}, wd={dense_weight_decay})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
            )
            dense_groups = []
            if dense_decay_params:
                dense_groups.append({'params': dense_decay_params, 'weight_decay': dense_weight_decay})
            if dense_no_decay_params:
                dense_groups.append({'params': dense_no_decay_params, 'weight_decay': 0.0})
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_groups, lr=lr, betas=(0.9, 0.98)
            )
        else:
            self.sparse_optimizer = None
            self.dense_param_names = [
                name for name, p in model.named_parameters() if p.requires_grad
            ]
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=dense_weight_decay
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.label_smoothing: float = float(label_smoothing)
        self.logit_l2: float = float(logit_l2)
        self.rank_loss_weight: float = float(rank_loss_weight)
        self.rank_loss_temperature: float = max(float(rank_loss_temperature), 1.0e-6)
        self.rank_loss_max_pairs: int = int(rank_loss_max_pairs)
        self.sample_weight_mode: str = str(sample_weight_mode or 'none')
        self.tail_aux_loss_weight: float = max(float(tail_aux_loss_weight), 0.0)
        self.tail_residual_l2: float = max(float(tail_residual_l2), 0.0)
        self.tail_high_score_quantile: float = min(max(float(tail_high_score_quantile), 0.0), 0.99)
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.dense_weight_decay: float = dense_weight_decay
        self.dense_ema_decay: float = min(max(float(dense_ema_decay), 0.0), 0.9999)
        self.dense_ema_start_step: int = max(int(dense_ema_start_step), 0)
        self._train_step_count: int = 0
        self._dense_ema_has_updates: bool = False
        self._dense_ema_state: Dict[str, torch.Tensor] = {}
        if self.dense_ema_decay > 0.0 and self.dense_param_names:
            named_params = dict(self.model.named_parameters())
            self._dense_ema_state = {
                name: named_params[name].detach().clone()
                for name in self.dense_param_names
                if name in named_params and named_params[name].requires_grad
            }
            dense_ema_param_count = sum(t.numel() for t in self._dense_ema_state.values())
            logging.info(
                f"Dense EMA enabled: decay={self.dense_ema_decay}, "
                f"start_step={self.dense_ema_start_step}, "
                f"params={len(self._dense_ema_state)} tensors/{dense_ema_param_count:,} values"
            )
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.train_config: Optional[Dict[str, Any]] = train_config
        self.valid_metric: str = valid_metric
        self.valid_target_hours: List[int] = _parse_int_set(valid_target_hours, [7, 8, 9])
        self.valid_target_weekdays: List[int] = _parse_int_set(valid_target_weekdays, [0])
        self.valid_target_min_samples: int = int(valid_target_min_samples)

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"label_smoothing={self.label_smoothing}, logit_l2={self.logit_l2}, "
                     f"rank_loss_weight={self.rank_loss_weight}, "
                     f"rank_loss_temperature={self.rank_loss_temperature}, "
                     f"sample_weight_mode={self.sample_weight_mode}, "
                     f"tail_aux_loss_weight={self.tail_aux_loss_weight}, "
                     f"tail_residual_l2={self.tail_residual_l2}, "
                     f"tail_high_score_quantile={self.tail_high_score_quantile}, "
                     f"dense_ema_decay={self.dense_ema_decay}, "
                     f"dense_ema_start_step={self.dense_ema_start_step}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}, "
                     f"valid_metric={self.valid_metric}, "
                     f"valid_target_hours={self.valid_target_hours}, "
                     f"valid_target_weekdays={self.valid_target_weekdays}, "
                     f"valid_target_min_samples={self.valid_target_min_samples}")

    def _dense_ema_ready(self) -> bool:
        return (
            self.dense_ema_decay > 0.0
            and bool(self._dense_ema_state)
            and self._dense_ema_has_updates
            and self._train_step_count >= self.dense_ema_start_step
        )

    def _update_dense_ema(self) -> None:
        if (
            self.dense_ema_decay <= 0.0
            or not self._dense_ema_state
            or self._train_step_count < self.dense_ema_start_step
        ):
            return

        named_params = dict(self.model.named_parameters())
        with torch.no_grad():
            if not self._dense_ema_has_updates:
                for name, ema in self._dense_ema_state.items():
                    p = named_params.get(name)
                    if p is not None:
                        ema.copy_(p.detach())
                self._dense_ema_has_updates = True
                return

            decay = self.dense_ema_decay
            one_minus_decay = 1.0 - decay
            for name, ema in self._dense_ema_state.items():
                p = named_params.get(name)
                if p is not None:
                    ema.mul_(decay).add_(p.detach(), alpha=one_minus_decay)

    def _swap_in_dense_ema(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self._dense_ema_ready():
            return None

        named_params = dict(self.model.named_parameters())
        backup: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, ema in self._dense_ema_state.items():
                p = named_params.get(name)
                if p is None:
                    continue
                backup[name] = p.detach().clone()
                p.copy_(ema.to(device=p.device, dtype=p.dtype))
        logging.info(
            f"Using dense EMA weights for validation/checkpoint "
            f"(decay={self.dense_ema_decay}, train_step={self._train_step_count})"
        )
        return backup or None

    def _restore_dense_params(self, backup: Optional[Dict[str, torch.Tensor]]) -> None:
        if not backup:
            return
        named_params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, value in backup.items():
                p = named_params.get(name)
                if p is not None:
                    p.copy_(value.to(device=p.device, dtype=p.dtype))

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        """Build a checkpoint sub-directory name such as
        ``global_step2500.layer_2.head_4.hidden_64[.best_model]``.
        """
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}_{self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``.

        Currently persists up to three files, all overwritten on every call:

        - ``schema.json`` (copied from ``self.schema_path``): feature layout
          metadata needed to rebuild the Parquet dataset.
        - ``ns_groups.json`` (copied from ``self.ns_groups_path`` when set
          and the file exists): NS-token grouping used to construct the
          tokenizer. Making a per-ckpt copy lets evaluation environments
          consume the checkpoint without having to ship the original
          project-level ``ns_groups.json``.
        - ``train_config.json`` (serialized from ``self.train_config``):
          full set of training-time hyperparameters. When ``ns_groups.json``
          is copied into ``ckpt_dir``, the ``ns_groups_json`` field is
          rewritten to the bare filename so that ``infer.py`` resolves it
          against ``ckpt_dir`` rather than the original absolute path on
          the training machine.
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        if self.train_config:
            import json
            cfg_to_dump = self.train_config
            if ns_groups_copied:
                # Override the stored path to a filename relative to ckpt_dir;
                # infer.py already falls back to `<ckpt_dir>/<basename>` when
                # the recorded path is not absolute, which keeps the ckpt
                # portable across hosts.
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump['ns_groups_json'] = os.path.basename(
                    self.ns_groups_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
    ) -> str:
        """Save ``model.pt`` plus sidecar files under a ``global_step`` sub-dir.

        Args:
            global_step: current global step used to name the directory.
            is_best: whether this is a new-best checkpoint.
            skip_model_file: if True, skip writing ``model.pt`` (because the
                caller, e.g. EarlyStopping, has already persisted it to the
                same path). Sidecar files are still (re)written.

        Returns:
            The absolute path of the checkpoint directory.
        """
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self) -> None:
        """Delete stale ``*.best_model`` directories so that only the latest
        best checkpoint is kept on disk.
        """
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in ``batch`` to ``self.device`` (``non_blocking=True``,
        to cooperate with ``pin_memory``). Non-tensor values pass through.
        """
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
        val_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a new-best checkpoint atomically.

        Flow (ordered to avoid leaving empty sidecar-only directories on disk):

        1. Decide whether ``val_auc`` is *likely* to beat the current best
           using the same threshold as ``EarlyStopping._is_not_improved``,
           so our pre-cleanup and EarlyStopping's internal save decision
           stay in sync.
        2. If unlikely, short-circuit: do nothing on disk. We must NOT
           touch ``self.early_stopping.checkpoint_path`` or call
           ``_write_sidecar_files`` because the target directory may not
           exist yet (sidecar-only dirs would otherwise be created here,
           producing checkpoints with missing ``model.pt``).
        3. If likely, point ``EarlyStopping`` at the canonical
           ``global_stepN.best_model/model.pt`` path, remove any stale
           ``*.best_model`` dirs, then run ``EarlyStopping`` (which writes
           ``model.pt`` when it actually confirms a new best).
        4. Only after ``EarlyStopping`` has confirmed a new best
           (``best_score != old_best``) do we write the sidecar files into
           the freshly-created directory; this is guarded so that a
           razor-close score that tripped ``is_likely_new_best`` but not
           ``EarlyStopping``'s own gate does not create a stray dir.
        """
        old_best = self.early_stopping.best_score
        metadata: Dict[str, Any] = {
            "best_val_AUC": val_metrics.get('all_auc', val_auc) if val_metrics else val_auc,
            "best_val_logloss": val_logloss,
            "best_val_monitor": val_auc,
            "best_val_metric": self.valid_metric,
        }
        if val_metrics:
            for key, value in val_metrics.items():
                if value is not None and isinstance(value, (int, float, str)):
                    metadata[f"best_val_{key}"] = value
        is_likely_new_best = (
            old_best is None
            or val_auc > old_best + self.early_stopping.delta
        )
        if not is_likely_new_best:
            # No new best anticipated: leave disk untouched. The previous
            # best_model dir (with its model.pt + sidecars) remains valid.
            self.early_stopping(val_auc, self.model, metadata)
            return

        # Point EarlyStopping at the canonical best-model location for this
        # step. Only done on the likely-new-best branch so that a skipped
        # save never leaks the unused path into EarlyStopping state.
        best_dir = os.path.join(
            self.save_dir,
            self._build_step_dir_name(total_step, is_best=True),
        )
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")

        # Remove stale best dirs first so EarlyStopping's write is the only
        # I/O needed when a new best is confirmed.
        self._remove_old_best_dirs()

        self.early_stopping(val_auc, self.model, metadata)

        # Write sidecar files only when EarlyStopping actually confirmed a
        # new best and wrote model.pt. If the score tripped our heuristic
        # but EarlyStopping internally declined to save, skip to avoid
        # creating an empty (sidecar-only) checkpoint directory.
        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path
        ):
            self._save_step_checkpoint(
                total_step, is_best=True, skip_model_file=True)

    def train(self) -> None:
        """Main training loop: iterates over epochs, performs step-level and
        epoch-level validation, triggers EarlyStopping and the periodic sparse
        re-initialization strategy.
        """
        print("Start training (PCVRHyFormer)")
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                              dynamic_ncols=True, disable=True)
            loss_sum = 0.0

            for step, batch in train_pbar:
                loss = self._train_step(batch)
                total_step += 1
                loss_sum += loss

                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, total_step)

                train_pbar.set_postfix({"loss": f"{loss:.4f}"})

                # Step-level validation (only when eval_every_n_steps > 0).
                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    logging.info(f"Evaluating at step {total_step}")
                    ema_backup = self._swap_in_dense_ema()
                    try:
                        val_score, val_logloss, val_metrics = self.evaluate(epoch=epoch)
                        self.model.train()
                        torch.cuda.empty_cache()

                        logging.info(
                            f"Step {total_step} Validation | monitor: {val_score}, "
                            f"AUC/all: {val_metrics.get('all_auc')}, "
                            f"AUC/target: {val_metrics.get('target_auc')}, LogLoss: {val_logloss}"
                        )

                        if self.writer:
                            self.writer.add_scalar('AUC/valid', val_metrics.get('all_auc', val_score), total_step)
                            self.writer.add_scalar('AUC/valid_target', val_metrics.get('target_auc') or 0.0, total_step)
                            self.writer.add_scalar('AUC/valid_monitor', val_score, total_step)
                            self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                        self._handle_validation_result(total_step, val_score, val_logloss, val_metrics)
                    finally:
                        self._restore_dense_params(ema_backup)

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        return

            logging.info(f"Epoch {epoch}, Average Loss: {loss_sum / len(self.train_loader)}")

            ema_backup = self._swap_in_dense_ema()
            try:
                val_score, val_logloss, val_metrics = self.evaluate(epoch=epoch)
                self.model.train()
                torch.cuda.empty_cache()

                logging.info(
                    f"Epoch {epoch} Validation | monitor: {val_score}, "
                    f"AUC/all: {val_metrics.get('all_auc')}, "
                    f"AUC/target: {val_metrics.get('target_auc')}, LogLoss: {val_logloss}"
                )

                if self.writer:
                    self.writer.add_scalar('AUC/valid', val_metrics.get('all_auc', val_score), total_step)
                    self.writer.add_scalar('AUC/valid_target', val_metrics.get('target_auc') or 0.0, total_step)
                    self.writer.add_scalar('AUC/valid_monitor', val_score, total_step)
                    self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                self._handle_validation_result(total_step, val_score, val_logloss, val_metrics)
            finally:
                self._restore_dense_params(ema_backup)

            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            # After the configured epoch, reinitialize high-cardinality sparse
            # params (Embeddings) as a form of cold restart to reduce overfit.
            # Reference: KuaiShou Tech., "MultiEpoch: Reusing Training Data
            # for Click-Through Rate Prediction",
            # https://arxiv.org/pdf/2305.19531
            if (epoch >= self.reinit_sparse_after_epoch
                    and self.sparse_optimizer is not None):
                # Snapshot Adagrad state per parameter via data_ptr, so state
                # of low-cardinality embeddings can be preserved across rebuild.
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                sparse_params = self.model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                # Restore optimizer state for low-cardinality embeddings only.
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ``ModelInput`` NamedTuple from a device_batch dict."""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
        )

    def _batch_time_parts(self, timestamps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ts = timestamps.long()
        if ts.numel() > 0 and int(ts.max().item()) > 10**12:
            ts = ts // 1000
        hours = ((ts // 3600 + 8) % 24).long()
        days = (ts + 8 * 3600) // 86400
        weekdays = ((days + 3) % 7).long()
        return hours, weekdays

    def _compute_sample_weights(self, device_batch: Dict[str, Any]) -> torch.Tensor:
        label = device_batch['label']
        weights = torch.ones_like(label, dtype=torch.float32, device=self.device)
        mode = self.sample_weight_mode
        if mode == 'none':
            return weights

        if mode in ('target_time', 'online_proxy'):
            hours, weekdays = self._batch_time_parts(device_batch['timestamp'])
            hour_mask = torch.zeros_like(weights, dtype=torch.bool)
            weekday_mask = torch.zeros_like(weights, dtype=torch.bool)
            for h in self.valid_target_hours:
                hour_mask |= hours == int(h)
            for d in self.valid_target_weekdays:
                weekday_mask |= weekdays == int(d)
            target_mask = hour_mask & weekday_mask
            weights = weights + 0.20 * hour_mask.float() + 0.35 * target_mask.float()

        if mode in ('long_history', 'online_proxy'):
            long_score = torch.zeros_like(weights)
            seq_domains = device_batch['_seq_domains']
            for domain in seq_domains:
                seq_len = device_batch[f'{domain}_len'].float()
                max_len = max(float(device_batch[domain].shape[2]), 1.0)
                ratio = (seq_len / max_len).clamp(0.0, 1.0)
                long_score = long_score + ratio / max(float(len(seq_domains)), 1.0)
                if domain in ('seq_c', 'seq_d'):
                    weights = weights + 0.12 * (ratio >= 0.75).float()
                if domain == 'seq_d':
                    weights = weights + 0.18 * (ratio >= 0.90).float()
            weights = weights + 0.15 * (long_score >= 0.70).float()

        return weights / weights.mean().clamp_min(1.0e-6)

    def _batch_online_long_mask(self, device_batch: Dict[str, Any]) -> torch.Tensor:
        seq_domains = device_batch['_seq_domains']
        mask = torch.zeros_like(device_batch['label'], dtype=torch.bool, device=self.device)
        ratios: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_len = device_batch[f'{domain}_len'].float()
            max_len = max(float(device_batch[domain].shape[2]), 1.0)
            ratios[domain] = (seq_len / max_len).clamp(0.0, 1.0)

        if 'seq_d' in ratios:
            mask |= ratios['seq_d'] >= 0.90
        if 'seq_c' in ratios and 'seq_d' in ratios:
            mask |= (ratios['seq_c'] >= 0.75) & (ratios['seq_d'] >= 0.75)
        if all(name in ratios for name in ('seq_a', 'seq_b', 'seq_c')):
            mask |= (
                (ratios['seq_a'] >= 0.875)
                & (ratios['seq_b'] >= 0.875)
                & (ratios['seq_c'] >= 0.50)
            )
        return mask

    def _compute_tail_aux_mask(
        self,
        device_batch: Dict[str, Any],
        score_logits: torch.Tensor,
    ) -> torch.Tensor:
        hours, weekdays = self._batch_time_parts(device_batch['timestamp'])
        hour_mask = torch.zeros_like(score_logits, dtype=torch.bool, device=self.device)
        weekday_mask = torch.zeros_like(score_logits, dtype=torch.bool, device=self.device)
        for h in self.valid_target_hours:
            hour_mask |= hours == int(h)
        for d in self.valid_target_weekdays:
            weekday_mask |= weekdays == int(d)
        target_mask = hour_mask & weekday_mask
        long_mask = self._batch_online_long_mask(device_batch)

        scores = torch.sigmoid(score_logits.detach())
        if scores.numel() <= 1:
            high_score_mask = torch.ones_like(target_mask)
        else:
            keep_frac = max(1.0 - self.tail_high_score_quantile, 1.0 / float(scores.numel()))
            k = max(int(scores.numel() * keep_frac), 1)
            threshold = torch.topk(scores, k=k, largest=True).values.min()
            high_score_mask = scores >= threshold

        return target_mask | long_mask | high_score_mask

    def _pairwise_auc_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        pos_mask = labels > 0.5
        neg_mask = ~pos_mask
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return logits.new_zeros(())

        pos_logits = logits[pos_mask]
        neg_logits = logits[neg_mask]
        pos_weights = sample_weights[pos_mask]
        neg_weights = sample_weights[neg_mask]
        n_pairs = int(pos_logits.numel() * neg_logits.numel())
        max_pairs = max(int(self.rank_loss_max_pairs), 1)

        if n_pairs > max_pairs:
            pos_idx = torch.randint(pos_logits.numel(), (max_pairs,), device=logits.device)
            neg_idx = torch.randint(neg_logits.numel(), (max_pairs,), device=logits.device)
            diff = pos_logits[pos_idx] - neg_logits[neg_idx]
            pair_weights = torch.sqrt(pos_weights[pos_idx] * neg_weights[neg_idx])
        else:
            diff = (pos_logits[:, None] - neg_logits[None, :]).reshape(-1)
            pair_weights = torch.sqrt((pos_weights[:, None] * neg_weights[None, :]).reshape(-1))

        pair_loss = F.softplus(-diff / self.rank_loss_temperature)
        return (pair_loss * pair_weights).sum() / pair_weights.sum().clamp_min(1.0e-6)

    def _train_step(self, batch: Dict[str, Any]) -> float:
        """Run a single training step and return the scalar loss value."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        model_input = self._make_model_input(device_batch)
        use_tail_aux = (
            hasattr(self.model, 'forward_with_tail_aux')
            and (self.tail_aux_loss_weight > 0.0 or self.tail_residual_l2 > 0.0)
        )
        base_logits = None
        residual_logits = None
        tail_gate = None
        if use_tail_aux:
            logits, base_logits, residual_logits, tail_gate = self.model.forward_with_tail_aux(model_input)
        else:
            logits = self.model(model_input)  # (B, 1)
        logits = logits.squeeze(-1)  # (B,)
        if base_logits is not None:
            base_logits = base_logits.squeeze(-1)
        if residual_logits is not None:
            residual_logits = residual_logits.squeeze(-1)
        if tail_gate is not None:
            tail_gate = tail_gate.squeeze(-1)
        sample_weights = self._compute_sample_weights(device_batch)

        if self.loss_type == 'focal':
            loss = sigmoid_focal_loss(logits, label, alpha=self.focal_alpha, gamma=self.focal_gamma)
        else:
            if self.label_smoothing > 0.0:
                smooth = min(max(self.label_smoothing, 0.0), 0.49)
                target = label * (1.0 - 2.0 * smooth) + smooth
            else:
                target = label
            per_sample_loss = F.binary_cross_entropy_with_logits(
                logits, target, reduction='none')
            loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1.0e-6)

        if self.rank_loss_weight > 0.0:
            loss = loss + self.rank_loss_weight * self._pairwise_auc_loss(
                logits, label, sample_weights)
        if self.tail_aux_loss_weight > 0.0 and base_logits is not None:
            tail_mask = self._compute_tail_aux_mask(device_batch, base_logits)
            if bool(tail_mask.any()):
                tail_loss = F.binary_cross_entropy_with_logits(
                    logits[tail_mask], label[tail_mask], reduction='none')
                tail_weights = sample_weights[tail_mask]
                loss = loss + self.tail_aux_loss_weight * (
                    tail_loss * tail_weights).sum() / tail_weights.sum().clamp_min(1.0e-6)
        if self.tail_residual_l2 > 0.0 and residual_logits is not None and tail_gate is not None:
            scale = float(getattr(self.model, 'tail_residual_scale', 1.0))
            tail_delta = scale * tail_gate * residual_logits
            loss = loss + self.tail_residual_l2 * (tail_delta.pow(2) * sample_weights).mean()
        if self.logit_l2 > 0.0:
            loss = loss + self.logit_l2 * (logits.pow(2) * sample_weights).mean()
        loss.backward()
        # foreach=False: avoids a PyTorch _foreach_norm CUDA kernel bug observed
        # with certain tensor shapes in this project.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)

        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()

        self._train_step_count += 1
        self._update_dense_ema()

        return loss.item()

    def _subset_metrics(
        self,
        name: str,
        mask: np.ndarray,
        labels_np: np.ndarray,
        probs: np.ndarray,
        metrics: Dict[str, Any],
    ) -> Optional[float]:
        n = int(mask.sum())
        pos = int(labels_np[mask].sum()) if n > 0 else 0
        neg = n - pos
        auc = _safe_binary_auc(labels_np[mask], probs[mask]) if n > 0 else None
        metrics[f'{name}_n'] = n
        metrics[f'{name}_pos'] = pos
        metrics[f'{name}_neg'] = neg
        metrics[f'{name}_auc'] = auc
        return auc

    def _online_long_mask(
        self,
        seq_lens_np: np.ndarray,
        seq_domains: List[str],
    ) -> np.ndarray:
        if seq_lens_np.size == 0:
            return np.zeros(0, dtype=bool)
        cols = {domain: i for i, domain in enumerate(seq_domains)}

        def _col(name: str) -> np.ndarray:
            if name not in cols:
                return np.zeros(seq_lens_np.shape[0], dtype=np.int64)
            return seq_lens_np[:, cols[name]]

        seq_a = _col('seq_a')
        seq_b = _col('seq_b')
        seq_c = _col('seq_c')
        seq_d = _col('seq_d')
        return (
            (seq_d >= 480)
            | ((seq_c >= 384) & (seq_d >= 384))
            | ((seq_a >= 224) & (seq_b >= 224) & (seq_c >= 256))
        )

    def _add_online_proxy_metric(self, metrics: Dict[str, Any]) -> None:
        components = [
            ('target_auc', 0.40),
            ('online_long_auc', 0.30),
            ('all_auc', 0.20),
            ('hour_auc', 0.10),
        ]
        total_weight = 0.0
        score = 0.0
        desc = []
        for key, weight in components:
            value = metrics.get(key)
            if value is None:
                continue
            score += float(value) * weight
            total_weight += weight
            desc.append(f"{key}:{float(value):.6f}*{weight:.2f}")
        metrics['online_proxy_auc'] = score / total_weight if total_weight > 0 else metrics.get('all_auc', 0.0)
        metrics['online_proxy_n'] = metrics.get('all_n', 0)
        if desc:
            logging.info(f"online_proxy_auc components: {','.join(desc)}")

    def _select_monitor_score(self, metrics: Dict[str, Any]) -> float:
        all_auc = metrics.get('all_auc')
        if all_auc is None:
            all_auc = 0.0
        metric_name = self.valid_metric
        candidate = metrics.get(metric_name)
        n_key = metric_name[:-4] + '_n' if metric_name.endswith('_auc') else ''
        n = int(metrics.get(n_key, 0)) if n_key else 0
        if candidate is None or n < self.valid_target_min_samples:
            logging.info(
                f"Validation monitor fallback: {metric_name} unavailable or too small "
                f"(n={n}); using all_auc={all_auc:.6f}"
            )
            metrics['monitor_metric'] = 'all_auc'
            metrics['monitor_score'] = float(all_auc)
            return float(all_auc)
        metrics['monitor_metric'] = metric_name
        metrics['monitor_score'] = float(candidate)
        return float(candidate)

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float, Dict[str, Any]]:
        """Run validation and return (monitor_score, logloss, metric_dict)."""
        self.model.eval()
        if not epoch:
            epoch = -1

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), disable=True)

        all_logits_list = []
        all_labels_list = []
        all_timestamps_list = []
        all_seq_lens_list = []
        seq_domain_order: Optional[List[str]] = None

        with torch.no_grad():
            for step, batch in pbar:
                logits, labels, timestamps, seq_lens, seq_domains = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())
                all_timestamps_list.append(timestamps.detach().cpu())
                all_seq_lens_list.append(seq_lens.detach().cpu())
                if seq_domain_order is None:
                    seq_domain_order = list(seq_domains)

        all_logits = torch.cat(all_logits_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0).long()
        all_timestamps = torch.cat(all_timestamps_list, dim=0).long()
        all_seq_lens = torch.cat(all_seq_lens_list, dim=0).long()

        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()
        timestamps_np = all_timestamps.numpy()
        seq_lens_np = all_seq_lens.numpy()

        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN, filtering them out")
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]
            timestamps_np = timestamps_np[valid_mask]
            seq_lens_np = seq_lens_np[valid_mask]
            all_logits = all_logits[torch.from_numpy(valid_mask)]
            all_labels = all_labels[torch.from_numpy(valid_mask)]

        metrics: Dict[str, Any] = {}
        all_mask = np.ones(len(labels_np), dtype=bool)
        all_auc = self._subset_metrics('all', all_mask, labels_np, probs, metrics)
        if all_auc is None:
            all_auc = 0.0
            metrics['all_auc'] = all_auc

        hours, weekdays = _hour_weekday_from_timestamp(timestamps_np)
        hour_mask = np.isin(hours, self.valid_target_hours)
        weekday_mask = np.isin(weekdays, self.valid_target_weekdays)
        target_mask = hour_mask & weekday_mask

        self._subset_metrics('hour', hour_mask, labels_np, probs, metrics)
        self._subset_metrics('weekday', weekday_mask, labels_np, probs, metrics)
        self._subset_metrics('target', target_mask, labels_np, probs, metrics)
        online_long_mask = self._online_long_mask(seq_lens_np, seq_domain_order or [])
        self._subset_metrics('online_long', online_long_mask, labels_np, probs, metrics)
        metrics['target_share'] = float(target_mask.mean()) if len(target_mask) else 0.0
        metrics['target_hours'] = ','.join(str(x) for x in self.valid_target_hours)
        metrics['target_weekdays'] = ','.join(str(x) for x in self.valid_target_weekdays)
        metrics['online_long_share'] = (
            float(online_long_mask.mean()) if len(online_long_mask) else 0.0)
        self._add_online_proxy_metric(metrics)

        if len(all_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(all_logits, all_labels.float()).item()
        else:
            logloss = float('inf')

        monitor_score = self._select_monitor_score(metrics)
        logging.info(
            "Validation slices | "
            f"all_auc={_format_auc(metrics.get('all_auc'))}, n={metrics.get('all_n', 0)}, "
            f"target_auc={_format_auc(metrics.get('target_auc'))}, "
            f"target_n={metrics.get('target_n', 0)}, target_share={metrics.get('target_share', 0.0):.4f}, "
            f"online_long_auc={_format_auc(metrics.get('online_long_auc'))}, "
            f"online_long_n={metrics.get('online_long_n', 0)}, "
            f"hour_auc={_format_auc(metrics.get('hour_auc'))}, hour_n={metrics.get('hour_n', 0)}, "
            f"weekday_auc={_format_auc(metrics.get('weekday_auc'))}, weekday_n={metrics.get('weekday_n', 0)}, "
            f"monitor={metrics.get('monitor_metric')}:{monitor_score:.6f}"
        )

        return monitor_score, logloss, metrics

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """Run a single validation step and return logits plus probe fields."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']
        timestamps = device_batch['timestamp']
        seq_domains = list(device_batch['_seq_domains'])
        seq_lens = torch.stack(
            [device_batch[f'{domain}_len'] for domain in seq_domains], dim=1)

        model_input = self._make_model_input(device_batch)
        logits, _ = self.model.predict(model_input)  # (B, 1), (B, D)
        logits = logits.squeeze(-1)  # (B,)

        return logits, label, timestamps, seq_lens, seq_domains
