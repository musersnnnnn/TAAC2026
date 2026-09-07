"""PCVRHyFormer inference script (uploaded by the contestant into the
evaluation container).

Model construction mirrors ``train.py``: we rebuild the model from
``schema.json`` + ``ns_groups.json`` + ``train_config.json``. All model
hyperparameters are resolved first from the ckpt directory's
``train_config.json`` (written by ``trainer.py`` when saving a checkpoint),
falling back to ``_FALLBACK_MODEL_CFG`` below (which must stay consistent
with the CLI defaults in ``train.py``).

Only the Parquet data format is supported.

Environment variables:
    MODEL_OUTPUT_PATH  Checkpoint directory (points at the ``global_step``
                       sub-directory containing ``model.pt`` / ``train_config.json``).
    EVAL_DATA_PATH     Test data directory (*.parquet + schema.json).
    EVAL_RESULT_PATH   Directory for the generated ``predictions.json``.
"""

import os
import json
import logging
import math
import hashlib
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FeatureSchema, PCVRParquetDataset, NUM_TIME_BUCKETS
from model import PCVRHyFormer, ModelInput


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


# Fallback values used only when ``train_config.json`` is missing from the
# ckpt directory.
#
# These MUST match the argparse defaults in ``train.py``; otherwise once the
# fallback path is actually taken the built model will shape-mismatch the
# saved state_dict.
#
# Special note on ``num_time_buckets``: this value is strictly determined by
# ``dataset.BUCKET_BOUNDARIES`` and is NOT an independent hyperparameter.
# When the feature is enabled we therefore use the constant exposed by the
# dataset module; ``0`` means disabled.
_FALLBACK_MODEL_CFG = {
    'd_model': 64,
    'emb_dim': 64,
    'num_queries': 1,
    'num_hyformer_blocks': 2,
    'num_heads': 4,
    'seq_encoder_type': 'transformer',
    'hidden_mult': 4,
    'dropout_rate': 0.01,
    'seq_top_k': 50,
    'seq_causal': False,
    'action_num': 1,
    'num_time_buckets': NUM_TIME_BUCKETS,
    'rank_mixer_mode': 'full',
    'use_rope': False,
    'rope_base': 10000.0,
    'emb_skip_threshold': 0,
    'seq_id_threshold': 10000,
    'ns_tokenizer_type': 'rankmixer',
    'user_ns_tokens': 0,
    'item_ns_tokens': 0,
    'user_dense_tokens': 1,
    'item_dense_tokens': 1,
    'tail_residual_scale': 0.0,
}

_FALLBACK_SEQ_MAX_LENS = 'seq_a:256,seq_b:256,seq_c:512,seq_d:512'
_FALLBACK_BATCH_SIZE = 256
_FALLBACK_NUM_WORKERS = 16


# Hyperparameter keys used to build the model. Everything else in
# ``train_config.json`` is ignored when constructing ``PCVRHyFormer``.
_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build ``feature_specs = [(vocab_size, offset, length), ...]`` in the
    order of ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def _parse_seq_max_lens(sml_str: str) -> Dict[str, int]:
    """Parse a string like ``'seq_a:256,seq_b:256,...'`` into a dict."""
    seq_max_lens: Dict[str, int] = {}
    for pair in sml_str.split(','):
        k, v = pair.split(':')
        seq_max_lens[k.strip()] = int(v.strip())
    return seq_max_lens


def load_train_config(model_dir: str) -> Dict[str, Any]:
    """Load ``train_config.json`` from the ckpt directory.

    Returns an empty dict (which triggers fallback resolution) if the file is
    not present.
    """
    train_config_path = os.path.join(model_dir, 'train_config.json')
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            cfg = json.load(f)
        logging.info(f"Loaded train_config from {train_config_path}")
        return cfg
    logging.warning(
        f"train_config.json not found in {model_dir}, "
        f"falling back to hardcoded defaults. "
        f"Shape mismatch may occur if training used non-default hyperparameters.")
    return {}


def resolve_model_cfg(train_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model hyperparameters from ``train_config``; missing keys fall
    back to ``_FALLBACK_MODEL_CFG``.

    Special handling for ``num_time_buckets``: it is not exposed on the CLI
    as an independent hyperparameter; the bucket count is uniquely determined
    by the length of ``dataset.BUCKET_BOUNDARIES``. Resolution order:

      1) ``train_config`` contains ``num_time_buckets`` directly (legacy ckpt)
         -> use that value;
      2) ``train_config`` contains ``use_time_buckets`` (new-style training)
         -> derive as ``NUM_TIME_BUCKETS`` or ``0``;
      3) neither is present -> fall back to ``_FALLBACK_MODEL_CFG[...]``.
    """
    cfg: Dict[str, Any] = {}
    for key in _MODEL_CFG_KEYS:
        if key == 'num_time_buckets':
            if 'num_time_buckets' in train_config:
                cfg[key] = train_config['num_time_buckets']
            elif 'use_time_buckets' in train_config:
                cfg[key] = NUM_TIME_BUCKETS if train_config['use_time_buckets'] else 0
            else:
                cfg[key] = _FALLBACK_MODEL_CFG[key]
                logging.warning(
                    f"train_config missing both 'num_time_buckets' and 'use_time_buckets', "
                    f"using fallback = {cfg[key]}")
            continue

        if key in train_config:
            cfg[key] = train_config[key]
        else:
            cfg[key] = _FALLBACK_MODEL_CFG[key]
            logging.warning(
                f"train_config missing '{key}', using fallback = {cfg[key]}")
    return cfg


def build_model(
    dataset: PCVRParquetDataset,
    model_cfg: Dict[str, Any],
    ns_groups_json: Optional[str] = None,
    device: str = 'cpu',
) -> PCVRHyFormer:
    """Construct a ``PCVRHyFormer`` from the dataset schema, an NS-groups JSON,
    and a resolved ``model_cfg`` dict.

    Args:
        dataset: a ``PCVRParquetDataset`` providing the feature schema.
        model_cfg: resolved model hyperparameters, typically the output of
            ``resolve_model_cfg``.
        ns_groups_json: path to the NS-groups JSON file, or ``None`` / empty
            string to disable it (each feature becomes its own singleton group).
        device: torch device.
    """
    # NS grouping. The JSON schema uses *fid* (feature id) values; convert
    # them to positional indices into ``user_int_schema.entries`` /
    # ``item_int_schema.entries`` so ``GroupNSTokenizer`` /
    # ``RankMixerNSTokenizer`` can index ``feature_specs`` directly. This is
    # the same conversion ``train.py`` performs when loading the JSON; doing
    # it here keeps infer.py symmetric with training.
    user_ns_groups: List[List[int]]
    item_ns_groups: List[List[int]]
    if ns_groups_json and os.path.exists(ns_groups_json):
        logging.info(f"Loading NS groups from {ns_groups_json}")
        with open(ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.user_int_schema.entries)
        }
        item_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.item_int_schema.entries)
        }
        try:
            user_ns_groups = [
                [user_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg['user_ns_groups'].values()
            ]
            item_ns_groups = [
                [item_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg['item_ns_groups'].values()
            ]
        except KeyError as exc:
            raise KeyError(
                f"NS-groups JSON references fid {exc.args[0]} which is not "
                f"present in the checkpoint's schema.json. The ns_groups.json "
                f"and schema.json must come from the same training run."
            ) from exc
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(dataset.item_int_schema.entries))]

    # Feature specs.
    user_int_feature_specs = build_feature_specs(
        dataset.user_int_schema, dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        dataset.item_int_schema, dataset.item_int_vocab_sizes)

    logging.info(f"Building PCVRHyFormer with cfg: {model_cfg}")
    model = PCVRHyFormer(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        **model_cfg,
    ).to(device)

    return model


def load_model_state_strict(
    model: nn.Module,
    ckpt_path: str,
    device: str,
) -> None:
    """Strictly load ``state_dict``; any missing/unexpected key fails fast
    with a diagnostic message.
    """
    state_dict = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        logging.error(
            "Failed to load state_dict in strict mode. This usually means the "
            "model constructed by build_model does NOT match the checkpoint. "
            "Check that train_config.json in the ckpt dir is present and matches "
            "the training hyperparameters.")
        raise e


def get_ckpt_path() -> Optional[str]:
    """Locate the first ``*.pt`` file inside the directory pointed at by
    ``$MODEL_OUTPUT_PATH``. Returns ``None`` if no checkpoint is found.
    """
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if not ckpt_path:
        return None
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)
    return None


def _batch_to_model_input(
    batch: Dict[str, Any],
    device: str,
) -> ModelInput:
    """Convert a batch dict to ``ModelInput``, handling dynamic seq domains."""
    device_batch: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            device_batch[k] = v.to(device, non_blocking=True)
        else:
            device_batch[k] = v

    seq_domains = device_batch['_seq_domains']
    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_time_buckets: Dict[str, torch.Tensor] = {}
    for domain in seq_domains:
        seq_data[domain] = device_batch[domain]
        seq_lens[domain] = device_batch[f'{domain}_len']
        B, _, L = device_batch[domain].shape
        seq_time_buckets[domain] = device_batch.get(
            f'{domain}_time_bucket',
            torch.zeros(B, L, dtype=torch.long, device=device))

    return ModelInput(
        user_int_feats=device_batch['user_int_feats'],
        item_int_feats=device_batch['item_int_feats'],
        user_dense_feats=device_batch['user_dense_feats'],
        item_dense_feats=device_batch['item_dense_feats'],
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
    )



# -----------------------------
# Log-only test feature probe
# -----------------------------
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _quantiles(values: List[float], qs: List[float]) -> Dict[str, float]:
    if not values:
        return {f"p{int(q * 1000) / 10:g}": float('nan') for q in qs}
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return {f"p{int(q * 1000) / 10:g}": float('nan') for q in qs}
    n = len(vals)
    out: Dict[str, float] = {}
    for q in qs:
        idx = min(n - 1, max(0, int(round(q * (n - 1)))))
        key = f"p{int(q * 1000) / 10:g}"
        out[key] = vals[idx]
    return out


def _short_hash(x: Any) -> str:
    s = str(x).encode('utf-8', errors='ignore')
    return hashlib.sha1(s).hexdigest()[:12]


class SparseFieldProbe:
    """Streaming sparse field statistics. It prints aggregate statistics only."""
    def __init__(self, name: str, schema: FeatureSchema, max_counter_size: int = 20000):
        self.name = name
        self.entries = list(schema.entries)
        self.max_counter_size = max_counter_size
        self.n_rows = 0
        self.total_pos = [0 for _ in self.entries]
        self.zero_pos = [0 for _ in self.entries]
        self.any_nonzero_rows = [0 for _ in self.entries]
        self.active_pos_sum = [0 for _ in self.entries]
        self.unique_sets = [set() for _ in self.entries]
        self.unique_overflow = [False for _ in self.entries]
        self.top_counters = [Counter() for _ in self.entries]

    def update(self, tensor: torch.Tensor) -> None:
        if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return
        x = tensor.detach().cpu().long()
        if x.dim() == 1:
            x = x.view(-1, 1)
        B = int(x.shape[0])
        self.n_rows += B
        for i, (fid, offset, length) in enumerate(self.entries):
            if offset >= x.shape[1]:
                continue
            end = min(offset + length, x.shape[1])
            v = x[:, offset:end]
            pos_n = int(v.numel())
            self.total_pos[i] += pos_n
            nonzero = v > 0
            nz_count = int(nonzero.sum().item())
            self.zero_pos[i] += pos_n - nz_count
            self.active_pos_sum[i] += nz_count
            self.any_nonzero_rows[i] += int(nonzero.any(dim=1).sum().item()) if v.numel() else 0
            vals = v[nonzero].flatten().tolist()
            if vals:
                if not self.unique_overflow[i]:
                    s = self.unique_sets[i]
                    for val in vals[:5000]:
                        s.add(int(val))
                        if len(s) > self.max_counter_size:
                            self.unique_overflow[i] = True
                            break
                c = self.top_counters[i]
                c.update(int(val) for val in vals)
                if len(c) > self.max_counter_size * 2:
                    self.top_counters[i] = Counter(dict(c.most_common(self.max_counter_size)))

    def log(self, max_fields: int = 120, topk: int = 5) -> None:
        logging.info(f"[TEST_PROBE][{self.name}] fields={len(self.entries)} rows={self.n_rows}")
        rows = []
        for i, (fid, offset, length) in enumerate(self.entries):
            total = max(1, self.total_pos[i])
            zero_rate = self.zero_pos[i] / total
            nonzero_row_rate = self.any_nonzero_rows[i] / max(1, self.n_rows)
            active_mean = self.active_pos_sum[i] / max(1, self.n_rows)
            uniq = len(self.unique_sets[i])
            if self.unique_overflow[i]:
                uniq_s = f">={uniq}"
            else:
                uniq_s = str(uniq)
            top_items = self.top_counters[i].most_common(topk)
            top_s = ",".join([f"{k}:{v / max(1, self.active_pos_sum[i]):.4f}" for k, v in top_items])
            rows.append((fid, offset, length, zero_rate, nonzero_row_rate, active_mean, uniq_s, top_s))
        # Print all fields if not too many; otherwise print the most informative high-zero / high-cardinality fields.
        if len(rows) > max_fields:
            rows = sorted(rows, key=lambda r: (r[3], str(r[6])), reverse=True)[:max_fields]
        for fid, offset, length, zero_rate, nonzero_row_rate, active_mean, uniq_s, top_s in rows:
            logging.info(
                f"[TEST_PROBE][{self.name}] fid={fid} offset={offset} len={length} "
                f"zero_pos_rate={zero_rate:.6f} nonzero_row_rate={nonzero_row_rate:.6f} "
                f"active_pos_mean={active_mean:.4f} approx_unique={uniq_s} top{topk}_id_share=[{top_s}]"
            )


class DenseProbe:
    def __init__(self, name: str, max_sample_rows: int = 20000):
        self.name = name
        self.n = 0
        self.dim = None
        self.sum = None
        self.sumsq = None
        self.min = None
        self.max = None
        self.zero = None
        self.nan = None
        self.sample_chunks: List[torch.Tensor] = []
        self.max_sample_rows = max_sample_rows
        self.sample_rows = 0

    def update(self, tensor: torch.Tensor) -> None:
        if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return
        x = tensor.detach().cpu().float()
        if x.dim() == 1:
            x = x.view(-1, 1)
        B, D = int(x.shape[0]), int(x.shape[1])
        if D == 0:
            return
        finite = torch.isfinite(x)
        x_safe = torch.where(finite, x, torch.zeros_like(x))
        if self.dim is None:
            self.dim = D
            self.sum = torch.zeros(D)
            self.sumsq = torch.zeros(D)
            self.min = torch.full((D,), float('inf'))
            self.max = torch.full((D,), float('-inf'))
            self.zero = torch.zeros(D)
            self.nan = torch.zeros(D)
        self.n += B
        self.sum += x_safe.sum(dim=0)
        self.sumsq += (x_safe * x_safe).sum(dim=0)
        self.min = torch.minimum(self.min, torch.where(finite, x, torch.full_like(x, float('inf'))).min(dim=0).values)
        self.max = torch.maximum(self.max, torch.where(finite, x, torch.full_like(x, float('-inf'))).max(dim=0).values)
        self.zero += ((x == 0) & finite).sum(dim=0).float()
        self.nan += (~finite).sum(dim=0).float()
        if self.sample_rows < self.max_sample_rows:
            take = min(B, self.max_sample_rows - self.sample_rows)
            self.sample_chunks.append(x_safe[:take].clone())
            self.sample_rows += take

    def log(self, max_dims: int = 80) -> None:
        if self.dim is None or self.n == 0:
            logging.info(f"[TEST_PROBE][{self.name}] empty")
            return
        mean = self.sum / max(1, self.n)
        var = torch.clamp(self.sumsq / max(1, self.n) - mean * mean, min=0.0)
        std = torch.sqrt(var)
        zero_rate = self.zero / max(1, self.n)
        nan_rate = self.nan / max(1, self.n)
        sample = torch.cat(self.sample_chunks, dim=0) if self.sample_chunks else None
        logging.info(f"[TEST_PROBE][{self.name}] rows={self.n} dim={self.dim} sampled_rows_for_quantile={self.sample_rows}")
        dim_indices = list(range(self.dim))
        if self.dim > max_dims:
            # Prioritize dimensions with high std, high zero rate, or non-finite values.
            score = std + zero_rate + 10.0 * nan_rate
            dim_indices = torch.argsort(score, descending=True).tolist()[:max_dims]
        for j in dim_indices:
            q_s = ""
            if sample is not None and sample.shape[0] > 0:
                col = sample[:, j].tolist()
                qs = _quantiles(col, [0.01, 0.5, 0.99])
                q_s = f" p1={qs['p1']:.6g} p50={qs['p50']:.6g} p99={qs['p99']:.6g}"
            logging.info(
                f"[TEST_PROBE][{self.name}] dim={j} mean={mean[j].item():.6g} std={std[j].item():.6g} "
                f"min={self.min[j].item():.6g} max={self.max[j].item():.6g} "
                f"zero_rate={zero_rate[j].item():.6f} nan_rate={nan_rate[j].item():.6f}{q_s}"
            )


class SeqProbe:
    def __init__(self, domain: str):
        self.domain = domain
        self.lengths: List[int] = []
        self.rows = 0
        self.empty = 0
        self.pos_positions = 0
        self.total_positions = 0
        self.row_any = 0
        self.time_bucket_counter = Counter()

    def update(self, seq_tensor: torch.Tensor, len_tensor: torch.Tensor, tb_tensor: Optional[torch.Tensor]) -> None:
        if seq_tensor is None or len_tensor is None:
            return
        x = seq_tensor.detach().cpu()
        lens = len_tensor.detach().cpu().long().view(-1)
        B = int(lens.numel())
        self.rows += B
        self.lengths.extend([int(v) for v in lens.tolist()])
        self.empty += int((lens <= 0).sum().item())
        if isinstance(x, torch.Tensor) and x.numel() > 0:
            # Expected shape: [B, F, L]
            if x.dim() == 3:
                pos_any = (x > 0).any(dim=1)
            elif x.dim() == 2:
                pos_any = x > 0
            else:
                pos_any = x.view(B, -1) > 0
            self.pos_positions += int(pos_any.sum().item())
            self.total_positions += int(pos_any.numel())
            self.row_any += int(pos_any.any(dim=1).sum().item())
        if tb_tensor is not None and isinstance(tb_tensor, torch.Tensor) and tb_tensor.numel() > 0:
            tb = tb_tensor.detach().cpu().long().flatten().tolist()
            self.time_bucket_counter.update(int(v) for v in tb)

    def log(self, topk_buckets: int = 20) -> None:
        qs = _quantiles([float(v) for v in self.lengths], [0.0, 0.5, 0.9, 0.95, 0.99, 1.0])
        pos_rate = self.pos_positions / max(1, self.total_positions)
        row_any_rate = self.row_any / max(1, self.rows)
        empty_rate = self.empty / max(1, self.rows)
        mean_len = sum(self.lengths) / max(1, len(self.lengths))
        bucket_total = sum(self.time_bucket_counter.values())
        top_buckets = ",".join([f"{k}:{v / max(1, bucket_total):.4f}" for k, v in self.time_bucket_counter.most_common(topk_buckets)])
        logging.info(
            f"[TEST_PROBE][seq_{self.domain}] rows={self.rows} empty_rate={empty_rate:.6f} "
            f"mean_len={mean_len:.3f} p50_len={qs['p50']:.1f} p90_len={qs['p90']:.1f} "
            f"p95_len={qs['p95']:.1f} p99_len={qs['p99']:.1f} max_len={qs['p100']:.1f} "
            f"row_any_id_rate={row_any_rate:.6f} position_positive_rate={pos_rate:.6f} "
            f"top_time_bucket_share=[{top_buckets}]"
        )


class LogOnlyTestProbe:
    def __init__(self, dataset: PCVRParquetDataset, sample_n: int = 12):
        self.user_sparse = SparseFieldProbe('user_int', dataset.user_int_schema)
        self.item_sparse = SparseFieldProbe('item_int', dataset.item_int_schema)
        self.user_dense = DenseProbe('user_dense')
        self.item_dense = DenseProbe('item_dense')
        self.seq_probes: Dict[str, SeqProbe] = {}
        self.probs: List[float] = []
        self.sample_n = sample_n
        self.sample_rows: List[Dict[str, Any]] = []
        self.first_batch_logged = False

    def update(self, batch: Dict[str, Any], probs: torch.Tensor, batch_idx: int) -> None:
        if not self.first_batch_logged:
            key_shapes = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    key_shapes[k] = list(v.shape)
                elif isinstance(v, list):
                    key_shapes[k] = f"list_len={len(v)}"
                else:
                    key_shapes[k] = type(v).__name__
            logging.info(f"[TEST_PROBE] first_batch_keys_and_shapes={key_shapes}")
            self.first_batch_logged = True
        p_list = [float(x) for x in probs.detach().cpu().view(-1).tolist()]
        self.probs.extend(p_list)
        self.user_sparse.update(batch.get('user_int_feats'))
        self.item_sparse.update(batch.get('item_int_feats'))
        self.user_dense.update(batch.get('user_dense_feats'))
        self.item_dense.update(batch.get('item_dense_feats'))
        seq_domains = batch.get('_seq_domains', [])
        for domain in seq_domains:
            if domain not in self.seq_probes:
                self.seq_probes[domain] = SeqProbe(domain)
            self.seq_probes[domain].update(
                batch.get(domain), batch.get(f'{domain}_len'), batch.get(f'{domain}_time_bucket')
            )
        # Print only compact per-sample fingerprints for first N rows, not raw feature dumps.
        if len(self.sample_rows) < self.sample_n:
            B = len(p_list)
            user_ids = batch.get('user_id', [])
            user_int = batch.get('user_int_feats')
            item_int = batch.get('item_int_feats')
            user_dense = batch.get('user_dense_feats')
            item_dense = batch.get('item_dense_feats')
            for i in range(min(B, self.sample_n - len(self.sample_rows))):
                uid = user_ids[i] if isinstance(user_ids, list) and i < len(user_ids) else i + batch_idx * B
                row = {
                    'row_no': batch_idx * B + i,
                    'user_id_hash': _short_hash(uid),
                    'pred': p_list[i],
                    'user_int_zero_rate': None,
                    'item_int_zero_rate': None,
                    'user_dense_norm': None,
                    'item_dense_norm': None,
                    'seq_lens': {},
                }
                if isinstance(user_int, torch.Tensor) and user_int.numel() > 0:
                    row['user_int_zero_rate'] = float((user_int[i].detach().cpu() <= 0).float().mean().item())
                if isinstance(item_int, torch.Tensor) and item_int.numel() > 0:
                    row['item_int_zero_rate'] = float((item_int[i].detach().cpu() <= 0).float().mean().item())
                if isinstance(user_dense, torch.Tensor) and user_dense.numel() > 0:
                    row['user_dense_norm'] = float(torch.nan_to_num(user_dense[i].detach().cpu().float()).norm().item())
                if isinstance(item_dense, torch.Tensor) and item_dense.numel() > 0:
                    row['item_dense_norm'] = float(torch.nan_to_num(item_dense[i].detach().cpu().float()).norm().item())
                for domain in seq_domains:
                    lens = batch.get(f'{domain}_len')
                    if isinstance(lens, torch.Tensor) and lens.numel() > i:
                        row['seq_lens'][domain] = int(lens[i].detach().cpu().item())
                self.sample_rows.append(row)

    def log_final(self) -> None:
        logging.info("[TEST_PROBE] ==================== LOG-ONLY TEST PROBE SUMMARY BEGIN ====================")
        n = len(self.probs)
        if n > 0:
            probs = [float(x) for x in self.probs]
            mean = sum(probs) / n
            var = sum((x - mean) * (x - mean) for x in probs) / max(1, n)
            qs = _quantiles(probs, [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0])
            logging.info(
                f"[TEST_PROBE][prediction_distribution] n={n} mean={mean:.8f} std={math.sqrt(var):.8f} "
                f"min={qs['p0']:.8f} p0.1={qs['p0.1']:.8f} p1={qs['p1']:.8f} p5={qs['p5']:.8f} "
                f"p10={qs['p10']:.8f} p25={qs['p25']:.8f} p50={qs['p50']:.8f} p75={qs['p75']:.8f} "
                f"p90={qs['p90']:.8f} p95={qs['p95']:.8f} p99={qs['p99']:.8f} p99.9={qs['p99.9']:.8f} max={qs['p100']:.8f}"
            )
            bins = [0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
            counts = [0 for _ in range(len(bins) - 1)]
            for x in probs:
                for j in range(len(bins) - 1):
                    if (x >= bins[j] and x < bins[j + 1]) or (j == len(bins) - 2 and x <= bins[j + 1]):
                        counts[j] += 1
                        break
            hist_s = "; ".join([f"[{bins[j]:g},{bins[j+1]:g})={counts[j]}({counts[j]/n:.4%})" for j in range(len(counts))])
            logging.info(f"[TEST_PROBE][prediction_histogram] {hist_s}")
        for row in self.sample_rows:
            logging.info(f"[TEST_PROBE][sample_fingerprint] {row}")
        self.user_sparse.log()
        self.item_sparse.log()
        self.user_dense.log()
        self.item_dense.log()
        for domain in sorted(self.seq_probes.keys()):
            self.seq_probes[domain].log()
        logging.info("[TEST_PROBE] ===================== LOG-ONLY TEST PROBE SUMMARY END =====================")


def main() -> None:
    # ---- Read environment variables ----
    model_dir = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir = os.environ.get('EVAL_DATA_PATH')
    result_dir = os.environ.get('EVAL_RESULT_PATH')

    os.makedirs(result_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- Schema: prefer the one from model_dir (to exactly match training);
    #      fall back to the one in data_dir if missing. ----
    schema_path = os.path.join(model_dir, 'schema.json')
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, 'schema.json')
    logging.info(f"Using schema: {schema_path}")

    # ---- Load train_config.json (single source of truth for all hyperparams) ----
    train_config = load_train_config(model_dir)

    # ---- Parse seq_max_lens ----
    sml_str = train_config.get('seq_max_lens', _FALLBACK_SEQ_MAX_LENS)
    seq_max_lens = _parse_seq_max_lens(sml_str)
    logging.info(f"seq_max_lens: {seq_max_lens}")

    # ---- Data loading: reuse batch_size / num_workers from training config ----
    batch_size = int(train_config.get('batch_size', _FALLBACK_BATCH_SIZE))
    num_workers = int(train_config.get('num_workers', _FALLBACK_NUM_WORKERS))

    test_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
    )
    total_test_samples = test_dataset.num_rows
    logging.info(f"Total test samples: {total_test_samples}")

    probe_enabled = os.environ.get('TEST_PROBE_ENABLE', '1') != '0'
    probe_sample_n = int(os.environ.get('TEST_PROBE_SAMPLE_N', '12'))
    test_probe = LogOnlyTestProbe(test_dataset, sample_n=probe_sample_n) if probe_enabled else None
    logging.info(f"TEST_PROBE_ENABLE={probe_enabled}, TEST_PROBE_SAMPLE_N={probe_sample_n}; probe is log-only and writes no extra files.")

    # ---- Build model: every structural hyperparameter is resolved from train_config ----
    model_cfg = resolve_model_cfg(train_config)

    # ns_groups_json also comes from training config (e.g. run.sh may have
    # passed an empty string to disable it). When trainer.py has copied the
    # JSON into the ckpt dir, train_config records just the basename, so try
    # resolving against ``model_dir`` first before honoring the raw (possibly
    # absolute) path as a fallback.
    ns_groups_json = train_config.get('ns_groups_json', None)
    if ns_groups_json:
        local_candidate = os.path.join(model_dir, os.path.basename(ns_groups_json))
        if os.path.exists(local_candidate):
            ns_groups_json = local_candidate

    model = build_model(
        test_dataset,
        model_cfg=model_cfg,
        ns_groups_json=ns_groups_json,
        device=device,
    )

    # ---- Strictly load weights ----
    ckpt_path = get_ckpt_path()
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No *.pt file found under MODEL_OUTPUT_PATH={model_dir!r}. "
            f"The directory contains: {os.listdir(model_dir) if model_dir and os.path.isdir(model_dir) else 'N/A'}. "
            "This typically means the training job wrote only the sidecar "
            "files (schema.json / train_config.json) for this step but did "
            "not persist model.pt — a symptom of a race between "
            "_remove_old_best_dirs and EarlyStopping.save_checkpoint."
        )
    logging.info(f"Loading checkpoint from {ckpt_path}")
    load_model_state_strict(model, ckpt_path, device)
    model.eval()
    logging.info("Model loaded successfully")

    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=torch.cuda.is_available(),
    )

    all_probs = []
    all_user_ids = []
    logging.info("Starting inference...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            model_input = _batch_to_model_input(batch, device)
            user_ids = batch.get('user_id', [])

            logits, _ = model.predict(model_input)
            logits = logits.squeeze(-1)
            probs_tensor = torch.sigmoid(logits).detach().cpu()
            if test_probe is not None:
                test_probe.update(batch, probs_tensor, batch_idx)
            probs = probs_tensor.numpy()
            all_probs.extend(probs.tolist())
            all_user_ids.extend(user_ids)

            if (batch_idx + 1) % 100 == 0:
                logging.info(f"  Processed {(batch_idx + 1) * batch_size} samples")

    logging.info(f"Inference complete: {len(all_probs)} predictions")
    if test_probe is not None:
        test_probe.log_final()

    predictions = {
        "predictions": dict(zip(all_user_ids, all_probs)),
    }

    # ---- Save predictions.json ----
    output_path = os.path.join(result_dir, 'predictions.json')
    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    logging.info(f"Saved {len(all_probs)} predictions to {output_path}")


if __name__ == "__main__":
    main()
