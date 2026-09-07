"""Prepare the official 1k-row HuggingFace sample for local smoke tests.

This script:
- downloads demo_1000.parquet from HF (TAAC2026/data_sample_1000), or reuses
  a local copy when HF is unreachable
- splits it into two local parquet shards so the smoke test has train/valid rows
- generates a debug schema.json based on the sample (NOT the official vocab)

The generated schema is only meant to make the project runnable on the sample.
Do NOT use it for official training.

Usage:
  python tools/prepare_hf_sample.py --out_dir ./data_sample_1000

Then:
  bash run.sh --data_dir ./data_sample_1000 --schema_path ./data_sample_1000/schema.json

Notes:
- Requires: pyarrow, numpy, huggingface_hub
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _as_array(arr: pa.Array | pa.ChunkedArray) -> pa.Array:
    if isinstance(arr, pa.ChunkedArray):
        return arr.combine_chunks()
    return arr


def _max_in_list_array(arr: pa.Array | pa.ChunkedArray) -> tuple[int, int]:
    """Return (max_value, max_len) for an integer ListArray."""
    arr = _as_array(arr)
    offsets = arr.offsets.to_numpy()
    lens = offsets[1:] - offsets[:-1]
    max_len = int(lens.max()) if len(lens) else 0

    values = arr.values
    if len(values) == 0:
        return 0, max_len

    v = values.to_numpy(zero_copy_only=False)
    if v.dtype.kind == "f":
        v = v[~np.isnan(v)]
    if len(v) == 0:
        return 0, max_len
    return int(np.max(v)), max_len


def _max_in_scalar(col: pa.Array | pa.ChunkedArray) -> int:
    col = _as_array(col)
    v = col.to_numpy(zero_copy_only=False)
    if v.dtype.kind == "f":
        v = v[~np.isnan(v)]
    if len(v) == 0:
        return 0
    return int(np.max(v))


def build_debug_schema(parquet_path: str) -> dict:
    table = pq.read_table(parquet_path)
    names = table.schema.names

    schema: dict = {"user_int": [], "item_int": [], "user_dense": [], "seq": {}}

    seq_cols: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    pattern = re.compile(r"^(domain_[abcd]_seq)_(\d+)$")
    for name in names:
        match = pattern.match(name)
        if not match:
            continue
        prefix_base, fid = match.group(1), int(match.group(2))
        domain = prefix_base.replace("_seq", "")
        seq_cols[domain].append((fid, name, prefix_base))

    for name in names:
        if name.startswith("user_int_feats_"):
            fid = int(name.split("_")[-1])
            col = table[name]
            if pa.types.is_list(col.type):
                mx, max_len = _max_in_list_array(col)
                schema["user_int"].append([fid, mx + 1, max_len])
            else:
                mx = _max_in_scalar(col)
                schema["user_int"].append([fid, mx + 1, 1])

        elif name.startswith("item_int_feats_"):
            fid = int(name.split("_")[-1])
            col = table[name]
            if pa.types.is_list(col.type):
                mx, max_len = _max_in_list_array(col)
                schema["item_int"].append([fid, mx + 1, max_len])
            else:
                mx = _max_in_scalar(col)
                schema["item_int"].append([fid, mx + 1, 1])

        elif name.startswith("user_dense_feats_"):
            fid = int(name.split("_")[-1])
            col = table[name]
            if pa.types.is_list(col.type):
                col = _as_array(col)
                offsets = col.offsets.to_numpy()
                lens = offsets[1:] - offsets[:-1]
                max_dim = int(lens.max()) if len(lens) else 0
            else:
                max_dim = 1
            schema["user_dense"].append([fid, max_dim])

    for domain, cols in seq_cols.items():
        cols = sorted(cols, key=lambda x: x[0])
        prefix = cols[0][2]
        feats = []
        for fid, colname, _ in cols:
            col = table[colname]
            if pa.types.is_list(col.type):
                mx, _ = _max_in_list_array(col)
            else:
                mx = _max_in_scalar(col)
            feats.append([fid, mx + 1])
        schema["seq"][domain] = {
            "prefix": prefix,
            "ts_fid": None,
            "features": feats,
        }

    schema["user_int"] = sorted(schema["user_int"], key=lambda x: x[0])
    schema["item_int"] = sorted(schema["item_int"], key=lambda x: x[0])
    schema["user_dense"] = sorted(schema["user_dense"], key=lambda x: x[0])
    schema["seq"] = {k: schema["seq"][k] for k in sorted(schema["seq"].keys())}

    return schema


def _local_sample_candidates() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    workspace_dir = os.path.dirname(os.path.dirname(project_dir))
    return [
        os.path.join(project_dir, "demo_1000.parquet"),
        os.path.join(project_dir, "data_sample_1000", "demo_1000.parquet"),
        os.path.join(
            workspace_dir,
            "TAAC-baseline",
            "TAAC2026Baseline-main",
            "demo_1000.parquet",
        ),
    ]


def resolve_sample_path(source_parquet: str | None) -> str:
    if source_parquet:
        if not os.path.exists(source_parquet):
            raise FileNotFoundError(f"--source_parquet not found: {source_parquet}")
        return source_parquet

    try:
        from huggingface_hub import hf_hub_download  # type: ignore

        return hf_hub_download(
            repo_id="TAAC2026/data_sample_1000",
            filename="demo_1000.parquet",
            repo_type="dataset",
        )
    except Exception as exc:
        print(f"HF download failed, trying local sample fallback: {exc}", file=sys.stderr)

    for candidate in _local_sample_candidates():
        if os.path.exists(candidate):
            print(f"Using local sample: {candidate}", file=sys.stderr)
            return candidate

    raise RuntimeError(
        "Could not download the HF sample and no local demo_1000.parquet was found. "
        "Pass --source_parquet /path/to/demo_1000.parquet to run offline."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--source_parquet",
        type=str,
        default=None,
        help="Optional local demo_1000.parquet path for offline smoke tests.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    schema_out = os.path.join(args.out_dir, "schema.json")
    train_out = os.path.join(args.out_dir, "demo_1000_train.parquet")
    valid_out = os.path.join(args.out_dir, "demo_1000_valid.parquet")

    path = resolve_sample_path(args.source_parquet)

    table = pq.read_table(path)
    num_rows = table.num_rows
    split_idx = max(1, int(num_rows * 0.9))

    # Keep the smoke directory deterministic if the script is re-run.
    for name in os.listdir(args.out_dir):
        if name.startswith("demo_1000") and name.endswith(".parquet"):
            os.remove(os.path.join(args.out_dir, name))

    pq.write_table(table.slice(0, split_idx), train_out)
    pq.write_table(table.slice(split_idx), valid_out)

    schema = build_debug_schema(path)
    with open(schema_out, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {train_out}")
    print(f"Wrote: {valid_out}")
    print(f"Wrote: {schema_out}")
    print("NOTE: schema.json is derived from the 1k-row sample and is only for smoke tests.")


if __name__ == "__main__":
    main()
