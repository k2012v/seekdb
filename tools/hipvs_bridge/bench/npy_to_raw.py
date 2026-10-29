#!/usr/bin/env python3
"""Convert .npy benchmark inputs to the raw little-endian files bench_cagra.c reads."""
import argparse
import os
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="contiguous float32 .npy")
    parser.add_argument("--queries", required=True, help="contiguous float32 .npy")
    parser.add_argument("--gt", required=True, help="int .npy with >=100 columns")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    base = np.load(args.base).astype(np.float32, copy=False)
    queries = np.load(args.queries).astype(np.float32, copy=False)
    gt = np.load(args.gt).astype(np.int32, copy=False)
    assert base.ndim == 2 and queries.ndim == 2, "base/queries must be 2-D"
    assert base.shape[1] == queries.shape[1], "dimension mismatch"
    assert gt.ndim == 2 and gt.shape[0] == queries.shape[0], "GT shape mismatch"

    os.makedirs(args.out_dir, exist_ok=True)
    base.tofile(f"{args.out_dir}/base.f32")
    queries.tofile(f"{args.out_dir}/queries.f32")
    gt.tofile(f"{args.out_dir}/gt100.i32")
    print(f"n={base.shape[0]} dim={base.shape[1]} nq={queries.shape[0]} gt_k={gt.shape[1]}")


if __name__ == "__main__":
    main()
