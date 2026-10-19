"""Write one ground-truth pickle per scenario for the fast WOSAC metric (validation, or training shards for rewards).

Uses TrajTok's converter (wosac_fast_eval_tool.scenario_gt_converter.extract_gt_scenario), one
worker per shard. Also writes <output_dir>_index.tsv (scenario_id, shard file).
Usage: extract_val_gt.py --input_dir <dir of shards> --output_dir <dir> [--num_workers 8] [--max_shards N]
"""
import argparse
import os
import pickle
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "TrajTok"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def process_shard(shard: str, output_dir: str):
    import tensorflow as tf
    import torch
    from waymo_open_dataset.protos import scenario_pb2
    from wosac_fast_eval_tool.scenario_gt_converter import extract_gt_scenario

    torch.set_num_threads(1)
    ids = []
    for rec in tf.data.TFRecordDataset(shard, compression_type=""):
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytes(rec.numpy()))
        out = Path(output_dir) / f"{scenario.scenario_id}.pkl"
        if not out.exists():
            with open(out, "wb") as f:
                pickle.dump(extract_gt_scenario(scenario), f)
        ids.append(scenario.scenario_id)
    return Path(shard).name, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_shards", type=int, default=0)
    args = ap.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    shards = sorted(p.as_posix() for p in Path(args.input_dir).glob("*tfrecord*"))
    if args.max_shards:
        shards = shards[:args.max_shards]
    with Pool(args.num_workers) as pool:
        results = pool.map(partial(process_shard, output_dir=args.output_dir), shards)
    index = Path(str(Path(args.output_dir)) + "_index.tsv")
    with open(index, "w") as f:
        for shard, ids in sorted(results):
            for sid in ids:
                f.write(f"{sid}\t{shard}\n")
    print(f"{sum(len(i) for _, i in results)} scenarios from {len(shards)} shards -> {args.output_dir}")


if __name__ == "__main__":
    main()
