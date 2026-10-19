"""Fixed evaluation splits from the 8 active validation shards (disjoint shards per split).

val_frozen  = 1000 scenarios sampled (seed 0) from validation shards 0-3 (dev set)
test_hidden = 1000 scenarios sampled (seed 0) from validation shards 4-7 (held out, scored once)
Reads scenario ids straight from the shards; writes eval/splits/{val_frozen,test_hidden}.txt (sorted).
"""
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402
from waymo_open_dataset.protos import scenario_pb2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "data" / "womd" / "uncompressed" / "scenario" / "validation"
OUT = ROOT / "eval" / "splits"
N = 1000


def shard_ids(shard: Path):
    ids = []
    for rec in tf.data.TFRecordDataset(str(shard), compression_type=""):
        s = scenario_pb2.Scenario()
        s.ParseFromString(bytes(rec.numpy()))
        ids.append(s.scenario_id)
    return ids


def main():
    shards = sorted(VAL.glob("*tfrecord*"))
    if len(shards) != 8:
        sys.exit(f"expected 8 active validation shards, found {len(shards)}")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, group in (("val_frozen", shards[:4]), ("test_hidden", shards[4:])):
        pool = sorted({i for s in group for i in shard_ids(s)})
        pick = sorted(random.Random(0).sample(pool, N))
        (OUT / f"{name}.txt").write_text("\n".join(pick) + "\n")
        print(f"{name}: {N} of {len(pool)} scenarios from {[s.name for s in group]}")


if __name__ == "__main__":
    main()
