#!/usr/bin/env bash
# CPU-only WOMD preprocessing into data/cache/SMART:
#   validation/                      catk scenario pickles (8 active val shards)
#   validation_tfrecords_splitted/   one tfrecord per val scenario (official WOSAC metric)
#   validation_gt/                   ground-truth pickles for the fast WOSAC metric (TrajTok converter)
#   training/                        catk scenario pickles (100 train shards)
# Runs catk's own preprocessing module unchanged. Usage: bash scripts/preprocess.sh [num_workers]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
IN="$ROOT/data/womd/uncompressed/scenario"
OUT="$ROOT/data/cache/SMART"
NW="${1:-14}"
export CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=2 OMP_NUM_THREADS=1
mkdir -p "$OUT"

t0=$(date +%s)
(cd "$ROOT/third_party/catk" && "$PY" -m src.data_preprocess --split validation --num_workers 8 \
    --input_dir "$IN" --output_dir "$OUT") &
"$PY" "$ROOT/scripts/extract_val_gt.py" --input_dir "$IN/validation" --output_dir "$OUT/validation_gt" --num_workers 6 &
wait
echo "validation done in $(( $(date +%s) - t0 )) s: $(ls "$OUT/validation" | wc -l) pickles, $(ls "$OUT/validation_gt" | wc -l) gt"

t0=$(date +%s)
(cd "$ROOT/third_party/catk" && "$PY" -m src.data_preprocess --split training --num_workers "$NW" \
    --input_dir "$IN" --output_dir "$OUT")
echo "training done in $(( $(date +%s) - t0 )) s: $(ls "$OUT/training" | wc -l) pickles"
du -sh "$OUT"/*
