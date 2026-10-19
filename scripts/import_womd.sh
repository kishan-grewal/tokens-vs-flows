#!/usr/bin/env bash
# Organise downloaded WOMD v1.3.0 scenario shards into this repo's layout.
#
# Usage:  bash scripts/import_womd.sh <SRC_SCENARIO_DIR> [--copy] [--val-shards N]
#   SRC_SCENARIO_DIR  the downloaded uncompressed/scenario folder, with training/ and
#                     validation/ holding the *.tfrecord-* shards
#   --copy            copy instead of move (default: move, which is instant on the same disk)
#   --val-shards N    validation shards made active for catk preprocessing (default 8, ~2,300
#                     scenarios: shards 0-3 feed val_frozen, 4-7 feed test_hidden). The rest go to
#                     validation_spare/, untouched.
#
# Result (catk's src.data_preprocess globs <input_dir>/<split>/* NON-recursively, so each split
# folder must hold only tfrecord shards):
#   data/womd/uncompressed/scenario/training/          100 shards
#   data/womd/uncompressed/scenario/validation/        N shards
#   data/womd/uncompressed/scenario/validation_spare/  remaining val shards
set -euo pipefail

SRC="${1:?usage: import_womd.sh <SRC_SCENARIO_DIR> [--copy] [--val-shards N]}"; shift
MODE=mv; VAL_N=8
while [ $# -gt 0 ]; do
  case "$1" in
    --copy) MODE="cp"; shift ;;
    --val-shards) VAL_N="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="$ROOT/data/womd/uncompressed/scenario"
mkdir -p "$DST/training" "$DST/validation" "$DST/validation_spare"

move_split () {  # $1 split name in SRC, $2 dest dir, $3 max count (0 = all), $4 skip first K
  local split="$1" dest="$2" max="$3" skip="$4" i=0 n=0
  while IFS= read -r f; do
    i=$((i+1)); [ "$i" -le "$skip" ] && continue
    [ "$max" -gt 0 ] && [ "$n" -ge "$max" ] && break
    if [ "$MODE" = mv ]; then mv -n "$f" "$dest/"; else cp -n "$f" "$dest/"; fi
    n=$((n+1))
  done < <(find "$SRC/$split" -type f -name "*tfrecord*" | sort)
  echo "$split -> $(basename "$dest"): $n shards ($MODE)"
}

move_split training   "$DST/training"         0        0
move_split validation "$DST/validation"       "$VAL_N" 0
move_split validation "$DST/validation_spare" 0        "$VAL_N"

# Verify: count records per shard (TFRecord framing, no TensorFlow needed) and catch truncation.
python3 - "$DST" <<'EOF'
import glob, struct, sys
root = sys.argv[1]
def count(p):
    n = 0
    with open(p, 'rb') as f:
        while True:
            h = f.read(8)
            if not h: return n
            if len(h) < 8: raise SystemExit(f'TRUNCATED header: {p}')
            (L,) = struct.unpack('<Q', h); f.read(4)
            if len(f.read(L)) < L: raise SystemExit(f'TRUNCATED record: {p}')
            f.read(4); n += 1
for split in ['training', 'validation', 'validation_spare']:
    fs = sorted(glob.glob(f'{root}/{split}/*'))
    bad = [p for p in fs if 'tfrecord' not in p]
    if bad: raise SystemExit(f'non-shard files in {split}: {bad[:3]}')
    c = [count(p) for p in fs] if split != 'training' else [count(p) for p in fs[:5]]
    tag = 'all' if split != 'training' else 'first 5'
    print(f'{split:17} {len(fs):4} shards; records ({tag}): sum={sum(c)} min={min(c, default=0)} max={max(c, default=0)}')
EOF
du -sh "$DST"
echo "OK."
