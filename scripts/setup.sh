#!/usr/bin/env bash
# Build the uv environment (.venv) following third_party/catk/README.md, translated to uv.
# Idempotent: re-running only installs what is missing. See docs/SETUP.md.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

[ -d third_party/catk ] || { echo "run 'just clone' first"; exit 1; }
[ -x .venv/bin/python ] || uv venv .venv --python 3.11.9
export VIRTUAL_ENV="$ROOT/.venv"
PY=.venv/bin/python

# tensorflow-cpu 2.12 needs numpy < 1.24; later installs must not upgrade it.
CONSTRAINTS="$ROOT/scripts/constraints.txt"

uv pip install --python $PY torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python $PY -c "$CONSTRAINTS" -r third_party/catk/install/requirements.txt
uv pip install --python $PY -c "$CONSTRAINTS" torch_geometric==2.6.1  # 2.8 makes BaseTransform.forward abstract (catk transforms define __call__)
uv pip install --python $PY -c "$CONSTRAINTS" torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
# --no-deps is required: its numpy/tf pins break py3.11 resolution (catk README does the same).
# 1.6.7 instead of catk's 1.6.4: TrajTok's fast-metric ground-truth converter needs
# submission_specs.ChallengeType and the 2025 sim-agents config, which 1.6.4 lacks.
uv pip install --python $PY --no-deps waymo-open-dataset-tf-2-12-0==1.6.7
uv pip install --python $PY -c "$CONSTRAINTS" scipy matplotlib immutabledict pytest

$PY - <<'EOF'
import numpy, torch, torch_geometric, torch_scatter, torch_cluster, lightning, tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2  # noqa: F401
print("numpy", numpy.__version__, "| torch", torch.__version__, "| PyG", torch_geometric.__version__,
      "| lightning", lightning.__version__, "| tf", tf.__version__)
assert torch.cuda.is_available(), "CUDA not available"
print("GPU:", torch.cuda.get_device_name(0))
EOF
