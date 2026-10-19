set shell := ["bash", "-uc"]
set dotenv-load := true

venv := ".venv/bin/python"

default:
    @just --list

# --- setup ---
clone:
    mkdir -p third_party
    [ -d third_party/catk ] || git clone https://github.com/NVlabs/catk third_party/catk
    [ -d third_party/TrajTok ] || git clone https://github.com/Thinklab-SJTU/TrajTok third_party/TrajTok

setup: clone
    @echo "See docs/SETUP.md (uv venv py3.11.9 + pinned torch/PyG/TF); scripts/setup.sh runs it."
    if [ -f scripts/setup.sh ]; then bash scripts/setup.sh; fi
    {{venv}} -c "import torch;assert torch.cuda.is_available();print(torch.cuda.get_device_name(0))"

import-data src:
    bash scripts/import_womd.sh {{src}}

data-subset:
    bash scripts/data_subset.sh      # see docs/SETUP.md

preprocess:
    CUDA_VISIBLE_DEVICES="" bash scripts/preprocess.sh

# --- checks ---
smoke:
    PYTHONPATH=src {{venv}} -m tokens_vs_flows.smoke

eval-sanity:
    {{venv}} eval/score.py --sanity --split val_frozen
