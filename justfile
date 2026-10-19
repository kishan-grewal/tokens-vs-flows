set shell := ["bash", "-uc"]
set dotenv-load := true

venv := ".venv/bin/python"

default:
    @just --list

# --- setup ---
catk_commit := "d23886761fc5b5628c5973148c40284452745745"
trajtok_commit := "7a9bb997d304d8fc604061c8137b1e109c618696"

clone:
    mkdir -p third_party
    [ -d third_party/catk ] || { git clone https://github.com/NVlabs/catk third_party/catk && git -C third_party/catk checkout -q {{catk_commit}}; }
    [ -d third_party/TrajTok ] || { git clone https://github.com/Thinklab-SJTU/TrajTok third_party/TrajTok && git -C third_party/TrajTok checkout -q {{trajtok_commit}}; }
    just patch

# Apply patches/ to third_party/catk (idempotent).
patch:
    for p in patches/*.patch; do \
      if git -C third_party/catk apply --reverse --check "../../$p" 2>/dev/null; then echo "already applied: $p"; \
      else git -C third_party/catk apply "../../$p" && echo "applied: $p"; fi; \
    done

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
