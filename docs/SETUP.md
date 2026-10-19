# Setup (uv + just)

## Tools (once per machine)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install rust-just          # provides `just`
uv tool install nvitop             # optional
sudo apt install -y ffmpeg         # catk installs it via conda; we use the system one
```

## Python environment
catk's tested stack (from its README) is Python 3.11.9, torch 2.4.1, lightning 2.4.0, PyG built for cu121, tensorflow-cpu 2.12, and waymo-open-dataset-tf-2-12-0==1.6.4. TrajTok uses the same pins. Keep them: version drift is the main failure mode for this stack.

```bash
uv venv .venv --python 3.11.9
source .venv/bin/activate
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install -c scripts/constraints.txt -r third_party/catk/install/requirements.txt   # pins torch 2.4.1, lightning 2.4.0, tensorflow-cpu 2.12.0, tfp 0.19.0, hydra 1.3.2
uv pip install -c scripts/constraints.txt torch_geometric==2.6.1   # 2.8 breaks catk's target builders (abstract BaseTransform.forward)
uv pip install -c scripts/constraints.txt torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
uv pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.7   # --no-deps is REQUIRED (its pins break py3.11); 1.6.7, not catk's 1.6.4: TrajTok's ground-truth converter and the 2025 metric config need it
uv pip install -c scripts/constraints.txt scipy matplotlib immutabledict pytest
python -c "import torch;print(torch.cuda.get_device_name(0), torch.__version__)"
```
`scripts/setup.sh` runs exactly this (`scripts/constraints.txt` keeps numpy < 1.24 for TensorFlow 2.12). It follows `third_party/catk/README.md`, with two deliberate changes: torch_geometric pinned to 2.6.1 and waymo-open-dataset 1.6.7. `just clone` checks out pinned commits of catk and TrajTok and applies `patches/` to catk.

## Data (WOMD v1.3.0 scenario format)
Download WOMD v1.3.0 from the [official website](https://waymo.com/open/download/) (`waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario`). This project uses a subset: 100 training shards (`training.tfrecord-00000..00099-of-01000`, ~49k scenarios, 467-540 per shard) and 100 validation shards (`validation.tfrecord-00000..00099-of-00150`, ~290 per shard). Research use only; never re-host it.

1. Import once: `just import-data <SRC>/uncompressed/scenario`. This moves the shards into `data/womd/uncompressed/scenario/{training,validation,validation_spare}/`, makes only 8 val shards active (~2,300 scenarios), and verifies record counts. If `data/womd/uncompressed/scenario/training` already holds 100 shards, skip this step. The data is organised as follows:
   ```
   data/womd/uncompressed/scenario/
   ├── training/          100 shards
   ├── validation/        8 active shards (0-3: dev split, 4-7: held-out split)
   └── validation_spare/  the remaining validation shards
   ```
2. Splits (`just data-subset`, via `scripts/data_subset.sh`): parse scenario IDs with `waymo_open_dataset.protos.scenario_pb2`. val_frozen = 1000 scenarios from val shards 0-3; test_hidden = 1000 from val shards 4-7 (different shards). Write `eval/splits/val_frozen.txt` and `eval/splits/test_hidden.txt`.
3. Preprocess (`just preprocess` = `scripts/preprocess.sh`, CPU-only): runs catk's `python -m src.data_preprocess` module unchanged from third_party/catk with the uv venv, for `validation` then `training` (catk's `scripts/cache_womd.sh` hardcodes conda and paths). It globs `<input_dir>/<split>/*` non-recursively, so a split folder must hold only shards; for validation it also writes per-scenario tfrecords to `validation_tfrecords_splitted/` (official metric). `scripts/extract_val_gt.py` writes the ground-truth pickles of the fast metric (TrajTok's `extract_gt_scenario`) to `validation_gt/`. Measured on 16 threads: validation 78 s, training 854 s.
4. Disk (measured): 69 GB raw (100 train shards, 8 active + 92 spare val shards); cache 13 GB (training pickles 8.8 GB, validation pickles 0.4 GB, val ground truth 1.9 GB, val tfrecords 2.1 GB).
5. catk was built on WOMD v1.2.1; we use v1.3.0 (TrajTok needs v1.3.x). Field mismatches are fixed with a patch in patches/, not in place.

## Repos
`just clone` puts NVlabs/catk and Thinklab-SJTU/TrajTok in third_party/. Project code lives in `src/tokens_vs_flows/` and imports catk modules. Don't fork-edit catk in place, except for the head hook (the head interface in DESIGN.md), done as a small, documented patch file in patches/.

## Licence
WOMD's terms forbid sharing weights trained on it. Never push checkpoints to a public repo. Code and figures are fine.
