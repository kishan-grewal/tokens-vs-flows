#!/usr/bin/env bash
# Fixed eval splits (val_frozen, test_hidden) from the active validation shards; see scripts/make_splits.py.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python scripts/make_splits.py
