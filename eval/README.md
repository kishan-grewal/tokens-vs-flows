# eval/
- `score.py`: WOSAC realism (TrajTok fast metric, 2025 version) plus the extra metrics in docs/DESIGN.md. Writes `runs/<id>/metrics.json`.
- `splits/val_frozen.txt`: 1000 dev scenarios (validation shards 0-3).
- `splits/test_hidden.txt`: 1000 held-out scenarios (validation shards 4-7), scored once at the end.

The scoring code and splits are frozen once the sanity checks pass (ground-truth replay ceiling, constant-velocity > random, fast vs official metric within 1e-3 on 20 scenarios). They are not edited afterwards.
