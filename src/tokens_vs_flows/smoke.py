"""Smoke test: for each head, 1 training step + 1 closed-loop rollout + scoring on 2 val_frozen scenarios.

Usage: python -m tokens_vs_flows.smoke   (must finish in < 5 min; exits non-zero on any failure)
"""
import json
import subprocess
import sys
import time

import torch

from tokens_vs_flows import ROOT
from tokens_vs_flows.data import read_split, train_loader
from tokens_vs_flows.model import ModelConfig, SimAgent
from tokens_vs_flows.rollout import write_rollouts
from tokens_vs_flows.train import save, setup_cuda


def main() -> int:
    t0 = time.time()
    setup_cuda("fp32")
    ids = read_split("val_frozen")[:2]
    batch = next(iter(train_loader(batch_size=2, seed=0, epoch=0, start_sample=0, num_workers=2))).cuda()
    for head in ("token", "flow"):
        torch.manual_seed(0)
        run = ROOT / "runs" / f"smoke_{head}"
        (run / "ckpt").mkdir(parents=True, exist_ok=True)
        model = SimAgent(ModelConfig(head=head)).cuda().train()
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
        loss = model.loss(batch.clone())["loss"]
        loss.backward()
        opt.step()
        assert torch.isfinite(loss), f"{head}: non-finite loss"
        save(run / "ckpt" / "final.pt", model, None, {"step": 1})
        (run / "config.json").write_text(json.dumps({"head": head, "seed": 0}) + "\n")
        write_rollouts(model.eval(), ids, run / "rollouts", n_rollouts=1, batch_size=2, num_workers=0, log_every=0)
        r = subprocess.run([sys.executable, str(ROOT / "eval" / "score.py"), "--run", str(run), "--split", "val_frozen",
                            "--max-scenarios", "2", "--n-rollouts", "1"], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-4000:])
            return 1
        m = json.loads((run / "metrics.json").read_text())
        print(f"{head}: loss {loss.item():.3f} realism {m['realism']:.4f} minADE {m['minADE']:.2f} "
              f"({time.time() - t0:.0f} s elapsed)", flush=True)
    elapsed = time.time() - t0
    print(f"smoke test {'PASS' if elapsed < 300 else 'FAIL (too slow)'} in {elapsed:.0f} s")
    return 0 if elapsed < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())
