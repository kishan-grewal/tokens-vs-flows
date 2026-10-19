"""Datasets over catk's preprocessed scenario pickles (data/cache/SMART)."""
import pickle
import resource
from pathlib import Path
from typing import Iterator, List, Optional

import torch
from torch.utils.data import Sampler
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from tokens_vs_flows import CACHE, SPLITS
from src.smart.datamodules.target_builder import WaymoTargetBuilderTrain, WaymoTargetBuilderVal

# A collated PyG batch holds many small tensors; DataLoader workers share each through a file descriptor, which
# exhausts a 1024 soft limit ("received 0 items of ancdata"). Raise the soft limit to the hard limit.
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
if _soft < _hard:
    resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))


class ScenarioDataset(Dataset):
    """catk scenario pickles, from a directory or an explicit list of scenario ids."""

    def __init__(self, raw_dir: Path, scenario_ids: Optional[List[str]] = None, train: bool = False,
                 train_max_num: int = 32) -> None:
        raw_dir = Path(raw_dir)
        if scenario_ids is None:
            self.paths = sorted(p.as_posix() for p in raw_dir.glob("*.pkl"))
        else:
            self.paths = [(raw_dir / f"{s}.pkl").as_posix() for s in scenario_ids]
        transform = WaymoTargetBuilderTrain(train_max_num) if train else WaymoTargetBuilderVal()
        super().__init__(transform=transform)

    def len(self) -> int:
        return len(self.paths)

    def get(self, idx: int):
        with open(self.paths[idx], "rb") as f:
            return pickle.load(f)


def read_split(name: str) -> List[str]:
    return [l.strip() for l in (SPLITS / f"{name}.txt").read_text().splitlines() if l.strip()]


class EpochSampler(Sampler):
    """Seeded permutation per epoch that can start mid-epoch, so a resumed run sees the same batches."""

    def __init__(self, n: int, seed: int, epoch: int = 0, start: int = 0) -> None:
        self.n, self.seed, self.epoch, self.start = n, seed, epoch, start

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator().manual_seed(self.seed * 100003 + self.epoch)
        perm = torch.randperm(self.n, generator=g).tolist()
        return iter(perm[self.start:])

    def __len__(self) -> int:
        return self.n - self.start


def train_loader(batch_size: int, seed: int, epoch: int, start_sample: int, num_workers: int,
                 raw_dir: Path = CACHE / "training") -> DataLoader:
    ds = ScenarioDataset(raw_dir, train=True)
    sampler = EpochSampler(len(ds), seed, epoch, start_sample)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                      pin_memory=True, drop_last=True, persistent_workers=False,
                      prefetch_factor=4 if num_workers > 0 else None)


def eval_loader(scenario_ids: List[str], batch_size: int, num_workers: int,
                raw_dir: Path = CACHE / "validation") -> DataLoader:
    ds = ScenarioDataset(raw_dir, scenario_ids=scenario_ids)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=True, drop_last=False)
