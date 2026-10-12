"""Fixed-composition batch sampler (Section 4.3.2)."""
from __future__ import annotations

import random
from typing import Dict, Iterator, List, Sequence, Tuple

from .paths import is_synthetic


class QuotaBatchSampler:
    """Batches with a fixed number of samples from each named group.

    The ``normal`` group is drawn without replacement and sets the epoch
    length (``len(normal) // quota``); the other groups cycle, reshuffling
    when exhausted. Batch order is shuffled with the same seeded RNG.
    """

    def __init__(self, groups: Dict[str, Sequence[int]], quotas: Dict[str, int], seed: int):
        self.groups = {k: list(v) for k, v in groups.items()}
        self.quotas = dict(quotas)
        if set(self.groups) != set(self.quotas):
            raise ValueError("groups and quotas must have the same keys")
        for k, v in self.groups.items():
            if not v or self.quotas[k] <= 0:
                raise ValueError(f"empty group or non-positive quota: {k!r}")
        self.rng = random.Random(seed)
        self.num_batches = len(self.groups["normal"]) // self.quotas["normal"]
        for k in self.groups:
            self.rng.shuffle(self.groups[k])
        self.ptrs = {k: 0 for k in self.groups}

    def __len__(self) -> int:
        return self.num_batches

    def _draw(self, k: str, m: int) -> List[int]:
        idxs, out = self.groups[k], []
        while len(out) < m:
            take = min(m - len(out), len(idxs) - self.ptrs[k])
            out.extend(idxs[self.ptrs[k]:self.ptrs[k] + take])
            self.ptrs[k] += take
            if self.ptrs[k] >= len(idxs):
                self.rng.shuffle(idxs)
                self.ptrs[k] = 0
        return out

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.num_batches):
            batch: List[int] = []
            for k in sorted(self.quotas):
                batch.extend(self._draw(k, self.quotas[k]))
            self.rng.shuffle(batch)
            yield batch


def sample_groups(samples: Sequence[Tuple[str, int]], quotas: Dict[str, int]) -> Dict[str, List[int]]:
    """Index groups matching the quota keys: normal/defect, or normal/real/synthetic under VF."""
    if "synth_defect" in quotas:
        groups: Dict[str, List[int]] = {"normal": [], "real_defect": [], "synth_defect": []}
        for i, (path, label) in enumerate(samples):
            if label == 0:
                groups["normal"].append(i)
            elif is_synthetic(path):
                groups["synth_defect"].append(i)
            else:
                groups["real_defect"].append(i)
        return groups
    return {"normal": [i for i, (_, y) in enumerate(samples) if y == 0],
            "defect": [i for i, (_, y) in enumerate(samples) if y == 1]}
