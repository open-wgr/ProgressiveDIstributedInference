"""Benchmark dataset loaders for face recognition evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class PairBenchmark:
    """Base class for pair-based verification benchmarks (LFW-style)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def load_pairs(self) -> tuple[list[str], list[str], np.ndarray]:
        """Return (paths1, paths2, issame) for the benchmark protocol.

        Subclasses override this to parse the benchmark-specific pair file.
        """
        raise NotImplementedError


class LFWBenchmark(PairBenchmark):
    """LFW 6,000-pair verification protocol."""

    def load_pairs(self) -> tuple[list[str], list[str], np.ndarray]:
        pairs_path = self.root / "pairs.txt"
        if not pairs_path.exists():
            raise FileNotFoundError(f"LFW pairs file not found: {pairs_path}")

        paths1, paths2, issame = [], [], []
        with open(pairs_path) as f:
            header = f.readline().strip()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    # Same person: name idx1 idx2
                    name, idx1, idx2 = parts
                    paths1.append(f"{name}/{name}_{int(idx1):04d}.jpg")
                    paths2.append(f"{name}/{name}_{int(idx2):04d}.jpg")
                    issame.append(True)
                elif len(parts) == 4:
                    # Different person: name1 idx1 name2 idx2
                    n1, idx1, n2, idx2 = parts
                    paths1.append(f"{n1}/{n1}_{int(idx1):04d}.jpg")
                    paths2.append(f"{n2}/{n2}_{int(idx2):04d}.jpg")
                    issame.append(False)

        return paths1, paths2, np.array(issame)


class CFPFPBenchmark(PairBenchmark):
    """CFP-FP benchmark — placeholder for pair protocol loader."""

    def load_pairs(self) -> tuple[list[str], list[str], np.ndarray]:
        raise NotImplementedError("CFP-FP pair loading not yet implemented")


class AgeDB30Benchmark(PairBenchmark):
    """AgeDB-30 benchmark — placeholder for pair protocol loader."""

    def load_pairs(self) -> tuple[list[str], list[str], np.ndarray]:
        raise NotImplementedError("AgeDB-30 pair loading not yet implemented")
