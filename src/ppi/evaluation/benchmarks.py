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
    """CFP Frontal-Profile benchmark (7,000 pairs, 10 folds).

    Expected layout (from cfp-dataset.com):

        {root}/
        ├── Data/Images/{id:03d}/frontal/{idx:02d}.jpg   (10 per subject, 500 subjects)
        ├── Data/Images/{id:03d}/profile/{idx:02d}.jpg   (4 per subject)
        └── Protocol/Frontal-Profile/
            ├── split1/
            │   ├── same.txt   # genuine pairs
            │   └── diff.txt   # impostor pairs
            ...
            └── split10/

    same.txt — comma-separated, 1-indexed:
        person_id,frontal_idx,profile_idx
    diff.txt — comma-separated, 1-indexed:
        person_id1,frontal_idx1,person_id2,profile_idx2
    """

    def load_pairs(self) -> tuple[list[str], list[str], np.ndarray]:
        protocol_dir = self.root / "Protocol" / "Frontal-Profile"
        if not protocol_dir.exists():
            raise FileNotFoundError(
                f"CFP-FP protocol directory not found: {protocol_dir}\n"
                "Expected layout: {root}/Protocol/Frontal-Profile/split1/ ... split10/"
            )

        paths1: list[str] = []
        paths2: list[str] = []
        issame: list[bool] = []

        for split_idx in range(1, 11):
            split_dir = protocol_dir / f"split{split_idx}"

            with open(split_dir / "same.txt") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) != 3:
                        continue
                    pid, fidx, pidx = parts
                    paths1.append(f"Data/Images/{int(pid):03d}/frontal/{int(fidx):02d}.jpg")
                    paths2.append(f"Data/Images/{int(pid):03d}/profile/{int(pidx):02d}.jpg")
                    issame.append(True)

            with open(split_dir / "diff.txt") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) != 4:
                        continue
                    pid1, fidx1, pid2, pidx2 = parts
                    paths1.append(f"Data/Images/{int(pid1):03d}/frontal/{int(fidx1):02d}.jpg")
                    paths2.append(f"Data/Images/{int(pid2):03d}/profile/{int(pidx2):02d}.jpg")
                    issame.append(False)

        return paths1, paths2, np.array(issame)


class AgeDB30Benchmark(PairBenchmark):
    """AgeDB-30 benchmark (6,000 pairs, 10 folds, age gap ≤ 30 years).

    Expected layout:

        {root}/
        ├── pairs.txt    # Protocol file (space-separated)
        └── *.jpg        # All images in a flat directory, named {name}_{age}.jpg

    pairs.txt — space-separated, no header:
        img1_filename img2_filename 1   # genuine pair
        img3_filename img4_filename 0   # impostor pair

    Filenames in pairs.txt should match the flat image files directly
    (e.g. "Aaron_Eckhart_36.jpg Aaron_Eckhart_54.jpg 1").
    """

    def load_pairs(self) -> tuple[list[str], list[str], np.ndarray]:
        pairs_path = self.root / "pairs.txt"
        if not pairs_path.exists():
            raise FileNotFoundError(
                f"AgeDB-30 pairs file not found: {pairs_path}\n"
                "Expected: space-separated lines of 'img1 img2 1|0'"
            )

        paths1: list[str] = []
        paths2: list[str] = []
        issame: list[bool] = []

        with open(pairs_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                img1, img2, label = parts
                paths1.append(img1)
                paths2.append(img2)
                issame.append(int(label) == 1)

        return paths1, paths2, np.array(issame)
