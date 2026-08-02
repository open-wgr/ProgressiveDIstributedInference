"""Hyperparameter sweep launcher for Direction 2 boosting.

Runs train_boosting.py for each combination of sweep axes, parses pair_acc
from stdout, checks for monotonic improvement (P0 < P01 < P012), and prints
a ranked summary table.

Default sweep axes match the Direction 2 plan Phase 1 go/no-go criteria:
  --mining-strategy  {topk, band}
  --loss             {triplet, contrastive, arcface_reweighted, arcface_margin,
                      sub_center_arcface}

Usage:
  # Phase 1 default sweep (CIFAR-100, all mining × loss combinations)
  python scripts/sweep.py --config configs/direction_2_base.yaml --dataset cifar100

  # Custom axes
  python scripts/sweep.py --config configs/direction_2_base.yaml --dataset cifar100 \\
      --axes mining_strategy=topk,band loss=triplet,contrastive

  # Pass extra fixed flags to every run
  python scripts/sweep.py --config configs/direction_2_base.yaml --dataset cifar100 \\
      --fixed --backbone-state frozen --epochs-phase0 10 --epochs-per-phase 10
"""

from __future__ import annotations

import argparse
import itertools
import re
import subprocess
import sys
from itertools import combinations
from pathlib import Path


# ---------------------------------------------------------------------------
# Default Phase 1 sweep grid (Direction 2 plan §Task 9 Phase 1)
# ---------------------------------------------------------------------------

DEFAULT_AXES: dict[str, list[str]] = {
    "mining-strategy": ["topk", "band"],
    "loss": ["triplet", "contrastive", "arcface_reweighted", "arcface_margin",
             "sub_center_arcface"],
}

# Regex to extract pair_acc lines from train_boosting output:
#   "  P012        0.7341      0.1234"
_PAIR_ACC_RE = re.compile(r"^\s+(P[\d]+)\s+([\d.]+)\s+([\d.]+)\s*$")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_pair_accs(output: str) -> dict[str, float]:
    """Extract {config_name: pair_acc} from train_boosting stdout."""
    results: dict[str, float] = {}
    in_table = False
    for line in output.splitlines():
        if "CIFAR-100 Verification" in line or "pair_acc" in line:
            in_table = True
            continue
        if not in_table:
            continue
        m = _PAIR_ACC_RE.match(line)
        if m:
            results[m.group(1)] = float(m.group(2))
    return results


def _p0_anchored_subsets(num_partitions: int) -> list[tuple[int, ...]]:
    """All P0-anchored subsets, ordered by size then lex.

    For N=3: [(0,), (0,1), (0,2), (0,1,2)] → P0, P01, P02, P012.
    """
    optional = list(range(1, num_partitions))
    subsets: list[tuple[int, ...]] = []
    for r in range(num_partitions):
        for extra in combinations(optional, r):
            subsets.append(tuple(sorted((0,) + extra)))
    return subsets


def _subset_key(sub: tuple[int, ...]) -> str:
    return "P" + "".join(str(i) for i in sub)


def is_monotone(accs: dict[str, float], num_partitions: int) -> bool:
    """Strict lattice-monotone improvement across P0-anchored subsets.

    Requires accs[B] > accs[A] for every cover pair A ⊂ B (B = A ∪ {i} for
    some i ∉ A). For N=3 this is the four checks P0<P01, P0<P02, P01<P012,
    P02<P012 — catches the P02 > P012 case the old prefix-only check missed.
    """
    subsets = _p0_anchored_subsets(num_partitions)
    keys = [_subset_key(s) for s in subsets]
    if not all(k in accs for k in keys):
        return False
    subset_set = {frozenset(s) for s in subsets}
    for a in subsets:
        a_set = frozenset(a)
        for i in range(num_partitions):
            if i in a_set:
                continue
            b_set = a_set | {i}
            if b_set not in subset_set:
                continue
            b = tuple(sorted(b_set))
            if not (accs[_subset_key(b)] > accs[_subset_key(a)]):
                return False
    return True


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def build_grid(axes_spec: list[str] | None) -> dict[str, list[str]]:
    """Parse --axes NAME=v1,v2 ... into a grid dict, or return defaults."""
    if not axes_spec:
        return DEFAULT_AXES
    grid: dict[str, list[str]] = {}
    for spec in axes_spec:
        name, _, values = spec.partition("=")
        grid[name.strip()] = [v.strip() for v in values.split(",")]
    return grid


def combo_run_name(combo: dict[str, str], prefix: str) -> str:
    """Unique run name encoding every swept axis.

    train_boosting.py's default run name is derived only from
    dataset/backbone_state/loss, so any sweep over another axis (K,
    mining-strategy, ...) collides and silently overwrites checkpoints in
    checkpoints/boosting/<run_name>/. Naming each run after its full
    combination keeps every checkpoint recoverable.
    """
    parts = [prefix] if prefix else []
    for name, value in combo.items():
        parts.append(f"{name.replace('-', '')}-{value}")
    return "_".join(parts)


def run_combination(
    base_cmd: list[str],
    combo: dict[str, str],
    run_idx: int,
    total: int,
    run_prefix: str = "sweep",
) -> tuple[dict[str, float], str, int]:
    """Run one combination. Returns (pair_accs, stdout, returncode)."""
    flags: list[str] = []
    for name, value in combo.items():
        flags += [f"--{name}", value]
    # Force a per-combination run name so checkpoints never collide.
    flags += ["--run-name", combo_run_name(combo, run_prefix)]

    cmd = base_cmd + flags
    label = " ".join(f"--{k} {v}" for k, v in combo.items())
    print(f"\n[sweep {run_idx}/{total}] {label}", flush=True)
    print(f"  cmd: {' '.join(cmd)}", flush=True)

    result = subprocess.run(cmd, capture_output=False, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = result.stdout or ""
    # Mirror to terminal
    for line in output.splitlines():
        print(f"  | {line}")
    if result.returncode != 0:
        print(f"  [sweep] run exited with code {result.returncode}", flush=True)

    pair_accs = parse_pair_accs(output)
    return pair_accs, output, result.returncode


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(
    results: list[tuple[dict[str, str], dict[str, float], bool]],
    num_partitions: int,
) -> None:
    """Print ranked results table. Monotone-improving configs listed first."""
    full_key = "P" + "".join(str(i) for i in range(num_partitions))

    print(f"\n{'=' * 90}")
    print("  Sweep Summary")
    print(f"{'=' * 90}")

    col_names = [_subset_key(s) for s in _p0_anchored_subsets(num_partitions)]

    header = f"  {'monotone':<9}  {full_key + '_acc':<12}"
    for k in col_names:
        header += f"  {k:<8}"
    header += "  config"
    print(header)
    print("  " + "-" * (len(header) - 2))

    def sort_key(row: tuple) -> tuple:
        combo, accs, mono = row
        full_acc = accs.get(full_key, -1.0)
        return (0 if mono else 1, -full_acc)

    for combo, accs, mono in sorted(results, key=sort_key):
        full_acc = accs.get(full_key, float("nan"))
        mark = "YES" if mono else "no"
        row = f"  {mark:<9}  {full_acc:<12.4f}"
        for k in col_names:
            row += f"  {accs.get(k, float('nan')):<8.4f}"
        row += "  " + " ".join(f"--{k} {v}" for k, v in combo.items())
        print(row)

    monotone_count = sum(1 for _, _, m in results if m)
    print(f"\n  {monotone_count}/{len(results)} configurations achieved monotone improvement.")
    if monotone_count == 0:
        print("  No monotone improvement found. Revisit mining/loss axes before scaling to CASIA.")
    else:
        print("  Go/no-go: PASS. Advance to Phase 2 (CASIA subset) with the top config above.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Base config YAML")
    p.add_argument("--dataset", default=None,
                   help="Dataset override passed to every run (e.g. cifar100)")
    p.add_argument("--axes", nargs="*", metavar="NAME=v1,v2",
                   help="Sweep axes. Default: mining-strategy × loss (Phase 1 grid)")
    p.add_argument("--num-partitions", type=int, default=3)
    p.add_argument("--run-prefix", default="sweep",
                   help="Prefix for per-combination run names (keeps checkpoints distinct)")
    p.add_argument("--fixed", nargs=argparse.REMAINDER, default=[],
                   help="Extra flags forwarded verbatim to every train_boosting.py call")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    grid = build_grid(args.axes)
    keys = list(grid.keys())
    combos = [
        dict(zip(keys, vals))
        for vals in itertools.product(*[grid[k] for k in keys])
    ]
    total = len(combos)
    print(f"[sweep] {total} combinations across axes: {keys}", flush=True)

    # Base command shared by every run
    script = Path(__file__).parent / "train_boosting.py"
    base_cmd = [sys.executable, str(script), "--config", args.config]
    if args.dataset:
        base_cmd += ["--dataset", args.dataset]
    if args.fixed:
        base_cmd += args.fixed

    all_results: list[tuple[dict[str, str], dict[str, float], bool]] = []

    for idx, combo in enumerate(combos, 1):
        pair_accs, _, _ = run_combination(base_cmd, combo, idx, total, args.run_prefix)
        mono = is_monotone(pair_accs, args.num_partitions)
        all_results.append((combo, pair_accs, mono))

        # Early exit signal for interactive use — let the run finish
        if mono:
            print(f"  [sweep] Monotone improvement achieved with: {combo}", flush=True)

    print_summary(all_results, args.num_partitions)


if __name__ == "__main__":
    main()
