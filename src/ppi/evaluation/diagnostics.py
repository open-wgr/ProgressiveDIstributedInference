"""Dataset-agnostic diagnostics for progressive partitioned inference.

The subset accuracy table alone cannot distinguish a working mechanism from
several distinct failure modes that all produce plausible-looking numbers:

  * a partition collapsed to a single point (every pair scores identically,
    so the threshold search is arbitrary and accuracy sits at chance);
  * partitions that are redundant copies of P0 (adding them returns their
    mean, never more);
  * genuine complementary signal that the combination metric discards;
  * a task with no headroom above P0 for anything to progress into.

Each needs a different fix, so each gets its own measurement here. Nothing in
this module is dataset-specific: it takes pair-aligned partition embeddings and
a same/different vector, so CIFAR-100, LFW, CASIA and any future testbed share
one implementation and one output format.

Typical use::

    report = compute_diagnostics(emb_a, emb_b, issame, trunk_a=fa, trunk_b=fb)
    print(report.render())
    logger.log_dict(report.to_dict())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ppi.evaluation.metrics import compute_pair_accuracy, compute_tar_at_far

__all__ = [
    "CollapseStat",
    "SubsetMetric",
    "DiagnosticsReport",
    "compute_diagnostics",
    "partition_subsets",
    "subset_key",
]

# A partition whose pairwise-cosine spread is below this is degenerate: every
# pair scores the same, so no threshold separates them.
_COLLAPSE_STD = 0.01
_LOW_SPREAD_STD = 0.05


class _Combiner(Protocol):
    def combine(self, partition_embeddings: list[Tensor | None], **kwargs) -> Tensor: ...


# ---------------------------------------------------------------------------
# Subset enumeration
# ---------------------------------------------------------------------------

def subset_key(sub: tuple[int, ...]) -> str:
    return "P" + "".join(str(i) for i in sub)


def partition_subsets(num_partitions: int, anchored: bool = True) -> list[tuple[int, ...]]:
    """Non-empty partition subsets, ordered by size then lexicographically.

    anchored=True  -> only subsets containing P0. These are the deployable
                      configurations (P0 is always present on-device).
    anchored=False -> every non-empty subset. P1-alone and P12 are not
                      deployable but are diagnostic: they isolate whether a
                      partition has usable metric structure on its own.
    """
    out: list[tuple[int, ...]] = []
    for size in range(1, num_partitions + 1):
        for combo in combinations(range(num_partitions), size):
            if anchored and 0 not in combo:
                continue
            out.append(combo)
    return out


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class CollapseStat:
    partition: int
    mean_cos: float
    std_cos: float

    @property
    def status(self) -> str:
        if self.std_cos < _COLLAPSE_STD:
            return "COLLAPSED"
        if self.std_cos < _LOW_SPREAD_STD:
            return "low spread"
        return "ok"

    @property
    def collapsed(self) -> bool:
        return self.std_cos < _COLLAPSE_STD


@dataclass
class SubsetMetric:
    name: str
    pair_acc: float
    pair_std: float
    tar_at_far: float


@dataclass
class DiagnosticsReport:
    """Everything measured in one eval pass. Render for humans, dict for logs."""

    num_partitions: int
    collapse: list[CollapseStat]
    subsets: list[SubsetMetric]
    trunk: SubsetMetric | None = None

    # Complementarity (None when num_partitions == 1)
    per_partition_acc: dict[int, float] = field(default_factory=dict)
    oracle_acc: float | None = None
    rescue_rate: dict[int, float] = field(default_factory=dict)
    error_corr: dict[int, float] = field(default_factory=dict)
    strata: list[tuple[str, dict[int, float]]] = field(default_factory=list)

    # Realisable score-level combiner
    combiner_acc: float | None = None
    combiner_std: float | None = None
    combiner_gain: float | None = None
    combiner_2se: float | None = None

    far_target: float = 1e-3

    # -- derived -----------------------------------------------------------

    @property
    def p0_acc(self) -> float:
        return next(s.pair_acc for s in self.subsets if s.name == "P0")

    @property
    def full_key(self) -> str:
        return subset_key(tuple(range(self.num_partitions)))

    @property
    def collapsed_partitions(self) -> list[int]:
        return [c.partition for c in self.collapse if c.collapsed]

    @property
    def oracle_gain(self) -> float | None:
        if self.oracle_acc is None or not self.per_partition_acc:
            return None
        return self.oracle_acc - self.per_partition_acc[0]

    @property
    def specialised(self) -> bool:
        """True if any P_k beats P0 where P0 is least confident (stratum Q1)."""
        if not self.strata:
            return False
        _, q1 = self.strata[0]
        return any(q1.get(k, 0.0) > q1.get(0, 1.0) for k in range(1, self.num_partitions))

    @property
    def verdict(self) -> str:
        if self.collapsed_partitions:
            return (
                f"COLLAPSED: P{self.collapsed_partitions} map every input to one point. "
                "Their rows are meaningless and any apparent gain is noise."
            )
        if self.combiner_gain is None or self.combiner_2se is None:
            return "Single partition — no complementarity to assess."
        if self.combiner_gain > self.combiner_2se:
            return (
                "PROGRESSIVE IMPROVEMENT IS ACHIEVABLE. A deployable combiner beats "
                "P0 out-of-sample; the default combination is discarding signal."
            )
        if (self.oracle_gain or 0.0) >= 0.01 and not self.specialised:
            return (
                "Oracle gain is NOT realisable. No partition beats P0 in any stratum, "
                "so the union is noise-driven disagreement rather than complementary "
                "signal. The fix belongs in training, not the combiner."
            )
        return (
            "No exploitable complementarity at score level. If a partition beats P0 "
            "in Q1, an embedding-level combiner may still help; otherwise the "
            "partitions are redundant."
        )

    # -- output ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "collapse": {
                f"P{c.partition}": {"mean_cos": c.mean_cos, "std_cos": c.std_cos,
                                    "status": c.status}
                for c in self.collapse
            },
            "subsets": {
                s.name: {"pair_accuracy": s.pair_acc, "pair_std": s.pair_std,
                         f"tar_at_far_{self.far_target:g}": s.tar_at_far}
                for s in self.subsets
            },
            "verdict": self.verdict,
        }
        if self.trunk is not None:
            d["trunk"] = {"pair_accuracy": self.trunk.pair_acc,
                          "pair_std": self.trunk.pair_std,
                          "headroom_over_p0": self.trunk.pair_acc - self.p0_acc}
        if self.oracle_acc is not None:
            d["complementarity"] = {
                "oracle_acc": self.oracle_acc,
                "oracle_gain": self.oracle_gain,
                "per_partition_acc": dict(self.per_partition_acc),
                "rescue_rate": dict(self.rescue_rate),
                "error_correlation": dict(self.error_corr),
                "specialised": self.specialised,
            }
        if self.combiner_acc is not None:
            d["realisable_combiner"] = {
                "acc": self.combiner_acc, "std": self.combiner_std,
                "gain_over_p0": self.combiner_gain, "two_se": self.combiner_2se,
                "significant": bool((self.combiner_gain or 0) > (self.combiner_2se or 0)),
            }
        return d

    def render(self, title: str = "") -> str:
        L: list[str] = []
        if title:
            L += ["", f"  {title}"]

        L += ["", "  Per-partition embedding spread (collapse check):",
              f"  {'Partition':<10}  {'mean cos':>10}  {'std cos':>10}  {'status':>12}",
              "  " + "-" * 48]
        for c in self.collapse:
            L.append(f"  {'P' + str(c.partition):<10}  {c.mean_cos:>10.4f}  "
                     f"{c.std_cos:>10.4f}  {c.status:>12}")
        if self.collapsed_partitions:
            L += ["", f"  WARNING: P{self.collapsed_partitions} collapsed to a point. Rows below",
                  "           are meaningless and any 'improvement' is noise."]

        tar_label = f"TAR@{self.far_target:g}"
        L += ["", "  Verification by partition subset:",
              f"  {'Config':<10}  {'pair_acc':>10}  {'pair_std':>10}  {tar_label:>10}",
              "  " + "-" * 48]
        for s in self.subsets:
            L.append(f"  {s.name:<10}  {s.pair_acc:>10.4f}  {s.pair_std:>10.4f}  "
                     f"{s.tar_at_far:>10.4f}")
        if self.trunk is not None:
            L += ["  " + "-" * 48,
                  f"  {'TRUNK':<10}  {self.trunk.pair_acc:>10.4f}  "
                  f"{self.trunk.pair_std:>10.4f}  {self.trunk.tar_at_far:>10.4f}   (reference)",
                  "",
                  f"  Headroom above P0: {self.trunk.pair_acc - self.p0_acc:+.4f}"]

        if self.oracle_acc is not None:
            L += ["", "  Complementarity (do partitions fail on different pairs?):",
                  f"  {'':<24}{'acc':>8}"]
            for k in sorted(self.per_partition_acc):
                L.append(f"  {'P' + str(k) + ' alone':<24}{self.per_partition_acc[k]:>8.4f}")
            L.append(f"  {'ORACLE (any partition)':<24}{self.oracle_acc:>8.4f}   "
                     f"<- ceiling for ANY combiner")
            for k in sorted(self.rescue_rate):
                L.append(f"    P{k}: fixes {self.rescue_rate[k] * 100:5.1f}% of P0's errors "
                         f"| error correlation with P0 = {self.error_corr[k]:+.3f}")

        if self.strata:
            hdr = f"  {'stratum':<20}" + "".join(
                f"{'P' + str(i):>9}" for i in range(self.num_partitions))
            L += ["", "  Specialisation profile (accuracy by P0 decision margin):", hdr,
                  "  " + "-" * (len(hdr) - 2)]
            for label, accs in self.strata:
                L.append(f"  {label:<20}" + "".join(
                    f"{accs[i]:>9.4f}" for i in range(self.num_partitions)))
            L.append("    ^ complementarity looks like P_k > P0 in Q1 even if P_k < P0 overall.")

        if self.oracle_gain is not None:
            L += ["", f"  Oracle gain over P0: {self.oracle_gain:+.4f}  "
                      f"(upper bound, noise-inflated)"]
        if self.combiner_acc is not None:
            L += ["", "  Realisable combiner (logistic regression on partition scores,",
                  f"  {10}-fold out-of-sample): {self.combiner_acc:.4f} +/- {self.combiner_std:.4f}",
                  f"  vs P0 {self.p0_acc:.4f}  ->  {self.combiner_gain:+.4f}  "
                  f"(2*SE = {self.combiner_2se:.4f})"]

        L += ["", f"  -> {self.verdict}", ""]
        return "\n".join(L)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _pair_sims(emb_a: Tensor, emb_b: Tensor) -> np.ndarray:
    a = F.normalize(emb_a.float(), dim=1, eps=1e-12).cpu().numpy()
    b = F.normalize(emb_b.float(), dim=1, eps=1e-12).cpu().numpy()
    return (a * b).sum(axis=1)


def _best_threshold_mask(sims: np.ndarray, issame: np.ndarray) -> tuple[np.ndarray, float]:
    """Correctness mask at the in-sample optimal threshold, and that threshold.

    In-sample fitting makes this optimistic — deliberately so, since it feeds
    the oracle, which is meant to be an upper bound.
    """
    best, thr = -1.0, 0.0
    for t in np.linspace(sims.min(), sims.max(), 1000):
        acc = ((sims >= t) == issame).mean()
        if acc > best:
            best, thr = acc, t
    return (sims >= thr) == issame, thr


def _score_metrics(emb_a: Tensor, emb_b: Tensor, issame: np.ndarray,
                   far_target: float) -> tuple[float, float, float]:
    a = F.normalize(emb_a.float(), dim=1, eps=1e-12).cpu().numpy()
    b = F.normalize(emb_b.float(), dim=1, eps=1e-12).cpu().numpy()
    acc, std = compute_pair_accuracy(a, b, issame)
    sims = (a * b).sum(axis=1)
    tar = compute_tar_at_far(sims[issame], sims[~issame], far_target=far_target)
    return acc, std, tar


def _logistic_kfold(X: np.ndarray, y: np.ndarray, n_folds: int = 10,
                    seed: int = 0) -> tuple[float, float]:
    """Out-of-sample accuracy of logistic regression on per-partition scores.

    The simplest combiner that could actually be deployed. It sees exactly the
    information any score-level combination has, so if it cannot beat P0 the
    oracle gain is not real structure.
    """
    torch.manual_seed(seed)
    n = X.shape[0]
    fold = max(1, n // n_folds)
    accs: list[float] = []
    for f in range(n_folds):
        va = np.zeros(n, dtype=bool)
        va[f * fold:(f + 1) * fold] = True
        if not va.any() or va.all():
            continue
        tr = ~va
        xt = torch.tensor(X[tr], dtype=torch.float32)
        yt = torch.tensor(y[tr], dtype=torch.float32)
        xv = torch.tensor(X[va], dtype=torch.float32)
        yv = torch.tensor(y[va], dtype=torch.float32)
        mu, sd = xt.mean(0, keepdim=True), xt.std(0, keepdim=True).clamp_min(1e-6)
        xt, xv = (xt - mu) / sd, (xv - mu) / sd
        model = torch.nn.Linear(X.shape[1], 1)
        opt = torch.optim.LBFGS(model.parameters(), max_iter=200)

        def _closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(xt).squeeze(1), yt)
            loss.backward()
            return loss

        opt.step(_closure)
        with torch.no_grad():
            accs.append(((model(xv).squeeze(1) > 0).float() == yv).float().mean().item())
    return float(np.mean(accs)), float(np.std(accs))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def compute_diagnostics(
    emb_a: Tensor,
    emb_b: Tensor,
    issame: np.ndarray,
    *,
    raw_a: Tensor | None = None,
    raw_b: Tensor | None = None,
    trunk_a: Tensor | None = None,
    trunk_b: Tensor | None = None,
    combiner: _Combiner | None = None,
    anchored: bool = True,
    far_target: float = 1e-3,
    n_folds: int = 10,
    collapse_probe: int = 512,
) -> DiagnosticsReport:
    """Run the full diagnostic battery on one set of verification pairs.

    Parameters
    ----------
    emb_a, emb_b:
        (N, P, K) pair-aligned per-partition embeddings. Callers holding
        deduplicated embeddings should index them first.
    issame:
        (N,) bool — True for genuine pairs.
    raw_a, raw_b:
        Optional (N, P, K) pre-normalisation outputs, forwarded to combiners
        whose confidence signal depends on embedding magnitude.
    trunk_a, trunk_b:
        Optional (N, D) shared trunk features. Adds the TRUNK reference row,
        which answers whether the shared representation beats its own
        projection — i.e. whether headroom exists at all.
    combiner:
        Object exposing ``.combine(list_of_partition_embeddings)``. Defaults to
        CosineConcat.
    anchored:
        Restrict to P0-anchored (deployable) subsets. False adds P1-alone and
        similar, which are diagnostic only.
    """
    from ppi.boosting.combination import ConfidenceWeighted, CosineConcat, LearnedCombiner

    if combiner is None:
        combiner = CosineConcat()

    n_pairs, num_partitions, _ = emb_a.shape
    issame = np.asarray(issame).astype(bool)

    # --- collapse ---------------------------------------------------------
    probe = min(collapse_probe, n_pairs)
    eye = torch.eye(probe, dtype=torch.bool)
    collapse: list[CollapseStat] = []
    for i in range(num_partitions):
        e = F.normalize(emb_a[:probe, i, :].float(), dim=1, eps=1e-12)
        off = (e @ e.T)[~eye]
        collapse.append(CollapseStat(i, float(off.mean()), float(off.std())))

    # --- subset table -----------------------------------------------------
    def _assemble(src_norm: Tensor, src_raw: Tensor | None, active: set[int]) -> Tensor:
        parts = [src_norm[:, i, :].float() if i in active else None
                 for i in range(num_partitions)]
        if isinstance(combiner, ConfidenceWeighted):
            raws = [src_raw[:, i, :].float() if (src_raw is not None and i in active) else None
                    for i in range(num_partitions)]
            return combiner.combine(parts, raw_embeddings=raws)
        if isinstance(combiner, LearnedCombiner):
            mask = torch.zeros(src_norm.shape[0], num_partitions)
            for i in active:
                mask[:, i] = 1.0
            return combiner.combine(parts, mask=mask)
        return combiner.combine(parts)

    subsets: list[SubsetMetric] = []
    for combo in partition_subsets(num_partitions, anchored):
        active = set(combo)
        acc, std, tar = _score_metrics(
            _assemble(emb_a, raw_a, active), _assemble(emb_b, raw_b, active),
            issame, far_target,
        )
        subsets.append(SubsetMetric(subset_key(combo), acc, std, tar))

    report = DiagnosticsReport(
        num_partitions=num_partitions, collapse=collapse, subsets=subsets,
        far_target=far_target,
    )

    # --- trunk reference --------------------------------------------------
    if trunk_a is not None and trunk_b is not None:
        acc, std, tar = _score_metrics(trunk_a, trunk_b, issame, far_target)
        report.trunk = SubsetMetric("TRUNK", acc, std, tar)

    if num_partitions < 2:
        return report

    # --- complementarity --------------------------------------------------
    sims = [_pair_sims(emb_a[:, i, :], emb_b[:, i, :]) for i in range(num_partitions)]
    masks, thresholds = zip(*(_best_threshold_mask(s, issame) for s in sims))

    report.per_partition_acc = {i: float(m.mean()) for i, m in enumerate(masks)}
    union = np.zeros_like(masks[0])
    for m in masks:
        union |= m
    report.oracle_acc = float(union.mean())

    base = masks[0]
    for i in range(1, num_partitions):
        wrong = ~base
        report.rescue_rate[i] = float((wrong & masks[i]).sum() / max(wrong.sum(), 1))
        report.error_corr[i] = float(
            np.corrcoef(base.astype(float), masks[i].astype(float))[0, 1])

    # --- specialisation profile -------------------------------------------
    margin0 = np.abs(sims[0] - thresholds[0])
    q = np.quantile(margin0, [0.25, 0.5, 0.75])
    for label, sel in [
        ("Q1 (P0 least sure)", margin0 <= q[0]),
        ("Q2", (margin0 > q[0]) & (margin0 <= q[1])),
        ("Q3", (margin0 > q[1]) & (margin0 <= q[2])),
        ("Q4 (P0 most sure)", margin0 > q[2]),
    ]:
        report.strata.append(
            (label, {i: float(masks[i][sel].mean()) for i in range(num_partitions)}))

    # --- realisable combiner ----------------------------------------------
    acc, std = _logistic_kfold(np.stack(sims, axis=1), issame.astype(np.float32), n_folds)
    p0_std = next(s.pair_std for s in subsets if s.name == "P0")
    report.combiner_acc, report.combiner_std = acc, std
    report.combiner_gain = acc - report.p0_acc
    report.combiner_2se = 2 * max(p0_std, std) / (n_folds ** 0.5)

    return report
