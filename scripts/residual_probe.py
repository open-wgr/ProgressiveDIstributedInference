"""Residual-information probe: is there discriminative signal left after P0?

Measures how much subclass-discriminative information remains in the shared
trunk once everything linearly predictable from P0's embedding has been
removed. This separates two very different failure modes for Direction 2:

  * probe(F_perp) at chance  -> P0 saturates the discriminative subspace.
    No loss function or mining strategy can manufacture signal that isn't
    there. The testbed is exhausted; lower K or move datasets.

  * probe(F_perp) strong     -> the information IS present and D2's failure
    is in extraction, not in the information budget. The bug is in the
    loss/mining mechanism.

Method
------
Cache trunk features F (N, D) and partition embeddings E_k (N, K) over the
CIFAR-100 train and val splits. Fit the linear map B = pinv(E0_train) @ F_train
on train, then form the residual on both splits:

    F_perp = F - E0 @ B

This strips every direction linearly predictable from P0, regardless of the
partition head's internal architecture. Linear probes (multinomial logistic
regression, fit on train / scored on val) are then trained on F, E0, E_k and
F_perp against the 100-way subclass labels.

Usage
-----
  python scripts/residual_probe.py \\
      --config configs/direction_2_base.yaml \\
      --checkpoint checkpoints/boosting/<run>/phase_2/backbone.pt

  # Probe the phase-0 backbone instead (before boosting touched the trunk)
  python scripts/residual_probe.py \\
      --config configs/direction_2_base.yaml \\
      --checkpoint checkpoints/boosting/<run>/phase_0/backbone.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from ppi.backbones import build_backbone
from ppi.boosting.cifar100_adaptor import (
    CIFAR100_MEAN,
    CIFAR100_STD,
    _SUBCLASS_TO_SUPERCLASS,
)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (trunk_features (N,D), partitions (N,P,K), labels (N,)) on CPU."""
    backbone.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    feats: list[torch.Tensor] = []
    parts: list[torch.Tensor] = []
    labs: list[torch.Tensor] = []

    total = len(loader)
    for i, (images, labels) in enumerate(loader):
        if i % 25 == 0:
            print(f"    batch {i}/{total}", flush=True)
        images = images.to(device, non_blocking=True)
        out = backbone(images)
        feats.append(out["features"].float().cpu())
        parts.append(torch.stack(out["partitions"], dim=1).float().cpu())
        labs.append(labels)

    return torch.cat(feats), torch.cat(parts), torch.cat(labs)


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

def linear_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    num_classes: int,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> float:
    """Fit a multinomial logistic regression; return best val accuracy."""
    torch.manual_seed(seed)

    # Standardise using train statistics — linear probes are scale-sensitive.
    mu = x_train.mean(dim=0, keepdim=True)
    sigma = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    xt = ((x_train - mu) / sigma).to(device)
    xv = ((x_val - mu) / sigma).to(device)
    yt = y_train.to(device)
    yv = y_val.to(device)

    model = nn.Linear(xt.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = xt.shape[0]
    best = 0.0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start: start + batch_size]
            opt.zero_grad()
            loss = F.cross_entropy(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            acc = (model(xv).argmax(dim=1) == yv).float().mean().item()
        best = max(best, acc)

    return best


# ---------------------------------------------------------------------------
# Residual construction
# ---------------------------------------------------------------------------

def build_residual(
    f_train: torch.Tensor,
    e0_train: torch.Tensor,
    f_val: torch.Tensor,
    e0_val: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Regress F on E0 (fit on train), return (F_perp_train, F_perp_val, r2).

    r2 is the fraction of trunk variance linearly explained by P0 — a direct
    measure of how much of the trunk representation P0 already spans.
    """
    # Append a bias column so the regression can fit an intercept.
    def _aug(e: torch.Tensor) -> torch.Tensor:
        return torch.cat([e, torch.ones(e.shape[0], 1, dtype=e.dtype)], dim=1)

    a_train = _aug(e0_train)
    a_val = _aug(e0_val)

    # Solve via normal equations in float64. The Gram matrix is (K+1, K+1)
    # — tiny and numerically stable — which avoids materialising a float64
    # copy of the full (N, D) feature matrix.
    at = a_train.double()
    gram = at.T @ at
    gram += 1e-6 * torch.eye(gram.shape[0], dtype=gram.dtype) * gram.diagonal().mean()
    rhs = at.T @ f_train.double()
    b = torch.linalg.solve(gram, rhs).float()  # (K+1, D)

    resid_train = f_train - a_train @ b
    resid_val = f_val - a_val @ b

    ss_res = resid_train.double().pow(2).sum().item()
    centred = f_train - f_train.mean(dim=0, keepdim=True)
    ss_tot = centred.double().pow(2).sum().item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    return resid_train, resid_val, r2


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_backbone_state(checkpoint_path: Path) -> dict:
    """Accept both D2 checkpoint layouts."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt and "backbone" in ckpt["model_state_dict"]:
        return ckpt["model_state_dict"]["backbone"]
    if "backbone" in ckpt:
        return ckpt["backbone"]
    raise ValueError(
        f"Could not find a backbone state dict in {checkpoint_path}. "
        f"Top-level keys: {sorted(ckpt.keys())}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Config the model was trained with")
    p.add_argument("--checkpoint", required=True, help="Path to a phase backbone.pt")
    p.add_argument("--device", choices=["cuda", "cpu"], default=None)
    p.add_argument("--probe-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0, help="Probe init/shuffle seed")
    p.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[probe] device={device}", flush=True)

    # CIFAR-100 uses 32x32 inputs regardless of what the base config says.
    config.setdefault("data", {})["input_size"] = 32

    backbone = build_backbone(config).to(device)
    backbone.load_state_dict(load_backbone_state(Path(args.checkpoint)))
    print(f"[probe] Loaded backbone from {args.checkpoint}", flush=True)

    # Use the deterministic (val) transform for both splits: we are measuring
    # information content, not training a model, so augmentation adds only noise.
    root = config["data"]["root"]
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    train_ds = CIFAR100(root=root, train=True, download=True, transform=eval_transform)
    val_ds = CIFAR100(root=root, train=False, download=True, transform=eval_transform)

    print("[probe] Extracting train features...", flush=True)
    f_train, parts_train, y_train = extract_features(
        backbone, train_ds, device, args.batch_size, args.num_workers
    )
    print("[probe] Extracting val features...", flush=True)
    f_val, parts_val, y_val = extract_features(
        backbone, val_ds, device, args.batch_size, args.num_workers
    )

    num_partitions = parts_train.shape[1]
    print(
        f"[probe] trunk={tuple(f_train.shape)}  partitions={tuple(parts_train.shape)}",
        flush=True,
    )

    e0_train = parts_train[:, 0, :]
    e0_val = parts_val[:, 0, :]

    print("[probe] Building residual (regressing trunk on P0)...", flush=True)
    fperp_train, fperp_val, r2 = build_residual(f_train, e0_train, f_val, e0_val)
    print(f"[probe] Trunk variance linearly explained by P0: R^2 = {r2:.4f}", flush=True)

    # Subclass (100-way) is the task the verification eval actually scores.
    targets = {
        "trunk_F": (f_train, f_val),
        "P0_only": (e0_train, e0_val),
    }
    for k in range(1, num_partitions):
        targets[f"P{k}_only"] = (parts_train[:, k, :], parts_val[:, k, :])

    # Concatenated P0-anchored subsets — these are the EXACT vectors that
    # cosine_concat scores at eval time (the backbone already L2-normalises
    # each partition). A linear probe on the same vector that beats the P0
    # probe, while cosine verification does not, localises the failure to the
    # similarity metric rather than to the representation.
    from itertools import combinations
    for r in range(1, num_partitions):
        for extra in combinations(range(1, num_partitions), r):
            idxs = (0,) + extra
            name = "concat_P" + "".join(str(i) for i in idxs)
            targets[name] = (
                torch.cat([parts_train[:, i, :] for i in idxs], dim=1),
                torch.cat([parts_val[:, i, :] for i in idxs], dim=1),
            )

    targets["residual_F_perp"] = (fperp_train, fperp_val)

    results: dict[str, float] = {}
    print("\n[probe] Fitting linear probes (100-way subclass)...", flush=True)
    for name, (xt, xv) in targets.items():
        acc = linear_probe(
            xt, y_train, xv, y_val,
            num_classes=100, device=device, epochs=args.probe_epochs,
            seed=args.seed,
        )
        results[name] = acc
        print(f"    {name:<20} dim={xt.shape[1]:<6} val_acc={acc:.4f}", flush=True)

    # Superclass control: how much of P0's capacity went to its actual objective.
    y_train_super = torch.tensor([_SUBCLASS_TO_SUPERCLASS[int(v)] for v in y_train])
    y_val_super = torch.tensor([_SUBCLASS_TO_SUPERCLASS[int(v)] for v in y_val])
    super_acc = linear_probe(
        e0_train, y_train_super, e0_val, y_val_super,
        num_classes=20, device=device, epochs=args.probe_epochs,
        seed=args.seed,
    )
    results["P0_superclass"] = super_acc

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------
    chance_sub = 0.01
    trunk = results["trunk_F"]
    p0 = results["P0_only"]
    resid = results["residual_F_perp"]

    print(f"\n{'=' * 72}")
    print("  Residual-Information Probe")
    print(f"{'=' * 72}")
    print(f"  Trunk variance explained by P0 (R^2):        {r2:.4f}")
    print(f"  Subclass chance accuracy:                    {chance_sub:.4f}")
    print(f"  P0 on its own objective (20-way superclass): {super_acc:.4f}")
    print()
    print(f"  {'representation':<22} {'dim':>6} {'subclass acc':>14}")
    print("  " + "-" * 46)
    for name in targets:
        dim = targets[name][0].shape[1]
        print(f"  {name:<22} {dim:>6} {results[name]:>14.4f}")
    print()

    # Interpretation is a 2x2: is the information present, and did the trained
    # partitions capture it? Reading only the residual conflates two opposite
    # failure modes with opposite fixes.
    retained = (resid - chance_sub) / max(trunk - chance_sub, 1e-9)
    print(f"  Residual retains {retained * 100:.1f}% of the trunk's decodable subclass signal.")

    pk_accs = {k: results[f"P{k}_only"] for k in range(1, num_partitions)}
    best_pk = max(pk_accs.values()) if pk_accs else 0.0
    partitions_beat_p0 = best_pk > p0

    full_key = "concat_P" + "".join(str(i) for i in range(num_partitions))
    full_concat = results.get(full_key, float("nan"))

    info_present = resid >= chance_sub * 3 and retained >= 0.25

    print()
    if not info_present:
        print("  VERDICT: residual at/near chance. P0 saturates the discriminative")
        print("           subspace — no loss or mining strategy can recover signal that")
        print("           is not present. Reduce K or change testbed.")
    elif not partitions_beat_p0:
        print("  VERDICT: EXTRACTION FAILURE. Residual signal is available but the")
        print("           trained partitions did not capture it (P_k <= P0). The bug is")
        print("           in the loss / mining mechanism.")
    else:
        print("  VERDICT: EXTRACTION SUCCEEDED. The boosted partitions carry MORE")
        print("           task signal than P0 individually. If cosine-concat verification")
        print("           does not improve, the failure is in the COMBINATION metric,")
        print("           not in training. Try --combination confidence_weighted.")

    print()
    for k in range(1, num_partitions):
        pk = pk_accs[k]
        print(f"    P{k} subclass acc = {pk:.4f} vs P0 {p0:.4f}  "
              f"({'+' if pk > p0 else ''}{pk - p0:+.4f})")
    if full_concat == full_concat:  # not NaN
        print(f"    concat(all) probe = {full_concat:.4f} vs P0 {p0:.4f}  "
              f"({full_concat - p0:+.4f})")
        print()
        print("    ^ This is the exact vector cosine_concat scores. If this probe")
        print("      substantially beats P0 but P012 verification does not, the")
        print("      information is present in the vector and cosine is discarding it.")
    print()

    if args.output:
        out = {
            "checkpoint": args.checkpoint,
            "seed": args.seed,
            "r2_trunk_explained_by_p0": r2,
            "chance_subclass": chance_sub,
            "residual_retained_fraction": retained,
            "probes": results,
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"[probe] Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
