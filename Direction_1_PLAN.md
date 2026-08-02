# Direction 1: Learned Combiner Diagnostic — Implementation Plan

**Branch:** `direction-1-learned-combiner`  
**Goal:** Determine whether the cosine-on-concatenated-embeddings combination metric (not the partition training itself) was the bottleneck for Variants C and D. A small MLP combiner trained over frozen checkpoints either rescues performance or confirms the partitions are genuinely uninformative.

---

## Overview

Eight tasks, meant to run end-to-end from a single CLI entry point (`scripts/train_combiner.py`). Total compute budget: less than one working day on a single GPU.

New package: `src/ppi/combiner/` with four modules (`cache.py`, `mlp.py`, `dataset.py`, `trainer.py`). New evaluator: `src/ppi/evaluation/combiner_eval.py`. New config: `configs/direction_1.yaml`. New script: `scripts/train_combiner.py`.

---

## Task 1 — Embedding Cache

**File:** `src/ppi/combiner/cache.py`

### Purpose
Run the frozen backbone + partition heads over the full CASIA training set once and persist per-partition embeddings to disk. Combiner training epochs then become combiner-only forward/backward passes with no backbone compute. Total cache build: ~1 hour.

### Class

```python
class EmbeddingCache:
    def __init__(
        self,
        config: dict,
        checkpoint_path: str,
        cache_dir: str,
        batch_size: int = 256,
        device: str | None = None,
    ) -> None: ...

    def build(self) -> Path:
        """Run backbone over CASIA; save embeddings to cache_dir.

        Saves two .pt files per checkpoint:
          cache_dir/embeddings.pt   — shape (N, num_partitions, K), float32
          cache_dir/labels.pt       — shape (N,), int64

        Returns the cache directory Path.
        Skips build if cache already exists (checks for embeddings.pt).
        """

    @staticmethod
    def load(cache_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (embeddings, labels) tensors from a built cache."""
```

### Design notes
- `.pt` shards (not HDF5) because `torch.save`/`torch.load` requires no extra dependency and is sufficient for ~500k × 3 × K floats.
- Backbone is set to `eval()` and `requires_grad_(False)` before the loop. No gradients computed during cache build.
- `embeddings.pt` stores raw (pre-L2-norm) partition outputs: shape `(N, num_partitions, K)`. L2 normalisation is deferred to the combiner's input assembly, so both K and 3K output variants can share the same cache.
- Cache directory is keyed by checkpoint variant name (e.g. `cache_dir/variant_c/`), so both Variant C and Variant D caches coexist.

---

## Task 2 — Combiner Architecture

**File:** `src/ppi/combiner/mlp.py`

### Class

```python
class PartitionCombiner(nn.Module):
    def __init__(
        self,
        num_partitions: int,    # 3
        partition_dim: int,     # K (per-partition embedding dim)
        hidden_dim: int,        # e.g. 512
        output_dim: int,        # K_out — K or 3K
        dropout: float = 0.1,
    ) -> None: ...

    def forward(
        self,
        embeddings: Tensor,   # (B, num_partitions * partition_dim), zero-padded
        mask: Tensor,         # (B, num_partitions), float 0/1
    ) -> Tensor:
        """Return L2-normalised combined embedding of shape (B, output_dim)."""
```

### Architecture detail
- Input: concatenated per-partition L2-normalised embeddings (padded to `num_partitions * K`), concatenated with the `num_partitions`-bit presence mask. Total input dim = `num_partitions * K + num_partitions`.
- Two hidden layers: `Linear → GELU → Dropout → Linear → GELU → Dropout`.
- Output: `Linear(hidden_dim, output_dim)`, then `F.normalize(..., dim=1)`.
- No batch norm — the combiner sees variable-subset inputs whose statistics differ per subset, and batch norm would couple subsets seen in the same mini-batch.

### Why the mask is separate from the embedding
Zero-padded embeddings are ambiguous: a partition that genuinely produces a near-zero embedding is indistinguishable from an absent partition. The explicit mask gives the combiner subset identity as a first-class input and allows it to learn subset-conditional projections without ambiguity.

### K_out ablation
Two instances trained per experiment:
- `K_out = K` — same dimensionality as P0-alone; enables apples-to-apples comparison with the P0 cosine baseline.
- `K_out = 3K` — full capacity; tests whether dimensionality reduction is part of the problem.

---

## Task 3 — Combiner Training Dataset

**File:** `src/ppi/combiner/dataset.py`

### Classes

```python
class CachedPartitionDataset(Dataset):
    """Subset-sampling dataset for combiner training.

    For each sample, uniformly samples a non-empty subset of {0..N-1}
    at __getitem__ time (not at construction), applies zero-padding and
    presence mask, and returns the masked embedding + mask + label.

    This forces the combiner to handle all 2^N - 1 non-empty subsets
    during training, not only the full N-partition triple.
    """
    def __init__(
        self,
        embeddings: Tensor,   # (N, num_partitions, K), raw pre-norm
        labels: Tensor,       # (N,)
        num_partitions: int,
    ) -> None: ...

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return (masked_embedding, mask, label).

        masked_embedding: (num_partitions * K,) — L2-normed per partition,
                          zero-filled for absent partitions.
        mask:             (num_partitions,) — float 0/1.
        label:            scalar int.
        """


class FullTripleDataset(Dataset):
    """Control: always returns the full N-partition triple (no masking).

    Used for the robustness control run. Comparing uniform-subset training
    against full-triple-only training isolates whether explicit subset
    exposure is necessary for joint signal extraction.
    """
    def __init__(
        self,
        embeddings: Tensor,
        labels: Tensor,
        num_partitions: int,
    ) -> None: ...

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return (full_embedding, all_ones_mask, label)."""
```

### Subset sampling
At each `__getitem__` call, draw a random non-empty subset of `{0, ..., num_partitions-1}` uniformly from all `2^N - 1` possibilities. This means each training example is presented under different subsets across epochs, which is the correct inductive bias for a subset-agnostic combiner.

---

## Task 4 — Combiner Trainer

**File:** `src/ppi/combiner/trainer.py`

### Function

```python
def train_combiner(
    combiner: PartitionCombiner,
    arcface_head: ArcFaceHead,   # newly initialised, trained jointly with combiner
    arcface_loss: ArcFaceLoss,
    train_dataset: Dataset,      # CachedPartitionDataset or FullTripleDataset
    config: dict,
    logger: ExperimentLogger,
    device: torch.device,
) -> PartitionCombiner:
    """Train the combiner with ArcFace loss on combined embeddings.

    Returns the trained combiner (best checkpoint by training loss if
    abandoned at epoch 5, otherwise final epoch).

    Early abandonment: if loss has not decreased from epoch 1 to epoch 5,
    log a warning and return the best checkpoint without raising.
    """
```

### Details
- The combiner and a freshly-initialised `ArcFaceHead(output_dim, num_classes)` are trained jointly. The ArcFace head is the same `ArcFaceLoss(s=64, m=0.5)` as original training, keeping the comparison aligned.
- Optimizer: AdamW (combiner is small; SGD with ArcFace momentum tuning is unnecessary for a 100K-param MLP). Reasonable defaults: `lr=1e-3`, `weight_decay=1e-4`.
- Epochs: run until convergence or epoch 5 max, per the brief's compute constraint.
- Wandb logging: per-step loss, per-epoch mean loss, learning rate.
- Checkpoint saved at the end of each epoch to `run_dir/combiner_epoch_{n}.pt`.

---

## Task 5 — Combiner Evaluator

**File:** `src/ppi/evaluation/combiner_eval.py`

### Class

```python
class CombinerEvaluator:
    """Evaluate a trained PartitionCombiner on LFW (and optionally CFP-FP, AgeDB-30).

    Runs evaluation for all seven non-empty partition subsets, comparing:
      - Combiner output (cosine similarity on combiner embeddings)
      - Cosine-on-concatenated baseline (existing eval, zero-padded)

    Both evaluated at the same subset — the side-by-side comparison is the
    diagnostic output.
    """

    def __init__(
        self,
        backbone: nn.Module,
        combiner: PartitionCombiner,
        config: dict,
        device: torch.device,
    ) -> None: ...

    def evaluate_lfw(
        self,
        partition_configs: list[set[int]] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Run LFW pair verification for all subsets, both combiner and baseline.

        Returns dict keyed by e.g. "P012_combiner", "P012_baseline", "P0_combiner",
        "P0_baseline", etc. Each value has keys: pair_accuracy, pair_std,
        tar_at_far_1e-3.
        """

    def evaluate_graceful_degradation(
        self,
        image_paths: list[str],
        root: str,
    ) -> dict[str, float]:
        """Measure cosine similarity between combine({P0,P1,P2}) and combine({P0})
        for the same identity.

        High similarity → the combiner preserves identity geometry across subsets,
        supporting gallery indexing with mixed partition availability.

        Returns per-identity mean cosine similarity statistics.
        """
```

### Implementation notes
- Raw partition outputs are extracted once per image (backbone run once), then reused across all subset evaluations — same pattern as `Evaluator.extract_raw_partitions_from_paths` in the existing evaluator.
- For the cosine baseline: assemble zero-padded concatenated embedding and L2-normalise per-partition before concat, then compute cosine. This exactly replicates the existing eval.
- For the combiner: feed the zero-padded embedding + mask through `PartitionCombiner.forward()`, use the L2-normalised output directly for cosine similarity.
- CFP-FP and AgeDB-30 evaluation is parallel to LFW — same loop, different `Benchmark` class. Add `evaluate_cfp_fp()` and `evaluate_agedb30()` following the same signature as `evaluate_lfw()`.

---

## Task 6 — Configuration

**File:** `configs/direction_1.yaml`

```yaml
seed: 42

# Checkpoint paths for frozen partition models
checkpoints:
  variant_c: checkpoints/variant_c/final.pt
  variant_d: checkpoints/variant_d/final.pt

# Cache location (auto-built if missing)
cache:
  dir: cache/direction_1
  batch_size: 256

# Combiner architecture
combiner:
  hidden_dim: 512
  dropout: 0.1
  k_out_values: [K, 3K]   # both K and 3K are run; K resolved from partitions.K

# Training
training:
  epochs: 5
  optimizer:
    lr: 1e-3
    weight_decay: 1e-4
  batch_size: 512

# ArcFace — matches original training configuration
arcface:
  s: 64.0
  m: 0.5

# Evaluation benchmarks
evaluation:
  lfw:
    root: data/lfw/
    pairs: data/lfw/pairs.txt
  cfp_fp:
    root: data/cfp-fp/
    pairs: data/cfp-fp/pairs.txt
  agedb:
    root: data/agedb-30/
    pairs: data/agedb-30/pairs.txt

# Wandb
logging:
  wandb_project: ppi-direction-1
  run_name: null   # auto-generated if null
```

---

## Task 7 — Training Script

**File:** `scripts/train_combiner.py`

### CLI arguments

```
--config          Path to direction_1.yaml (required)
--checkpoint      Which frozen checkpoint to use: variant_c | variant_d (required)
--k-out           Output dimensionality: K | 3K (default: K)
--full-triple-only  Flag: use FullTripleDataset instead of CachedPartitionDataset
--cache-dir       Override cache directory from config
--wandb-project   Override wandb project from config
--device          cuda | cpu (default: auto-detect)
--seed            Override seed from config
```

### Execution flow

1. Load config (YAML), apply CLI overrides.
2. `EmbeddingCache.build()` — no-op if cache exists.
3. Load embeddings and labels via `EmbeddingCache.load()`.
4. Construct `CachedPartitionDataset` (or `FullTripleDataset` if `--full-triple-only`).
5. Instantiate `PartitionCombiner` with resolved `K_out`.
6. `train_combiner()` — returns trained combiner.
7. `CombinerEvaluator.evaluate_lfw()` — all 7 subsets, combiner vs baseline.
8. If CFP-FP or AgeDB-30 paths are configured, run those evaluations too.
9. `evaluate_graceful_degradation()` — log inter-subset cosine similarity.
10. Print and log results table.

### Reproducibility
Config + seed fully determine the run. No non-determinism in subset sampling (seeded RNG at dataset construction time, not at `__getitem__` time — use `torch.Generator` seeded from config seed to ensure identical subset draws across runs with the same seed).

---

## Task 8 — Results Table

The evaluator produces a results table covering:

| Rows | Columns |
|------|---------|
| All 7 non-empty subsets: P0, P01, P02, P12\*, P012, P1\*, P2\* | Checkpoint × K_out × Method = {combiner, baseline} |

\* P1-only and P2-only subsets are invalid at inference (P0 is the anchor and always present) but are included as ablation data points to understand each partition's standalone signal.

**Metrics per cell:** `pair_accuracy`, `pair_std`, `tar_at_far_1e-3`.

**Diagnostic outcome** is read from the P012 combiner row vs the P0 baseline row:
- P012 (combiner) ≥ P0 (baseline): success criterion met.
- P012 (combiner) > P012 (baseline): combination was the bottleneck (not just training).
- P012 (combiner) ≈ P012 (baseline): partitions are genuinely uninformative.

Separate tables per checkpoint (Variant C, Variant D) and per K_out setting. Total: 2 × 2 = 4 tables. Cross-checkpoint comparison directly addresses the diagnostic outcome matrix in the brief.

---

## Module Structure Summary

```
src/ppi/combiner/
    __init__.py
    cache.py        # EmbeddingCache
    mlp.py          # PartitionCombiner
    dataset.py      # CachedPartitionDataset, FullTripleDataset
    trainer.py      # train_combiner()

src/ppi/evaluation/
    combiner_eval.py   # CombinerEvaluator (new)
    evaluator.py       # unchanged
    ...

configs/
    direction_1.yaml   # new

scripts/
    train_combiner.py  # new entry point
```

---

## Execution Order

1. `python scripts/train_combiner.py --config configs/direction_1.yaml --checkpoint variant_c --k-out K`
2. `python scripts/train_combiner.py --config configs/direction_1.yaml --checkpoint variant_c --k-out 3K`
3. `python scripts/train_combiner.py --config configs/direction_1.yaml --checkpoint variant_d --k-out K`
4. `python scripts/train_combiner.py --config configs/direction_1.yaml --checkpoint variant_d --k-out 3K`
5. (optional control) `python scripts/train_combiner.py --config configs/direction_1.yaml --checkpoint variant_c --k-out K --full-triple-only`

Steps 1–4 cache embeddings on the first run; subsequent runs reuse the cache. Total wall time: ~1 hour cache build + ~30 minutes combiner training + ~15 minutes evaluation per run.
