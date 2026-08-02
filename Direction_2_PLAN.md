# Direction 2: Boosting Reformulation — Implementation Plan

**Branch:** `direction-2-boosting`  
**Goal:** Replace geometric orthogonality with content-based complementarity. Train each partition specifically on the verification pairs the previous ensemble fails on. Produces non-redundancy by construction against the actual evaluation objective, closing the semantic gap between "geometrically orthogonal" and "discriminatively complementary."

---

## Overview

Nine tasks producing a complete boosting training pipeline. All design decisions are exposed as CLI switches to support systematic ablation without code changes between runs. New package: `src/ppi/boosting/` with five modules. New config: `configs/direction_2_base.yaml`. New script: `scripts/train_boosting.py`. Evaluation extended via `src/ppi/evaluation/boosting_eval.py`.

---

## Task 1 — Hard-Pair Mining

**File:** `src/ppi/boosting/mining.py`

### Class

```python
class HardPairMiner:
    """Dynamic hard-pair mining from the previous ensemble's failure distribution.

    Supports two strategies:
      - band:  pairs whose previous-ensemble cosine score falls in [band_low, band_high]
      - topk:  top-k pairs ranked by previous-ensemble verification loss

    Mining is dynamic: the failure distribution shifts as the current partition
    trains (and the backbone shifts if not frozen), so the pair set is recomputed
    every refresh_every steps.
    """

    def __init__(
        self,
        strategy: str,            # "band" | "topk"
        band_low: float = 0.2,    # used when strategy == "band"
        band_high: float = 0.6,   # used when strategy == "band"
        topk_fraction: float = 0.1,  # used when strategy == "topk"
        refresh_every: int = 500,
    ) -> None: ...

    def mine(
        self,
        embeddings: Tensor,   # (N, ensemble_dim) — previous ensemble embeddings
        labels: Tensor,       # (N,)
        global_step: int,
    ) -> tuple[Tensor, Tensor, dict]:
        """Return (pair_indices_A, pair_indices_B, stats).

        pair_indices_A, pair_indices_B: (M,) int64 indices into the dataset.
        stats: dict with keys "n_pairs", "score_mean", "score_std", "refreshed".

        Mining is skipped (returns cached pair set) if global_step % refresh_every != 0
        and a cached pair set exists. Stats["refreshed"] is True when recomputed.
        """

    def build_hard_pair_dataset(
        self,
        pair_indices_a: Tensor,
        pair_indices_b: Tensor,
        source_dataset: Dataset,
    ) -> Dataset:
        """Wrap indexed source pairs into a Dataset for the DataLoader."""
```

### Design decisions
- **Partition-count-agnostic:** `mine()` accepts arbitrary-dim ensemble embeddings so it works identically for 1-partition (P0 alone), 2-partition (P0+P1), or N-1-partition ensembles. No hardcoding to N=3.
- **Dynamic refresh:** The failure distribution of the previous ensemble changes as the current partition trains (especially if the backbone is not frozen). `refresh_every=500` steps is the default, with very high values approximating static mining for ablation.
- **Cache invalidation:** When `global_step % refresh_every == 0`, recompute pair set. Use a `torch.Generator` seeded by `global_step` to ensure reproducibility of refresh-time randomness.
- **Band strategy:** Captures the genuine/impostor confusion zone. Pairs with cosine in `[band_low, band_high]` are ambiguous to the current ensemble — exactly where the new partition should contribute.
- **Topk strategy:** Ranks all pairs by previous-ensemble cross-entropy loss, selects top fraction. More aggressive than band; useful when the confusion zone is narrow.

---

## Task 2 — Boosting Losses

**File:** `src/ppi/boosting/losses.py`

### Classes

```python
class ArcFaceReweighted(nn.Module):
    """ArcFace with per-pair weights from previous ensemble confidence.

    Weight = 1 - prev_score for genuine pairs, prev_score for impostors.
    Acknowledged geometric assumption violation (ArcFace is a classification
    loss, not a metric loss); included for ablation completeness.
    """
    def __init__(self, s: float = 64.0, m: float = 0.5) -> None: ...
    def forward(
        self,
        cosine: Tensor,          # (B, num_classes) from ArcFaceHead
        labels: Tensor,          # (B,)
        pair_weights: Tensor,    # (B,) float — confidence from prev ensemble
    ) -> Tensor: ...


class ArcFaceMargin(nn.Module):
    """Two-loss: margin-based on hard pairs, ArcFace as regulariser on easy pairs.

    total_loss = hard_loss + easy_loss_weight * arcface_loss_on_easy_pairs
    """
    def __init__(
        self,
        s: float = 64.0,
        m: float = 0.5,
        easy_loss_weight: float = 0.3,
    ) -> None: ...
    def forward(
        self,
        hard_cosine: Tensor,    # (H, num_classes)
        hard_labels: Tensor,    # (H,)
        easy_cosine: Tensor,    # (E, num_classes)
        easy_labels: Tensor,    # (E,)
    ) -> Tensor: ...


class TripletLoss(nn.Module):
    """Triplet loss with anchors, positives, and negatives from hard pairs.

    Default loss. Optimises pairwise relative ordering directly, matching
    the verification evaluation objective more closely than classification losses.

    mining_strategy: "batch_hard" (hardest positive/negative per anchor in batch)
                  or "semi_hard" (hardest negative further than positive).
    """
    def __init__(
        self,
        margin: float = 0.3,
        mining_strategy: str = "batch_hard",
    ) -> None: ...
    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor: ...


class ContrastiveLoss(nn.Module):
    """Contrastive loss on hard genuine/impostor pairs."""
    def __init__(self, margin: float = 1.0) -> None: ...
    def forward(
        self,
        emb_a: Tensor,    # (B, K)
        emb_b: Tensor,    # (B, K)
        is_same: Tensor,  # (B,) bool
    ) -> Tensor: ...


class SubCenterArcFace(nn.Module):
    """Sub-center ArcFace: K sub-centers per class to handle intra-class variation."""
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        s: float = 64.0,
        m: float = 0.5,
        K: int = 3,
    ) -> None: ...
    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor: ...


def build_loss(config: dict) -> nn.Module:
    """Factory function. Reads config["boosting"]["loss"] and instantiates."""
```

### Why triplet is the default
Triplet loss optimises pairwise relative ordering directly, matching the LFW verification objective (is pair A same person?) more closely than classification-derived losses (ArcFace, sub-center ArcFace) which optimise the class boundary rather than pair similarity. For a boosting regime where training signal comes from hard pairs rather than all training data, triplet's explicit pair-level objective is the natural fit.

---

## Task 3 — Backbone State Control

**File:** `src/ppi/boosting/backbone_state.py`

### Function

```python
def set_backbone_state(
    backbone: nn.Module,
    state: str,                    # "frozen" | "fine_tuned" | "partial"
    frozen_stages: int = 3,        # used when state == "partial"
    backbone_lr_multiplier: float = 0.1,  # used when state != "frozen"
    base_lr: float = 0.1,
) -> list[dict]:
    """Configure backbone parameter groups for the optimizer.

    Returns a list of parameter group dicts (compatible with torch.optim).
    Frozen parameters are excluded from optimizer groups entirely (not set to
    lr=0, which would still waste memory on gradient buffers).

    For "partial": ResNet-50 is split into freezable groups:
      0: stem (conv1 + bn1)
      1: layer1 (residual stage 1)
      2: layer2 (residual stage 2)
      3: layer3 (residual stage 3)
      4: layer4 (residual stage 4)
      5: final projection / pooling
    Groups 0..frozen_stages-1 are frozen; groups frozen_stages..5 are trainable.
    Default frozen_stages=3 freezes stem + first two residual stages.
    """
```

### Why this is separate from the trainer
Backbone state reconfiguration happens at phase boundaries (after P0 training completes, before P1 training begins, and again before P2). Isolating this into a utility function makes the phase transition logic in `BoostingTrainer` explicit and testable.

---

## Task 4 — Boosting Trainer

**File:** `src/ppi/boosting/trainer.py`

### Class

```python
class BoostingTrainer:
    """Phase-sequential boosting trainer.

    Phase 0: Train P0 (standard ArcFace, backbone fully trainable).
    Phase k (k=1..N-1): Freeze previous partition heads; set backbone state
        per --backbone-state; mine hard pairs from previous ensemble; train
        current partition head on hard pairs with chosen loss; refresh pair
        set every refresh_every steps.

    All hyperparameters from config, all sweepable via CLI overrides.
    """

    def __init__(
        self,
        config: dict,
        device: torch.device,
        logger: ExperimentLogger,
    ) -> None: ...

    def train(self) -> None:
        """Run all phases. Saves per-phase checkpoints separately for
        backbone and partition heads.

        Checkpoint structure:
          checkpoints/boosting/<run_name>/phase_{k}/backbone.pt
          checkpoints/boosting/<run_name>/phase_{k}/partition_{k}.pt
        Separate checkpoint files allow Direction 1's combiner to load
        individual partition heads without loading the full model.
        """

    def _train_phase_0(self) -> None:
        """Train P0 with standard ArcFace (identical to existing trainer,
        backbone fully trainable)."""

    def _train_phase_k(self, k: int) -> None:
        """Train partition k on hard pairs from ensemble of partitions 0..k-1.

        1. Freeze partitions 0..k-1 heads.
        2. Set backbone state (frozen/fine_tuned/partial).
        3. Compute previous ensemble embeddings over training set.
        4. Run HardPairMiner to get initial pair set.
        5. Train for config epochs, refreshing pair set every refresh_every steps.
        6. Log: per-step loss, pair count at each refresh, score distribution.
        """
```

### Wandb logging
- Per-step: loss (all components), mining pair count (at refresh events), backbone LR, partition head LR.
- Per-epoch: epoch mean loss per phase, hard pair score distribution (mean, std, histogram).
- Per-phase boundary: pair count before and after first refresh, mining strategy params.
- Phase boundaries logged explicitly with timestamps.

### Checkpoint design
Backbone and partition heads are saved separately per phase. This supports:
1. Mid-training analysis of individual partition heads.
2. Direction 1 combiner loading individual partition heads without the full model.
3. Partial restarts (resume from a phase boundary without restarting phase 0).

---

## Task 5 — Combination Strategies

**File:** `src/ppi/boosting/combination.py`

### Classes

```python
class CosineConcat:
    """Cosine similarity on concatenated per-partition L2-normalised embeddings.

    Default baseline — exactly replicates the existing eval in Evaluator.evaluate_lfw().
    Absent partitions are zero-padded. Enables Direction 2 to be evaluated
    independently of Direction 1's combiner outcome.
    """
    def combine(
        self,
        partition_embeddings: list[Tensor],  # list of (B, K), None for absent
    ) -> Tensor:
        """Return (B, num_partitions * K) concatenated embedding."""


class ConfidenceWeighted:
    """Per-partition similarity scores weighted by per-partition confidence.

    confidence_source: "embedding_norm" | "cosine_magnitude" | "scalar_head"
      - embedding_norm: pre-normalisation embedding magnitude (proxy for
                        representation quality; high norm → confident).
      - cosine_magnitude: |cosine(emb_a, emb_b)| — decisive scores signal
                          confident partitions.
      - scalar_head: tiny per-partition learned scalar head, ConfidenceHead(K, 1),
                     producing a calibrated confidence score from the embedding.
                     Adds ~K parameters per partition; is the most direct.
    """
    def __init__(
        self,
        confidence_source: str,
        num_partitions: int,
        partition_dim: int,
    ) -> None: ...

    def combine(
        self,
        partition_embeddings: list[Tensor],
        emb_norms: list[Tensor] | None = None,   # pre-norm magnitudes
    ) -> Tensor:
        """Return weighted combination embedding."""


class LearnedCombiner:
    """Placeholder wrapping a trained Direction 1 PartitionCombiner.

    No retraining of Direction 2 partitions required. The combiner is loaded
    from a Direction 1 checkpoint and applied post-hoc over Direction 2's
    frozen partition outputs.

    Available as --combination learned_combiner only when a Direction 1
    combiner checkpoint is provided via --d1-combiner-path.
    """
    def __init__(self, combiner_checkpoint: str, device: torch.device) -> None: ...
    def combine(
        self,
        partition_embeddings: list[Tensor],
        mask: Tensor,
    ) -> Tensor: ...


def get_combiner(strategy: str, **kwargs) -> CosineConcat | ConfidenceWeighted | LearnedCombiner:
    """Factory. strategy: "cosine_concat" | "confidence_weighted" | "learned_combiner"."""
```

### Why `cosine_concat` is the default
Direction 2 must be evaluable on its own terms, independent of Direction 1's outcome. `cosine_concat` matches the existing Variant A–D evaluation exactly, providing a clean comparison. `confidence_weighted` is the natural Direction-2-native variant (P1's training on hard pairs gives its embeddings a natural confidence signal). `learned_combiner` becomes available post-hoc if Direction 1 validates the combiner.

---

## Task 6 — Configuration and CLI

**File:** `configs/direction_2_base.yaml`

```yaml
seed: 42
num_partitions: 3    # supports N > 3 via --num-partitions

backbone:
  name: resnet50
  # (other backbone config inherited from base.yaml)

boosting:
  backbone_state: partial       # frozen | fine_tuned | partial
  frozen_stages: 3              # for partial: freeze stem + first 2 residual stages
  backbone_lr_multiplier: 0.1   # backbone LR relative to partition head LR
  mining_strategy: topk         # band | topk
  mining_band_low: 0.2
  mining_band_high: 0.6
  mining_topk_fraction: 0.1
  mining_refresh_every: 500
  loss: triplet                 # arcface_reweighted | arcface_margin | triplet
                                # | contrastive | sub_center_arcface
  easy_loss_weight: 0.3         # for arcface_margin
  triplet_margin: 0.3
  triplet_mining: batch_hard    # batch_hard | semi_hard
  contrastive_margin: 1.0
  sub_center_K: 3
  combination: cosine_concat    # cosine_concat | confidence_weighted | learned_combiner
  confidence_source: embedding_norm  # embedding_norm | cosine_magnitude | scalar_head
  d1_combiner_path: null        # path to Direction 1 combiner checkpoint

training:
  epochs_phase0: 20
  epochs_per_phase: 20          # epochs for each boosting phase
  optimizer:
    lr: 0.1
    momentum: 0.9
    weight_decay: 5e-4
  grad_clip: 5.0
  checkpoint_interval: 5

data:
  dataset: casia                # cifar100 | casia_subset | casia
  # (data paths inherited from base.yaml or overridden via CLI)

arcface:
  s: 64.0
  m: 0.5
  num_classes: 10572            # CASIA-WebFace

evaluation:
  lfw:
    root: data/lfw/
    pairs: data/lfw/pairs.txt
  # cfp_fp and agedb keys optional

logging:
  wandb_project: ppi-direction-2
  run_name: null
```

**File:** `scripts/train_boosting.py`

### CLI arguments (all sweepable, override config)

```
--config                  Path to YAML config (required)
--backbone-state          frozen | fine_tuned | partial
--backbone-lr-multiplier  Float (default from config)
--frozen-stages           Int 0–5 (for partial backbone state)
--mining-strategy         band | topk
--mining-band-low         Float
--mining-band-high        Float
--mining-topk             Float fraction (e.g. 0.1 = top 10%)
--mining-refresh-every    Int steps
--loss                    arcface_reweighted | arcface_margin | triplet |
                          contrastive | sub_center_arcface
--easy-loss-weight        Float (for arcface_margin)
--triplet-margin          Float
--triplet-mining          batch_hard | semi_hard
--contrastive-margin      Float
--sub-center-K            Int
--combination             cosine_concat | confidence_weighted | learned_combiner
--confidence-source       embedding_norm | cosine_magnitude | scalar_head
--d1-combiner-path        Path to Direction 1 combiner checkpoint
--num-partitions          Int (default 3; N-agnostic design)
--dataset                 cifar100 | casia_subset | casia
--epochs-phase0           Int
--epochs-per-phase        Int
--wandb-project           String
--seed                    Int
--device                  cuda | cpu
--resume                  Path to phase boundary checkpoint to resume from
```

### Execution flow

1. Load config, apply CLI overrides.
2. Instantiate `BoostingTrainer`.
3. `trainer.train()` — phases 0 through N-1.
4. `BoostingEvaluator.evaluate()` — full eval grid: all 7 subsets × all combination strategies × LFW (+ CFP-FP, AgeDB-30 if configured).
5. Print results table; log to wandb.

---

## Task 7 — CIFAR-100 Smoke Test

**File:** `src/ppi/boosting/cifar100_adaptor.py`

### Purpose
Test the boosting mechanism in isolation on a dataset with a clean hierarchical structure before committing GPU-days to CASIA. Success criterion: strict monotone improvement P012 > P01 > P0 on CIFAR-100 superclass/subclass verification under at least one CLI configuration.

### Mapping

```python
class CIFAR100BoostingAdaptor:
    """Adapts CIFAR-100 superclass/subclass hierarchy to the PPI boosting interface.

    Partition assignment:
      P0: trained on superclass prediction (20 classes)
      P1: boosted on pairs where P0 fails at subclass level (100 classes)
      P2: boosted on pairs where P0+P1 fail

    Verification framing: given two images, are they the same subclass?
    This matches the LFW same-person/different-person structure.

    The CIFAR-100 superclass structure provides 20 groups of 5 subclasses each,
    giving a natural two-level hierarchy for the boosting mechanism to exploit.
    """

    def get_train_dataset(self) -> Dataset: ...
    def get_val_pairs(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return (images_a, images_b, is_same_subclass) for verification eval."""
    def superclass_label(self, subclass_label: int) -> int: ...
```

### Execution
Use `--dataset cifar100` in `scripts/train_boosting.py`. The adaptor is transparent to `BoostingTrainer` — it exposes the same dataset interface, just with superclass labels for P0 and subclass labels for P1/P2 hard-pair mining.

---

## Task 8 — Boosting Evaluation

**File:** `src/ppi/evaluation/boosting_eval.py`

### Class

```python
class BoostingEvaluator:
    """Evaluate Direction 2 models across all subset × combination strategy combinations.

    Produces the full comparison table:
      - All 7 non-empty partition subsets
      - All combination strategies: cosine_concat, confidence_weighted,
        learned_combiner (if --d1-combiner-path provided)
      - LFW, CFP-FP, AgeDB-30 (if configured)
      - Ablation comparison against Variants A–D results (loaded from
        saved result JSON files if --baseline-results-dir is provided)
    """

    def __init__(
        self,
        backbone: nn.Module,
        partition_heads: list[nn.Module],
        combination_strategies: list[str],
        config: dict,
        device: torch.device,
    ) -> None: ...

    def evaluate_all(self) -> dict[str, dict[str, dict[str, float]]]:
        """Run full eval grid.

        Returns nested dict: subset_name → strategy_name → metric_name → value.
        """
```

### Single eval pass
The backbone runs once per image over LFW. Raw partition outputs are cached. All 7 subsets × all combination strategies are then evaluated from the cache. One eval pass produces the complete table.

### Ablation comparison
If `--baseline-results-dir` is provided, load saved result JSONs from Variant A–D evaluations and include them as additional rows in the results table for direct comparison.

---

## Task 9 — Phased Execution Plan

### Phase 1: CIFAR-100 Smoke Test
**Goal:** Confirm the boosting mechanism produces strict P012 > P01 > P0 under at least one configuration.

**Config:** Use `--dataset cifar100 --backbone-state partial --frozen-stages 3 --mining-strategy topk --mining-topk 0.1 --mining-refresh-every 500 --loss triplet --combination cosine_concat`. This is the brief's suggested initial configuration.

**Go/no-go:** If strict monotone improvement appears under any configuration, advance to Phase 2. If not, sweep `--mining-strategy` and `--loss` on CIFAR-100 before scaling. Do not advance to CASIA without CIFAR-100 success.

**Compute:** Hours, not days. CIFAR-100 training is cheap; iterate rapidly.

### Phase 2: CASIA Subset Run
**Goal:** Gate full-scale runs and narrow the hyperparameter grid.

**Dataset:** 100k images, 2k identities subset of CASIA (add `--dataset casia_subset` support — random-seed-fixed identity-stratified sample from the full CASIA index).

**Config:** Use the best configuration from Phase 1. Run 2–3 configurations to compare.

**Go/no-go:** If P012 > P01 > P0 holds on the subset, advance to Phase 3 with the best configuration. If not, return to Phase 1 to check CIFAR-100 vs CASIA transfer.

**Compute:** ~6 hours per run (subset is ~20% of full CASIA).

### Phase 3: Full CASIA Training
**Goal:** Validate the candidate configuration at full scale.

**Config:** Best configuration from Phase 2.

**Success criteria:**
- P012 > P01 > P0 (strict monotone improvement).
- P0 ≥ functional floor (P0 accuracy acceptable for local-only deployment, i.e. competitive with "no system at all").
- P012 competitive with or exceeding centralised ArcFace baseline.

**Compute:** ~48 hours per run on a single GPU.

### Phase 4: Ablation Runs
**Goal:** Quantify the contribution of each CLI axis at full CASIA scale.

Ablation axes (in order of expected impact):
1. `--backbone-state`: frozen vs fine_tuned vs partial (frozen_stages sweep).
2. `--loss`: triplet vs contrastive vs arcface_margin.
3. `--mining-strategy`: topk vs band (band_low/band_high sweep).
4. `--combination`: cosine_concat vs confidence_weighted (confidence_source sweep).

Run ablations for the axes that remained consequential after Phase 2. If Phase 2 clearly resolved one axis, skip its Phase 4 ablation to conserve compute.

---

## Module Structure Summary

```
src/ppi/boosting/
    __init__.py
    mining.py           # HardPairMiner
    losses.py           # ArcFaceReweighted, ArcFaceMargin, TripletLoss,
                        # ContrastiveLoss, SubCenterArcFace, build_loss()
    backbone_state.py   # set_backbone_state()
    trainer.py          # BoostingTrainer
    combination.py      # CosineConcat, ConfidenceWeighted, LearnedCombiner,
                        # get_combiner()
    cifar100_adaptor.py # CIFAR100BoostingAdaptor

src/ppi/evaluation/
    boosting_eval.py    # BoostingEvaluator (new)
    evaluator.py        # unchanged
    ...

configs/
    direction_2_base.yaml   # new

scripts/
    train_boosting.py       # new entry point
```

---

## Relationship to Direction 1

- `BoostingTrainer` saves per-phase partition checkpoints separately, enabling Direction 1's `EmbeddingCache` to load them without loading the full Direction 2 model.
- `LearnedCombiner` in `combination.py` wraps Direction 1's `PartitionCombiner` as a post-hoc evaluation option — no retraining of Direction 2 partitions required.
- The default `cosine_concat` combination makes Direction 2 evaluable independently of Direction 1's outcome.
- Direction 1's verdict on Variant D's orthogonality contribution is informative but not gating: orthogonality is not part of Direction 2's design space.

---

## N-Agnosticism

All components are designed to generalise to N > 3 without architectural changes:
- `HardPairMiner.mine()` accepts arbitrary-dim ensemble embeddings.
- `BoostingTrainer._train_phase_k()` iterates k from 1 to N-1.
- `CosineConcat`, `ConfidenceWeighted`, `LearnedCombiner` accept variable-length partition lists.
- `--num-partitions N` in the CLI controls the partition count. Phase 3 N-scaling experiments (N=5, N=7) require no code changes.
