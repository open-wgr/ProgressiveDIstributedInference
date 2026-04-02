# Progressive Partitioned Inference — Project Bootstrap

**Working document for Claude Code development**
**FML Node Project — April 2026**

---

## 1. What This Document Covers

This is the development roadmap for the Phase 1 (simulation) codebase. It covers:

- Repository structure and conventions
- The shared framework that all four partition variants build on
- Staged validation strategy: CIFAR-100 → CASIA-WebFace → MS1MV2 (see Section 6)
- Testing and validation strategy
- Step-by-step implementation path from skeleton to Experiment 1a (first variant comparison)

It does **not** cover Phase 2 (latency modelling) or Phase 3 (on-hardware deployment). Those depend on Phase 1 results and will get their own bootstrap documents.

---

## 2. Repository Structure

```
ppi/
├── README.md
├── pyproject.toml                # Project metadata, dependencies
├── configs/
│   ├── base.yaml                 # Shared training config (Section 3 of exp plan)
│   ├── stage0_cifar100.yaml      # CIFAR-100 smoke test overrides
│   ├── stage1_casia.yaml         # CASIA-WebFace validation overrides
│   ├── stage2_ms1mv2.yaml        # MS1MV2 full-scale overrides
│   ├── variant_a.yaml            # Orthogonal partitions overrides
│   ├── variant_b.yaml            # Nested/slimmable overrides
│   ├── variant_c.yaml            # Residual boosting overrides
│   └── variant_d.yaml            # Combined overrides
├── src/
│   └── ppi/
│       ├── __init__.py
│       ├── backbones/
│       │   ├── __init__.py
│       │   ├── resnet.py         # ResNet-50 adapted for partitioned heads
│       │   └── mobilefacenet.py  # MobileFaceNet adapted for partitioned heads
│       ├── heads/
│       │   ├── __init__.py
│       │   ├── arcface.py        # ArcFace classification head
│       │   └── partition_head.py # Base partition head (K-dim output per partition)
│       ├── partitions/
│       │   ├── __init__.py
│       │   ├── base.py           # Abstract partition strategy interface
│       │   ├── orthogonal.py     # Variant A
│       │   ├── nested.py         # Variant B (slimmable)
│       │   ├── residual.py       # Variant C (boosting)
│       │   └── combined.py       # Variant D
│       ├── losses/
│       │   ├── __init__.py
│       │   ├── arcface_loss.py   # ArcFace with partition-aware forward
│       │   └── orthogonality.py  # Gram matrix regularisation (Variants A, D)
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py        # Main training loop
│       │   ├── partition_dropout.py  # Stochastic partition zeroing
│       │   └── schedulers.py     # LR schedule (warmup + cosine)
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── evaluator.py      # Runs all partition configs against benchmarks
│       │   ├── metrics.py        # TAR@FAR, rank-1, score distributions
│       │   └── benchmarks.py     # LFW, CFP-FP, AgeDB-30, IJB-B/C loaders
│       ├── data/
│       │   ├── __init__.py
│       │   ├── cifar100.py       # CIFAR-100 wrapper (Stage 0)
│       │   ├── casia.py          # CASIA-WebFace dataset (Stage 1)
│       │   └── ms1mv2.py         # MS1MV2 dataset + transforms (Stage 2)
│       └── utils/
│           ├── __init__.py
│           ├── config.py         # YAML config loading + merging
│           └── logging.py        # Experiment logging (tensorboard / wandb)
├── scripts/
│   ├── train.py                  # Entry point: python scripts/train.py --config configs/variant_a.yaml
│   ├── evaluate.py               # Entry point: python scripts/evaluate.py --checkpoint X --partitions 1,2,3
│   ├── sweep.py                  # Hyperparameter sweep launcher
│   └── convert_rec.py            # One-off: convert MXNet RecordIO → image folders (CASIA, MS1MV2)
├── tests/
│   ├── conftest.py               # Shared fixtures (tiny model, fake data)
│   ├── test_partition_dropout.py
│   ├── test_orthogonality_loss.py
│   ├── test_arcface_loss.py
│   ├── test_partition_assembly.py # Embedding assembly from subsets
│   ├── test_nested_bn.py         # Separate BN stats per width (Variant B)
│   ├── test_residual_freeze.py   # Gradient flow isolation (Variant C)
│   └── test_evaluation.py        # Metric computation on known distributions
└── notebooks/
    └── analysis.ipynb            # Post-training analysis and plotting
```

### Conventions

- **Config-driven**: All hyperparameters live in YAML. The training script takes `--config` and optional `--override key=value` args. No magic numbers in source code.
- **Variant as strategy**: Each partition variant implements the same interface (`PartitionStrategy`). The trainer doesn't know which variant it's running — it calls the same methods regardless.
- **Deterministic by default**: Every run seeds RNG (torch, numpy, python random) from config. Reproducibility is non-negotiable for the paper.
- **Logging**: Every run logs to a structured directory: `runs/{variant}_{timestamp}/`. Checkpoints, configs, metrics CSVs, and tensorboard logs all go here.

---

## 3. Dependencies

```
# Core
torch >= 2.1
torchvision                 # Also provides CIFAR-100 for Stage 0
timm                        # Pretrained backbones (optional, for init comparisons)
onnx                        # Export for TensorRT in Phase 3

# Training
pyyaml
tensorboard
wandb                       # Optional, for remote tracking

# Evaluation
scikit-learn                # ROC, metrics
scipy                       # Score distribution analysis
numpy
pandas                      # Results tables

# Data
mxnet                       # CASIA/MS1MV2 RecordIO conversion only (not needed at training time)
pillow
albumentations              # Augmentations

# Testing
pytest
pytest-cov
```

Pin exact versions in `pyproject.toml` once the environment is set up. Both CASIA-WebFace and MS1MV2 ship as MXNet `.rec` files — run `scripts/convert_rec.py` once to extract to image folders, then use standard `ImageFolder` loaders at training time. CIFAR-100 comes via `torchvision.datasets` and needs no conversion.

---

## 4. Shared Framework — What All Variants Share

Before implementing any variant, the shared infrastructure must be solid. This is the bulk of the early work.

### 4.1 Backbone with Partitioned Output

Both ResNet-50 and MobileFaceNet are used as feature extractors. The standard final FC layer (which produces the embedding) is replaced with a partitioned output structure:

```
Input (112×112 face crop)
    → Backbone (ResNet-50 or MobileFaceNet, everything up to final pooling)
    → Global average pool → feature vector (e.g. 512-d for ResNet-50)
    → Partition heads: 3 parallel linear projections, each producing K-dim output
    → Concatenation → 3K-dim full embedding
```

The partition heads are where the variants diverge. But the backbone, pooling, and concatenation logic are shared.

**Key implementation detail**: The backbone outputs a single feature vector. Each partition head is a `nn.Linear(backbone_dim, K)`. The variant-specific logic controls how these heads are trained, regularised, and initialised — not their forward pass architecture (which is identical for A, B, and D; Variant C differs in that heads are trained sequentially).

### 4.2 Partition Masking and Assembly

At inference time, the system must handle any subset of partitions. The embedding is always a fixed-length 3K vector. Absent partitions are zero-padded.

```python
def assemble_embedding(partition_outputs: dict[int, Tensor], K: int, num_partitions: int = 3) -> Tensor:
    """
    partition_outputs: {0: tensor(K,), 2: tensor(K,)} — whichever partitions are available
    Returns: tensor(3K,) with zeros for missing partitions
    """
    full = torch.zeros(num_partitions * K)
    for idx, vec in partition_outputs.items():
        full[idx * K : (idx + 1) * K] = vec
    return full
```

This is simple but critical — it must be identical in training and evaluation. During training, partition dropout zeros entire slots using this same convention.

### 4.3 ArcFace Head

The ArcFace classification head operates on the assembled (possibly masked) 3K-dim embedding. Standard ArcFace with `s=64, m=0.5`. The head must handle zero-padded inputs gracefully — when a partition is dropped, the corresponding embedding channels are zero, and the ArcFace head must still produce a valid loss.

**Important**: The ArcFace head is shared across all width configurations. There is **not** a separate classifier per partition count. The network must learn embeddings where the zero-padding convention works at all widths.

### 4.4 Partition Dropout

During training, entire partitions are stochastically zeroed. The dropout module:

1. Samples a configuration from the probability distribution (configurable per `configs/base.yaml`)
2. Zeros the selected partition outputs before assembly
3. Passes the masked embedding to ArcFace

Starting distribution from the experimental plan:

| Config | Probability | Meaning |
|--------|-------------|---------|
| 1 partition only | 40% | Randomly pick one of {0}, {1}, {2} |
| 2 partitions | 30% | Randomly pick one of {0,1}, {0,2}, {1,2} |
| All 3 | 20% | No masking |
| None | 10% | All zeros — regularisation, backbone must cope |

The "1 partition only" case selects uniformly from the three partitions, so each individual partition appears solo ~13.3% of the time. Same logic for pairs. This distribution is a hyperparameter to be swept in Experiment 1c.

### 4.5 Training Loop

The trainer handles:

1. Loading config (base + variant-specific overlay)
2. Building backbone + partition heads + ArcFace head
3. For each batch:
   - Forward through backbone
   - Forward through all partition heads
   - Apply partition dropout (sample config, zero selected outputs)
   - Assemble embedding
   - Compute ArcFace loss on assembled embedding
   - Add variant-specific auxiliary loss (e.g. orthogonality for A/D)
   - Backward, step optimiser
4. Periodic validation on LFW (fast check) at configurable interval
5. Checkpointing: save model + optimiser state + config + metrics at each epoch

**Variant-specific hooks in the training loop:**

- `compute_auxiliary_loss(partition_outputs)` — returns 0 for B/C, returns orthogonality loss for A/D
- `pre_training_setup()` — no-op for A/B/D, handles sequential freeze/unfreeze for C
- `get_trainable_parameters(phase)` — all params for A/B/D, phase-dependent for C

### 4.6 Evaluation Protocol

The evaluator takes a checkpoint and runs it at all 7 non-degenerate partition configurations:

- Single partitions: {0}, {1}, {2}
- Pairs: {0,1}, {0,2}, {1,2}
- Full: {0,1,2}

For each configuration, on each benchmark (LFW, CFP-FP, AgeDB-30, IJB-B, IJB-C):

- Rank-1 identification rate
- TAR @ FAR=1e-4
- Genuine/impostor score distributions (saved as numpy arrays for later plotting)

Additionally, a non-partitioned baseline (standard ArcFace at the same total embedding dimension) is trained and evaluated as the accuracy ceiling.

### 4.7 Config Schema

```yaml
# configs/base.yaml
seed: 42

backbone:
  name: resnet50          # or mobilefacenet
  pretrained: false

partitions:
  num_partitions: 3
  K: 128                  # per-partition embedding dim; total = 3K = 384
  dropout:
    enabled: true
    distribution:         # probabilities for [1-part, 2-part, 3-part, 0-part]
      - 0.40
      - 0.30
      - 0.20
      - 0.10

arcface:
  s: 64
  m: 0.5

training:
  epochs: 24
  batch_size: 256
  optimizer:
    type: sgd
    lr: 0.1
    momentum: 0.9
    weight_decay: 5.0e-4
  scheduler:
    type: cosine
    warmup_epochs: 1
  val_interval: 2         # validate every N epochs
  checkpoint_interval: 1

data:
  dataset: ms1mv2         # overridden per stage config
  root: /data/ms1mv2/
  num_workers: 8
  input_size: 112         # face crop size; overridden for CIFAR-100

logging:
  output_dir: runs/
  tensorboard: true
  wandb: false
```

Stage configs override dataset and training parameters:

```yaml
# configs/stage0_cifar100.yaml
_base_: base.yaml

data:
  dataset: cifar100
  root: /data/cifar100/       # auto-downloaded by torchvision
  input_size: 32
  num_workers: 4

backbone:
  name: resnet18              # smaller backbone for toy task

partitions:
  K: 64                      # smaller partitions; total = 192

training:
  epochs: 50                  # CIFAR-100 needs more epochs, but they're fast
  batch_size: 128

arcface:
  num_classes: 100            # CIFAR-100 classes, not face identities
```

```yaml
# configs/stage1_casia.yaml
_base_: base.yaml

data:
  dataset: casia
  root: /data/casia_webface/

training:
  epochs: 24                  # same schedule as MS1MV2

# Everything else inherits from base — same backbone, K, dropout, etc.
```

```yaml
# configs/stage2_ms1mv2.yaml
_base_: base.yaml
# Identical to base — exists for explicitness and stage naming consistency
```

Variant configs layer on top of stage configs:

```yaml
# configs/variant_a.yaml
# Applied AFTER a stage config: train.py --config stage1_casia.yaml --variant variant_a.yaml

variant: orthogonal

orthogonality:
  lambda: 0.1             # regularisation weight
  
positional_encoding:
  type: learned            # learned | fixed_slot | fixed_orthogonal
  dim: 16                  # only for learned
```

---

## 5. Testing Strategy

Tests run on CPU with tiny synthetic data. No GPU or real datasets required. The goal is to verify correctness of the framework mechanics, not model quality.

### 5.1 Shared Framework Tests

**`test_partition_assembly.py`** — Verify that:
- Assembling all 3 partitions produces a 3K vector with correct slot placement
- Assembling a subset zero-pads the missing slots
- Assembling an empty set produces all zeros
- Assembly is deterministic and matches between training and eval code paths

**`test_partition_dropout.py`** — Verify that:
- Dropout samples from the configured distribution (run 10k samples, chi-squared test against expected frequencies)
- Dropped partitions are exactly zero in the assembled embedding
- Non-dropped partitions are unmodified
- Dropout is disabled during eval mode
- The "0 partitions" case produces an all-zero embedding

**`test_arcface_loss.py`** — Verify that:
- Loss is finite and positive for a random embedding and random target
- Loss works with zero-padded embeddings (no NaN/Inf)
- Gradients flow correctly through non-zero partition slots
- Gradients are zero for zero-padded slots (no phantom gradients)

**`test_evaluation.py`** — Verify that:
- Known genuine/impostor distributions produce expected TAR@FAR
- Rank-1 rate is correct on a trivial gallery
- Evaluator runs all 7 partition configs without error
- Metrics are deterministic across runs

### 5.2 Variant-Specific Tests (written per variant as implemented)

**`test_orthogonality_loss.py`** (Variant A) — Verify that:
- Orthogonal partition outputs produce loss near zero
- Identical partition outputs produce high loss
- Gradient flows to all three partition heads
- Lambda scaling works correctly

**`test_nested_bn.py`** (Variant B) — Verify that:
- Separate BN statistics are maintained for each width config (K, 2K, 3K)
- Switching width at eval time uses the correct BN stats
- BN stats are not shared or leaked between width configs

**`test_residual_freeze.py`** (Variant C) — Verify that:
- During phase 1 training, only partition 0 head receives gradients
- During phase 2, partition 0 is frozen and only partition 1 receives gradients
- During phase 3, partitions 0 and 1 are frozen
- Optional joint fine-tuning unfreezes all heads

### 5.3 Integration Tests

**`test_training_loop.py`** — Run 2 epochs on a tiny synthetic dataset (100 images, 10 identities) with a miniature backbone (4-layer CNN, K=8). Verify that:
- Loss decreases
- Checkpoints are saved correctly
- Checkpoint loading reproduces the same model outputs
- Config is saved alongside checkpoint

**`test_config_loading.py`** — Verify that:
- Base config loads correctly
- Variant config overrides base values
- Missing required keys raise clear errors
- Unknown keys raise warnings

---

## 6. Implementation Order

This is the build sequence — each step depends on the previous ones being done and tested. The key change from the original plan: GPU-intensive validation is staged across three datasets of increasing size, so the concept is validated cheaply before committing serious compute.

### Step 1: Project Skeleton

Set up the repo structure, `pyproject.toml`, empty `__init__.py` files, and the config loading utility. Write `test_config_loading.py` and make it pass. Config loading must support the two-layer override pattern: `base → stage → variant`.

**Deliverable**: `python -m pytest tests/test_config_loading.py` passes. Repo structure matches Section 2.

### Step 2: Data Pipeline — CIFAR-100 First

Implement the CIFAR-100 data loader first (trivial — it's a `torchvision.datasets.CIFAR100` wrapper). This gives you a working data pipeline immediately with zero download hassle. The loader must conform to the same interface that CASIA and MS1MV2 loaders will use: returns `(image, identity_label)` pairs.

Also implement the CASIA-WebFace loader (reads from a pre-extracted image folder). MS1MV2 uses the same format and can share the loader code.

Write a smoke test that loads a small subset from each dataset and verifies shapes, dtypes, and label range.

**Deliverable**: DataLoader yields correct batches for both CIFAR-100 and CASIA. Augmentations are visually verified on a few samples.

### Step 3: Backbone + Partition Heads

Implement ResNet-50 with the final FC replaced by 3 parallel `nn.Linear(backbone_dim, K)` heads. The backbone forward returns a dict: `{"features": tensor, "partitions": [tensor, tensor, tensor]}`.

Also implement a ResNet-18 variant for CIFAR-100 (same partitioned output interface, smaller backbone). This is the Stage 0 model.

Start with ResNet-50 and ResNet-18. MobileFaceNet is added later (same interface, different architecture).

**Deliverable**: `model(dummy_input)` returns correctly shaped outputs for both backbone sizes. Parameter count matches expectations.

### Step 4: Embedding Assembly + Partition Dropout

Implement `assemble_embedding()` and the `PartitionDropout` module. Write and pass `test_partition_assembly.py` and `test_partition_dropout.py`.

**Deliverable**: All assembly and dropout tests pass.

### Step 5: ArcFace Loss

Implement the ArcFace loss head. This is standard — the only subtlety is that it must handle zero-padded embeddings without producing NaN. Write and pass `test_arcface_loss.py`.

For CIFAR-100 (Stage 0), ArcFace operates on 100 classes rather than face identities. The same head works — just a different `num_classes`.

**Deliverable**: ArcFace loss tests pass, including the zero-padding edge cases.

### Step 6: Training Loop (Variant-Agnostic)

Implement the trainer with the variant hooks stubbed out (auxiliary loss returns 0, all parameters trainable). Run the integration test on synthetic data.

**Deliverable**: `test_training_loop.py` passes — loss decreases over 2 epochs, checkpoints save/load correctly.

### Step 7: Evaluation Pipeline

Implement the evaluator. For Stage 0 (CIFAR-100), evaluation is just top-1 accuracy at each partition configuration — no face recognition benchmarks needed. For Stage 1+ (CASIA, MS1MV2), implement LFW evaluation (simplest benchmark — 6,000 pairs, well-documented protocol). Verify metrics against published results for a known pretrained model if available, or against hand-computed values on synthetic data.

**Deliverable**: `test_evaluation.py` passes. Evaluator correctly runs all 7 partition configs.

### Step 8: Variant A — Orthogonal Partitions

This is the first real variant. Implement:

- `orthogonal.py`: The partition strategy. Adds the orthogonality regularisation loss (Gram matrix of normalised partition outputs, penalise off-diagonal elements).
- `orthogonality.py` in losses: The loss computation itself.
- Learned positional encoding: A small `nn.Embedding(3, K)` whose output is added element-wise to each partition's embedding before assembly.
- `variant_a.yaml`: Config with lambda and encoding type.

Write and pass `test_orthogonality_loss.py`.

**Deliverable**: Full training run launches with Variant A config. Orthogonality loss logged alongside ArcFace loss. Ready for Stage 0 validation.

### Step 9: Stage 0 — CIFAR-100 Smoke Test

**This is the first GPU step.** Train Variant A + a non-partitioned baseline on CIFAR-100 with ResNet-18. Each run takes ~10–15 minutes on a consumer GPU.

The goal is NOT to validate the progressive concept on a classification task (CIFAR-100 is too different from face recognition for that). The goal is to verify the mechanics under real gradient flow:

- Does partition dropout produce a degradation curve (any curve, not necessarily a good one)?
- Does the orthogonality loss converge without fighting the classification loss?
- Does the training loop run end-to-end without crashes, NaN, or memory leaks?
- Are checkpoints saved and loadable correctly?

If any of these fail, fix them before spending hours on CASIA.

**Compute cost**: ~30 minutes total (baseline + Variant A).

**Go/no-go**: Does the code work? Proceed to Stage 1 if yes.

### Step 10: Stage 1 — CASIA-WebFace Variant Comparison

**This is the real concept validation.** Train all four variants + baseline on CASIA-WebFace with ResNet-50. Each run takes ~3–4 GPU-hours on an A100 or equivalent.

Evaluate each variant at all 7 partition configurations on LFW and CFP-FP. Record the degradation curve for each variant.

| Run | Approx. GPU-hours |
|-----|-------------------|
| Baseline (non-partitioned) | 3–4 |
| Variant A (orthogonal) | 3–4 |
| Variant B (nested/slimmable) | 3–4 |
| Variant C (residual boosting) | 4–6 (sequential training phases) |
| Variant D (combined) | 4–6 (phased training) |
| **Total** | **~18–24** |

**Go/no-go criteria**:
- Single-partition accuracy is meaningfully above chance (e.g. LFW > 85% for best single partition, vs ~99%+ for full baseline)
- 3-partition accuracy is within a few percent of the non-partitioned baseline
- The degradation curve is monotonic (more partitions = better accuracy)
- At least one variant shows a smooth, graceful curve rather than a sharp cliff

If these hold, the concept works and one or two winning variants are identified. If the curves are flat or non-monotonic, the architecture needs rethinking before committing more compute.

**Compute cost**: ~20 GPU-hours.

### Step 11: Remaining Experiment 1 Runs on CASIA

Using the winning variant(s) from Step 10, run Experiments 1b (positional encoding ablation), 1c (dropout schedule sweep), and 1d (embedding dimension sweep) — all on CASIA-WebFace.

These are cheaper than the full variant comparison because they test variations of a single variant. Estimate ~3–4 hours per run, ~10 runs total.

**Compute cost**: ~30–40 GPU-hours.

### Step 12: Stage 2 — MS1MV2 Confirmatory Run

Train only the winning variant (with the best config from Steps 10–11) on MS1MV2 at full scale. Evaluate on all benchmarks (LFW, CFP-FP, AgeDB-30, IJB-B, IJB-C). This is the run that goes in the paper.

Optionally, train a second variant if the Step 10 results were close, to confirm the ranking holds at scale.

**Compute cost**: ~25 GPU-hours for one variant, ~50 for two.

### Compute Summary

| Stage | Dataset | What | GPU-hours |
|-------|---------|------|-----------|
| 0 | CIFAR-100 | Mechanics check | ~0.5 |
| 1 | CASIA-WebFace | Full variant comparison (Exp 1a) | ~20 |
| 1 | CASIA-WebFace | Ablations (Exp 1b, 1c, 1d) | ~35 |
| 2 | MS1MV2 | Confirmatory run (winner only) | ~25–50 |
| | | **Total** | **~80–105** |

This is roughly a third of the original ~300 GPU-hour estimate, and the go/no-go decision comes at the 20-hour mark rather than the 50-hour mark.

---

## 7. Key Design Decisions to Lock Down Early

These should be decided before writing code, as they affect everything downstream.

### 7.1 Embedding Normalisation

ArcFace operates on L2-normalised embeddings. When should normalisation happen?

- **Option A**: Normalise each partition output independently, then concatenate. The full embedding is NOT L2-normalised as a whole.
- **Option B**: Concatenate raw partition outputs, then L2-normalise the full 3K vector.
- **Option C**: Both — normalise partitions individually AND normalise the assembled result.

Option B is the standard ArcFace convention and should be the default. But it means zero-padded slots affect the normalisation of non-zero slots (the L2 norm of a partially-filled vector is smaller). This needs explicit testing — does the ArcFace head learn to compensate, or does it degrade single-partition performance?

**Recommendation**: Start with Option B. If single-partition accuracy is unexpectedly low, test Option A as a diagnostic. Log the L2 norm of assembled embeddings at each partition count to monitor this.

### 7.2 Partition Head Architecture

The simplest partition head is a single `nn.Linear`. But face recognition benefits from a BN-FC-BN pattern after the backbone pooling. Options:

- **Minimal**: `nn.Linear(backbone_dim, K)` — fewest parameters, cleanest experiment.
- **Standard**: `BN → Linear(backbone_dim, K) → BN` — matches common ArcFace implementations.
- **Deeper**: `BN → Linear → ReLU → Linear(hidden, K) → BN` — more capacity per partition.

**Recommendation**: Start with Standard (BN-FC-BN). It matches the literature and gives the network enough capacity to learn useful per-partition representations without overcomplicating the architecture.

### 7.3 RecordIO vs Pre-extracted Images

Both CASIA-WebFace and MS1MV2 ship as MXNet RecordIO files. Two options:

- **Read RecordIO directly**: Requires `mxnet` as a dependency. Faster I/O, no disk space duplication.
- **Pre-extract to JPEG/PNG folder**: One-time conversion, then standard `ImageFolder` loader. No `mxnet` dependency at training time.

**Recommendation**: Pre-extract once via `scripts/convert_rec.py`. The mxnet dependency is heavy and sometimes conflicts with PyTorch CUDA versions. Both CASIA and MS1MV2 use the same RecordIO format, so one conversion script handles both. CIFAR-100 needs no conversion (loaded directly by torchvision).

### 7.4 Positional Encoding Dimensionality

For learned positional encodings (Variants A, D), the encoding vector is concatenated with the partition output. This means each partition slot is actually `K + pos_dim` rather than `K`, and the full embedding is `3(K + pos_dim)`.

Alternatively, the positional encoding can be *added* (element-wise) to the K-dim output, keeping the embedding dimension at 3K. This requires `pos_dim = K`.

**Recommendation**: Use addition, not concatenation. It keeps the embedding dimension constant across encoding strategies, making fair comparison easier. The learned embedding is `nn.Embedding(3, K)`, added element-wise to the partition output.

---

## 8. Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Single-partition accuracy too low to be useful | Kills the progressive concept | Partition dropout schedule heavily weights single-partition training (40%). If still poor, increase to 60% and accept lower full-ensemble accuracy. Caught at Stage 1 (CASIA) before committing MS1MV2 compute. |
| Orthogonality loss conflicts with ArcFace loss | Variant A doesn't converge | Start with very low lambda (0.001). If conflict persists, try alternating optimisation (ArcFace steps and orth steps on different batches). Caught at Stage 0 (CIFAR-100) within minutes. |
| MS1MV2 no longer available for download | Can't run final confirmatory stage | Mirror the dataset before starting. Stage 1 (CASIA) provides the variant comparison regardless — MS1MV2 is only needed for the confirmatory paper run. |
| CASIA results don't transfer to MS1MV2 scale | Variant ranking changes at full scale | The risk is real but bounded. Relative degradation curves (the shape, not absolute numbers) should be stable across datasets. If the Stage 2 confirmatory run surprises, run a second variant as insurance (~25 extra GPU-hours). |
| 3K embedding dimension too large for meaningful single-partition results | Weak degradation curve | Sweep K down to 64 (192-dim total) in Experiment 1d. Smaller embeddings may show cleaner progressive behaviour. |
| ArcFace head can't cope with zero-padded variable-length effective inputs | Training instability | Monitor gradient norms per partition. If unstable, try separate ArcFace heads per width config (breaks the shared-head simplicity but may be necessary). Caught at Stage 0. |
| Variant comparison inconclusive (all similar) | Less interesting paper | This is actually a fine outcome — it means the progressive concept is robust to design choices. Emphasise the concept over the specific variant. |
| CIFAR-100 results mislead about face recognition viability | False confidence from toy task | Stage 0 is explicitly NOT a concept validation — only a mechanics check. The go/no-go decision is at Stage 1 on actual face recognition. |

---

## 9. Compute Requirements for Phase 1

**Development phase** (Steps 1–8): All on CPU with synthetic data. Zero GPU cost.

**Staged GPU validation:**

| Stage | Dataset | Purpose | GPU-hours | Go/no-go? |
|-------|---------|---------|-----------|-----------|
| 0 | CIFAR-100 | Mechanics check (code works end-to-end) | ~0.5 | Code works → proceed |
| 1 | CASIA-WebFace | Concept validation (all 4 variants + baseline) | ~20 | Degradation curve is sensible → proceed |
| 1 | CASIA-WebFace | Ablations (Exp 1b, 1c, 1d on winning variant) | ~35 | Best config identified → proceed |
| 2 | MS1MV2 | Confirmatory run (winner only, for the paper) | ~25–50 | Paper-quality numbers |
| | | **Total** | **~80–105** | |

The original estimate of ~300–400 GPU-hours assumed all experiments on MS1MV2. By running the variant comparison and ablation sweeps on CASIA-WebFace and only promoting the winner to MS1MV2, total compute drops by roughly two-thirds. The concept go/no-go comes at the ~20-hour mark.

---

## 10. Quick Reference: Experimental Plan Mapping

| This Document | Experimental Plan Section | Notes |
|---------------|--------------------------|-------|
| Section 4.1 (Backbone) | Section 3.1 | Shared training config |
| Section 4.4 (Partition Dropout) | Section 3.2 | Dropout schedule |
| Section 4.6 (Evaluation) | Section 5.1 | Experiment 1a protocol |
| Step 8 (Variant A) | Section 2.1 | Orthogonal partitions |
| Step 9 (Stage 0) | Section 5 preamble | Not in original plan — added for compute efficiency |
| Step 10 (Stage 1) | Section 5.1 | Variant comparison, now on CASIA-WebFace |
| Step 12 (Stage 2) | Section 5.1 | Confirmatory run on MS1MV2 (winner only) |
| Section 7 (Design Decisions) | Not in exp plan | Implementation-level decisions that need resolving |
