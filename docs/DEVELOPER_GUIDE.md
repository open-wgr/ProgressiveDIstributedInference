# PPI Developer Guide

A practical guide to the Progressive Partitioned Inference codebase. For the research motivation and experimental plan, see [PPI_PROJECT_BOOTSTRAP.md](PPI_PROJECT_BOOTSTRAP.md).

---

## 1. Overview

PPI trains face recognition models whose embeddings are split into **partitions** (default: 3). At inference time, any subset of partitions can be used — more partitions yield higher accuracy, fewer save bandwidth/compute. The system learns to produce useful embeddings at every width through **partition dropout** during training.

Four variant strategies are planned for how partitions relate to each other:

| Variant | Strategy | Status |
|---------|----------|--------|
| A | Orthogonal (regularised independence) | Implemented |
| B | Nested / slimmable | Stub |
| C | Residual boosting (sequential training) | Stub |
| D | Combined (A + C) | Stub |

---

## 2. Quick Start

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run the full test suite (104 tests, ~40s on CPU)
python -m pytest tests/ -v

# Train Variant A on CIFAR-100 (Stage 0 smoke test)
python scripts/train.py --config configs/stage0_cifar100.yaml --variant configs/variant_a.yaml

# Train Variant A on CASIA-WebFace (Stage 1)
python scripts/train.py --config configs/stage1_casia.yaml --variant configs/variant_a.yaml

# Same, with Weights & Biases logging for remote monitoring
python scripts/train.py --config configs/stage1_casia.yaml --variant configs/variant_a.yaml \
    --wandb --wandb-project ppi --wandb-name "stage1-varA-lambda0.1"

# Evaluate a checkpoint — classification (CIFAR-100)
python scripts/evaluate.py --checkpoint runs/<run_dir>/checkpoint_epoch50.pt --config configs/stage0_cifar100.yaml

# Evaluate a checkpoint — LFW pair verification (CASIA-WebFace)
python scripts/evaluate.py --checkpoint runs/<run_dir>/checkpoint_epoch28.pt \
    --config configs/stage1_casia.yaml --variant configs/variant_a.yaml --benchmark lfw
```

---

## 3. Repository Layout

```
ppi/
├── pyproject.toml                          # Build config, dependencies (hatchling)
├── configs/
│   ├── base.yaml                           # Shared defaults (ResNet-50, K=128, MS1MV2)
│   ├── stage0_cifar100.yaml                # CIFAR-100 overrides (ResNet-18, K=64)
│   ├── stage1_casia.yaml                   # CASIA-WebFace overrides
│   ├── stage2_ms1mv2.yaml                  # MS1MV2 (identical to base, explicit)
│   ├── variant_a.yaml                      # Orthogonal: lambda, positional encoding
│   ├── variant_b.yaml                      # Nested (stub)
│   ├── variant_c.yaml                      # Residual (stub)
│   └── variant_d.yaml                      # Combined (stub)
├── src/ppi/
│   ├── backbones/
│   │   ├── __init__.py                     # build_backbone(config) factory
│   │   ├── resnet.py                       # PartitionedResNet (ResNet-18/50, modified stem)
│   │   └── mobilefacenet.py                # Stub
│   ├── heads/
│   │   ├── partition_head.py               # PartitionHead: BN → Linear → BN
│   │   └── arcface.py                      # ArcFaceHead: cosine similarity classifier
│   ├── partitions/
│   │   ├── base.py                         # PartitionStrategy ABC + DefaultStrategy
│   │   ├── orthogonal.py                   # Variant A: positional encoding + orth loss
│   │   ├── nested.py                       # Variant B (stub)
│   │   ├── residual.py                     # Variant C (stub)
│   │   └── combined.py                     # Variant D (stub)
│   ├── losses/
│   │   ├── arcface_loss.py                 # ArcFaceLoss: cos(theta+m) margin + scaling
│   │   └── orthogonality.py                # OrthogonalityLoss: Gram matrix regularisation
│   ├── training/
│   │   ├── trainer.py                      # Trainer: main training loop
│   │   ├── partition_dropout.py            # PartitionDropout + assemble_embedding()
│   │   └── schedulers.py                   # build_scheduler(): warmup + cosine
│   ├── evaluation/
│   │   ├── evaluator.py                    # Evaluator: run all partition configs
│   │   ├── metrics.py                      # TAR@FAR, rank-1, pair accuracy
│   │   └── benchmarks.py                   # LFW pair loader (CFP-FP, AgeDB-30 stubs)
│   ├── data/
│   │   ├── __init__.py                     # build_dataloader(config, split) factory
│   │   ├── cifar100.py                     # CIFAR100Dataset (torchvision wrapper)
│   │   ├── casia.py                        # FaceDataset base + CASIAWebFace
│   │   └── ms1mv2.py                       # MS1MV2 (same interface as CASIA)
│   └── utils/
│       ├── config.py                       # YAML loading, merging, CLI overrides
│       └── logging.py                      # ExperimentLogger: TensorBoard + checkpoints
├── scripts/
│   ├── train.py                            # CLI: --config, --variant, --override
│   ├── evaluate.py                         # CLI: --checkpoint, --config, --partitions
│   ├── sweep.py                            # Hyperparameter sweep (stub)
│   ├── convert_parquet.py                  # HuggingFace Parquet → ImageFolder
│   └── convert_rec.py                      # MXNet RecordIO → ImageFolder (no mxnet needed)
├── docs/
│   ├── PPI_PROJECT_BOOTSTRAP.md            # Research design, experiment plan
│   ├── DEVELOPER_GUIDE.md                  # This file
│   └── DATA_SETUP.md                       # Dataset download & installation guide
├── tests/                                  # 104 tests, all CPU, synthetic data
│   ├── conftest.py                         # Shared fixtures (tiny_config, dummy_batch)
│   ├── test_config_loading.py              # Config merging, validation, overrides
│   ├── test_data_pipeline.py               # Dataset shapes, DataLoader factory
│   ├── test_backbone.py                    # Forward shapes, param counts, factory
│   ├── test_partition_assembly.py          # Slot placement, zero-padding, L2 norm
│   ├── test_partition_dropout.py           # Distribution sampling, eval bypass
│   ├── test_arcface_loss.py                # Finite loss, gradient flow, zero handling
│   ├── test_orthogonality_loss.py          # Orth loss properties, strategy factory
│   ├── test_evaluation.py                  # Metrics on synthetic distributions
│   └── test_training_loop.py               # Integration: 2 epochs, checkpoint round-trip
├── notebooks/
│   └── analysis.ipynb                      # Post-training plotting (placeholder)
└── docs/
    ├── PPI_PROJECT_BOOTSTRAP.md            # Research design document
    └── DEVELOPER_GUIDE.md                  # This file
```

---

## 4. Architecture

### Training Data Flow

```
                         ┌─────────────────────────────────────┐
                         │            Trainer.train()           │
                         └─────────────┬───────────────────────┘
                                       │
  images, labels ──► Backbone (ResNet) ──► feature vector [B, 2048]
                                       │
                         ┌─────────────┴─────────────┐
                         │    3 × PartitionHead       │
                         │  (BN → Linear → BN each)   │
                         └──┬──────────┬──────────┬──┘
                            │          │          │
                         [B,K]      [B,K]      [B,K]
                            │          │          │
                 strategy.process_partitions()      ◄── positional encoding (Variant A)
                            │          │          │
                    PartitionDropout                ◄── stochastic zero-masking
                            │          │          │
                    assemble_embedding()           ◄── cat + L2-normalise → [B, 3K]
                            │
                      ArcFaceHead                  ◄── cosine similarities [B, C]
                            │
                ┌───────────┴───────────┐
                │                       │
          ArcFaceLoss          strategy.compute_auxiliary_loss()
                │                       │
                └───────┬───────────────┘
                        │
                   total_loss.backward()
```

### Evaluation Data Flow

```
  Evaluator.evaluate()
       │
       ├── For each of 7 partition configs ({0}, {1}, {2}, {0,1}, {0,2}, {1,2}, {0,1,2}):
       │       │
       │       ├── Forward through backbone
       │       ├── Zero inactive partition outputs
       │       ├── assemble_embedding()
       │       └── Compute metrics (accuracy / TAR@FAR / rank-1)
       │
       └── Return results dict keyed by config name (P0, P01, P012, etc.)
```

---

## 5. Key Concepts

### 5.1 Partitioned Embeddings

The backbone produces a single feature vector (e.g. 2048-d for ResNet-50). This feeds into `num_partitions` parallel `PartitionHead` modules, each outputting a K-dimensional vector. The full embedding is the concatenation of all partitions: `3K` dimensions total.

```python
# In PartitionedResNet.forward():
features = self.flatten(self.pool(x))          # [B, 2048]
partitions = [head(features) for head in self.partition_heads]  # 3 × [B, K]
```

### 5.2 Partition Dropout

During training, `PartitionDropout` randomly zeros entire partitions to force the network to learn useful representations at every width. The dropout distribution controls how often each width appears:

| Width | Default probability | Meaning |
|-------|-------------------|---------|
| 1 partition | 40% | One random partition active |
| 2 partitions | 30% | Two random partitions active |
| 3 partitions | 20% | All active (no masking) |
| 0 partitions | 10% | All zeros (regularisation) |

The same mask applies to every sample in the batch. Disabled during evaluation.

### 5.3 Embedding Assembly

`assemble_embedding()` concatenates partition outputs and L2-normalises the result. Missing partitions are zero-tensors, so the assembled vector has zeros in those slots. The L2 normalisation uses `eps=1e-12` to handle the all-zeros case without producing NaN.

```python
from ppi.training.partition_dropout import assemble_embedding

# partition_outputs: list of [B, K] tensors (zeros for dropped partitions)
embedding = assemble_embedding(partition_outputs)  # [B, 3K], unit norm
```

### 5.4 ArcFace Pipeline

The ArcFace pipeline is split into two modules:

- **`ArcFaceHead`** (`src/ppi/heads/arcface.py`): Produces cosine similarities between the embedding and a learnable weight matrix. Output is `[B, num_classes]` with values in `[-1, 1]`.
- **`ArcFaceLoss`** (`src/ppi/losses/arcface_loss.py`): Applies the additive angular margin `m` to the target class logit, scales by `s`, then computes cross-entropy.

This separation keeps the head reusable for inference (no loss needed).

### 5.5 Variant Strategy Pattern

Each partition variant implements the `PartitionStrategy` interface from `src/ppi/partitions/base.py`. The trainer calls strategy hooks without knowing which variant is active:

| Hook | When called | Default |
|------|-------------|---------|
| `process_partitions(outputs)` | After backbone, before dropout | Identity |
| `compute_auxiliary_loss(outputs)` | After main loss computation | Returns 0 |
| `pre_training_setup(model, config)` | Before training loop | No-op |
| `get_trainable_parameters(model, phase)` | During optimizer construction | All params |
| `post_epoch_hook(epoch, model)` | End of each epoch | No-op |

The factory `PartitionStrategy.from_config(config)` reads `config["variant"]` and returns the right subclass.

### 5.6 Config Inheritance

Configs use a three-layer override system:

```
base.yaml          ← shared defaults (backbone, training, data)
    ↑ _base_
stage0_cifar100.yaml  ← dataset/model overrides
    ↑ merged on top
variant_a.yaml     ← variant-specific settings
    ↑ merged on top
--override key=val ← CLI overrides
```

The `_base_` key is resolved relative to the file that contains it. Deep merge: dicts are recursively merged, lists are replaced entirely.

---

## 6. Configuration System

### Loading

```python
from ppi.utils.config import load_full_config, apply_overrides

# Load stage config (resolves _base_ automatically)
config = load_full_config("configs/stage0_cifar100.yaml")

# With variant overlay
config = load_full_config("configs/stage0_cifar100.yaml", variant_path="configs/variant_a.yaml")

# With CLI overrides
config = apply_overrides(config, ["training.optimizer.lr=0.01", "training.epochs=10"])
```

### Full Config Schema

```yaml
seed: 42                          # RNG seed (required)

backbone:
  name: resnet50                  # resnet18 | resnet50 | mobilefacenet (required)
  pretrained: false

partitions:
  num_partitions: 3               # Number of partition heads (required)
  K: 128                          # Per-partition embedding dim (required)
  dropout:
    enabled: true
    distribution: [0.4, 0.3, 0.2, 0.1]  # [1-part, 2-part, 3-part, 0-part]

arcface:
  s: 64                           # Scale factor
  m: 0.5                          # Angular margin
  num_classes: null                # Override; inferred from dataset if absent

training:
  epochs: 24                      # (required)
  batch_size: 256
  optimizer:
    type: sgd
    lr: 0.1
    momentum: 0.9
    weight_decay: 5.0e-4
  scheduler:
    type: cosine
    warmup_epochs: 1
  val_interval: 2
  checkpoint_interval: 1

data:
  dataset: ms1mv2                 # cifar100 | casia | ms1mv2 (required)
  root: /data/ms1mv2/
  num_workers: 8
  input_size: 112                 # 32 for CIFAR-100, 112 for face datasets

logging:
  output_dir: runs/
  tensorboard: true
  wandb: false

# Evaluation benchmarks (used by scripts/evaluate.py --benchmark):
evaluation:
  lfw:
    root: data/lfw/
    pairs: data/lfw/pairs.txt

# Variant-specific (only in variant configs):
variant: orthogonal               # orthogonal | nested | residual | combined
orthogonality:
  lambda: 0.1
  mode: correlation               # correlation (default) | cosine (original)
positional_encoding:
  type: learned
```

### Validation

- **Required keys**: `seed`, `backbone.name`, `partitions.num_partitions`, `partitions.K`, `training.epochs`, `data.dataset`
- **Unknown top-level keys**: emit `UserWarning`
- **Type casting in overrides**: `"42"` -> int, `"3.14"` -> float, `"true"` -> bool, `"hello"` -> str

---

## 7. Training Pipeline

`scripts/train.py` loads config and delegates to `Trainer` (`src/ppi/training/trainer.py`).

### Initialization (`Trainer.__init__`)

1. Seed all RNGs (torch, numpy, random, CUDA, cuDNN)
2. Build data loader via `build_dataloader(config, "train")` — also returns `num_classes`
3. Build backbone via `build_backbone(config)` — returns `PartitionedResNet`
4. Build `ArcFaceHead(embedding_dim=num_partitions*K, num_classes)`
5. Build `PartitionDropout` from `config["partitions"]["dropout"]`
6. Build variant strategy via `PartitionStrategy.from_config(config)`
7. Build SGD optimizer over `strategy.get_trainable_parameters(backbone)` + ArcFace head + strategy params
8. Build LR scheduler (warmup + cosine)
9. Create `ExperimentLogger` (run directory, TensorBoard, config copy)

### Per-Batch Forward Pass

```python
out = backbone(images)                                    # {"features": [B, D], "partitions": [3 × [B, K]]}
parts = strategy.process_partitions(out["partitions"])    # e.g. add positional encoding
parts = partition_dropout(parts)                          # stochastic zero-masking
embedding = assemble_embedding(parts)                     # [B, 3K], L2-normalised
cosine = arcface_head(embedding)                          # [B, num_classes]
loss = arcface_loss(cosine, labels)                       # scalar
aux_loss = strategy.compute_auxiliary_loss(out["partitions"])  # e.g. orthogonality
total_loss = loss + aux_loss
total_loss.backward()
optimizer.step()
```

### Checkpointing

Saved to `runs/{variant}_{timestamp}/checkpoint_epoch{N}.pt`:

```python
{
    "epoch": int,
    "model_state_dict": {
        "backbone": backbone.state_dict(),
        "arcface_head": arcface_head.state_dict(),
    },
    "optimizer_state_dict": optimizer.state_dict(),
    "metrics": {"loss": float},
}
```

---

## 8. Evaluation Pipeline

`scripts/evaluate.py` loads a checkpoint and delegates to `Evaluator` (`src/ppi/evaluation/evaluator.py`).

### Partition Configurations

For 3 partitions, there are 7 non-degenerate configurations:

```
Singles: {0}, {1}, {2}
Pairs:   {0,1}, {0,2}, {1,2}
Full:    {0,1,2}
```

Named as `P0`, `P1`, `P2`, `P01`, `P02`, `P12`, `P012` in output.

### Evaluation Modes

**Classification** (`Evaluator.evaluate()`) — used for CIFAR-100 (Stage 0):
- Extracts embeddings for the val set at each partition config
- Builds nearest-centroid classifier, reports top-1 accuracy

```bash
python scripts/evaluate.py --checkpoint CKPT --config configs/stage0_cifar100.yaml
```

**LFW pair verification** (`Evaluator.evaluate_lfw()`) — used for CASIA-WebFace (Stage 1):
- Loads the LFW 6,000-pair protocol from `pairs.txt`
- Extracts embeddings for all referenced images at each partition config
- Reports 10-fold pair accuracy and TAR@FAR=1e-3

```bash
python scripts/evaluate.py --checkpoint CKPT --config configs/stage1_casia.yaml \
    --variant configs/variant_a.yaml --benchmark lfw
```

### Metrics

| Metric | Function | Used for |
|--------|----------|----------|
| Nearest-centroid accuracy | `Evaluator.evaluate()` | CIFAR-100 (Stage 0) |
| Pair accuracy (10-fold) | `Evaluator.evaluate_lfw()` | LFW verification |
| TAR @ FAR=1e-3 | `Evaluator.evaluate_lfw()` | LFW verification |
| TAR @ FAR (general) | `metrics.compute_tar_at_far()` | Face verification |
| Rank-1 | `metrics.compute_rank1()` | Face identification |

### Data Setup

See **[DATA_SETUP.md](DATA_SETUP.md)** for download instructions, expected directory layouts, and verification scripts for all datasets.

---

## 9. Adding a New Variant

Example: implementing Variant B (nested/slimmable).

### Step 1: Create the strategy class

Edit `src/ppi/partitions/nested.py`:

```python
from ppi.partitions.base import PartitionStrategy

class NestedPartitionStrategy(PartitionStrategy):
    def __init__(self, config):
        # Read variant-specific config
        ...

    def process_partitions(self, partition_outputs):
        # Variant-specific processing
        return partition_outputs

    def compute_auxiliary_loss(self, partition_outputs):
        # Return variant-specific loss, or torch.tensor(0.0)
        ...
```

If the strategy has learnable parameters, also inherit from `nn.Module` (see `OrthogonalPartitionStrategy` for the pattern).

### Step 2: Register in the factory

The factory in `src/ppi/partitions/base.py` already has:

```python
if variant == "nested":
    from ppi.partitions.nested import NestedPartitionStrategy
    return NestedPartitionStrategy(config)
```

Just replace the `raise NotImplementedError` in the constructor.

### Step 3: Create/update the variant config

Edit `configs/variant_b.yaml`:

```yaml
variant: nested
# Add variant-specific keys here
```

If adding new top-level config keys, add them to `KNOWN_TOP_LEVEL_KEYS` in `src/ppi/utils/config.py`.

### Step 4: Write tests

Create `tests/test_nested_bn.py` (or similar). Test that:
- The strategy's hooks return correct types
- Auxiliary loss is finite
- `from_config` factory returns the right class
- Variant-specific behaviour is correct

---

## 10. Adding a New Dataset

### Step 1: Create the dataset class

If it's a face dataset (ImageFolder format), subclass `FaceDataset`:

```python
# src/ppi/data/my_dataset.py
from ppi.data.casia import FaceDataset

class MyDataset(FaceDataset):
    def __init__(self, root, train=True, input_size=112):
        super().__init__(root=root, train=train, input_size=input_size)
```

For non-ImageFolder formats, implement the `Dataset` interface directly with a `num_classes` property.

### Step 2: Register in the factory

In `src/ppi/data/__init__.py`, add to `_DATASET_REGISTRY`:

```python
from ppi.data.my_dataset import MyDataset

_DATASET_REGISTRY = {
    ...
    "my_dataset": MyDataset,
}
```

### Step 3: Create a stage config

```yaml
# configs/stage_my_dataset.yaml
_base_: base.yaml

data:
  dataset: my_dataset
  root: /data/my_dataset/
```

---

## 11. Testing

### Running Tests

```bash
# Full suite
python -m pytest tests/ -v

# Single file
python -m pytest tests/test_arcface_loss.py -v

# Single test
python -m pytest tests/test_backbone.py::TestPartitionedResNet::test_resnet18_32 -v
```

### Conventions

- **CPU only**: No GPU required. All tests use small synthetic data.
- **No real datasets**: Tests create temporary image folders or use random tensors.
- **Shared fixtures** in `tests/conftest.py`:
  - `tiny_config` — minimal valid config (ResNet-18, K=8, 10 classes)
  - `dummy_batch` — 4 images at 112x112
  - `dummy_batch_cifar` — 4 images at 32x32

### Test Coverage by Module

| Test file | What it covers |
|-----------|---------------|
| `test_config_loading.py` | YAML loading, `_base_` resolution, merge semantics, CLI overrides, validation |
| `test_data_pipeline.py` | Transform shapes, dataset interfaces, `build_dataloader`, worker seeding |
| `test_backbone.py` | PartitionHead, ArcFaceHead, PartitionedResNet shapes/params, `build_backbone` |
| `test_partition_assembly.py` | `assemble_embedding`: slot placement, zero-padding, all-zeros, L2 norm |
| `test_partition_dropout.py` | Distribution sampling (chi-squared), zero correctness, eval bypass, batch mask |
| `test_arcface_loss.py` | Finite loss, zero-padded inputs, gradient flow, margin effect |
| `test_orthogonality_loss.py` | Orthogonal vs identical inputs, lambda scaling, strategy factory |
| `test_evaluation.py` | TAR@FAR, rank-1, pair accuracy on synthetic data, 7-config generation |
| `test_training_loop.py` | Integration: 2 epochs, loss finite, checkpoint save/load round-trip |

---

## 12. Project Roadmap

### Current Status (Steps 1-8 complete)

- Shared framework: config, data, backbone, heads, losses, dropout, assembly
- Training loop with variant hooks
- Evaluation pipeline with all 7 partition configs
- Variant A (orthogonal): fully implemented with positional encoding + orthogonality loss
- Variants B, C, D: stubs (factory entries exist, constructors raise `NotImplementedError`)
- 104 tests passing

### What's Next

| Step | Task | A100 hours | L4 hours |
|------|------|-----------|----------|
| 9 | **Stage 0**: Train Variant A + baseline on CIFAR-100 (mechanics check) | ~0.5 | ~1 |
| 10 | **Stage 1**: Train all 4 variants + baseline on CASIA-WebFace | ~20 | ~60 |
| 11 | Ablations on winning variant (positional encoding, dropout schedule, embedding dim) | ~35 | ~105 |
| 12 | **Stage 2**: Confirmatory run on MS1MV2 (winner only) | ~25–50 | ~75–150 |

L4 estimates use a ~3× scaling factor based on measured Stage 0 timings.

### Still to Implement

- Variants B, C, D strategy classes
- MobileFaceNet backbone
- `scripts/sweep.py` (hyperparameter sweep launcher)
- CFP-FP, AgeDB-30, IJB-B/C benchmark loaders

See [PPI_PROJECT_BOOTSTRAP.md](PPI_PROJECT_BOOTSTRAP.md) for the full research context, risk register, and compute budget.
