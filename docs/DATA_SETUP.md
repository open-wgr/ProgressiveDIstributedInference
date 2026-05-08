# Data Setup Guide

This document covers how to obtain, install, and verify each dataset used in the PPI project.

## Directory Structure

All datasets live under `data/` in the project root (gitignored). The expected layout after setup:

```
data/
├── cifar100/              # Auto-downloaded by torchvision
│   ├── cifar-100-python/
│   └── ...
├── casia_webface/         # Stage 1 + Direction 2 training
│   ├── 0000045/           # One subdirectory per identity
│   │   ├── 001.jpg
│   │   └── ...
│   ├── 0000099/
│   └── ...  (~10,575 identity folders, ~494k images)
├── lfw/                   # Stage 1 + Direction 2 evaluation
│   ├── pairs.txt          # 6,000-pair verification protocol
│   ├── Aaron_Eckhart/
│   │   ├── Aaron_Eckhart_0001.jpg
│   │   └── ...
│   ├── Aaron_Guiel/
│   └── ...  (~5,749 identities, ~13,233 images)
├── cfp_fp/                # Direction 2 evaluation (frontal-profile)
│   ├── Data/Images/001/frontal/01.jpg
│   ├── Data/Images/001/profile/01.jpg
│   └── Protocol/Frontal-Profile/split1/same.txt  ...
├── agedb/                 # Direction 2 evaluation (age gap)
│   ├── pairs.txt          # 6,000-pair protocol
│   ├── Aaron_Eckhart_36.jpg
│   └── ...  (flat directory, all images at root)
└── ms1mv2/                # Stage 2 training (future)
    ├── 000000/
    └── ...  (~85k identity folders, ~5.8M images)
```

---

## CIFAR-100 (Stage 0)

**Purpose**: Mechanics check — verify training pipeline works before committing GPU hours.

**Setup**: Fully automatic. The first run downloads to `data/cifar100/` via torchvision:

```bash
python scripts/train.py --config configs/stage0_cifar100.yaml --variant configs/variant_a.yaml
```

**Size**: ~169 MB download, ~346 MB on disk.

**Verification**:

```bash
python -c "
from torchvision.datasets import CIFAR100
ds = CIFAR100('data/cifar100', train=True, download=True)
print(f'Training images: {len(ds)}')  # 50,000
ds_val = CIFAR100('data/cifar100', train=False)
print(f'Validation images: {len(ds_val)}')  # 10,000
"
```

---

## CASIA-WebFace (Stage 1 — Training)

**Purpose**: Primary training dataset for variant comparison. ~500k face images across ~10.5k identities.

### Download from Hugging Face

CASIA-WebFace is available on Hugging Face (search for `CASIA-WebFace` — several community uploads exist with the aligned/cropped 112x112 images).

```bash
# Option 1: Using the huggingface_hub Python API (recommended)
pip install huggingface-hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='<repo-id>',
    repo_type='dataset',
    local_dir='data/casia_webface/',
)
"

# Option 2: Using the huggingface-cli (if installed)
huggingface-cli download <repo-id> --repo-type dataset --local-dir data/casia_webface/

# Option 3: Using git-lfs
git lfs install
git clone https://huggingface.co/datasets/<repo-id> data/casia_webface/
```

> **Note**: Replace `<repo-id>` with the actual Hugging Face dataset repository ID (e.g. `user/CASIA-WebFace`). Some HF repos provide images inside a nested directory (e.g., `CASIA-WebFace/`). The `data.root` config key should point to the directory that directly contains the identity subdirectories.

### Expected Layout

The dataset must be in **ImageFolder** format — one subdirectory per identity:

```
data/casia_webface/
├── 0000045/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── 0000099/
│   ├── 001.jpg
│   └── ...
└── ...
```

### If Images Arrive as a Zip/Tar

```bash
# Extract and check
tar -xf casia_webface.tar.gz -C data/casia_webface/
# Or
unzip casia_webface.zip -d data/casia_webface/

# If there's a nested directory, flatten it:
# mv data/casia_webface/CASIA-WebFace/* data/casia_webface/
# rmdir data/casia_webface/CASIA-WebFace
```

### If Data Arrives as Hugging Face Parquet

Many HuggingFace dataset repos distribute images as Parquet files (binary blobs in columns) rather than as image files. You'll know this is the case if your download contains files like `train-00000-of-00004.parquet` instead of image directories.

```bash
# Install dependencies
pip install pandas pyarrow Pillow

# Convert — auto-detects image/label columns
python scripts/convert_parquet.py --input data/casia_parquet/ --output data/casia_webface/

# If auto-detection fails, specify columns explicitly
python scripts/convert_parquet.py --input data/casia_parquet/ --output data/casia_webface/ \
    --image-col image --label-col label
```

The script prints detected column names and progress every 10k images. CASIA-WebFace takes ~5–10 minutes to convert.

### If Data Arrives as MXNet RecordIO

Some sources provide CASIA-WebFace as `.rec` / `.idx` files. Our converter parses the binary format directly — **no mxnet package required** (avoiding the `numpy.bool` compatibility issues on Python 3.12+).

```bash
python scripts/convert_rec.py --input data/casia_webface/train.rec --output data/casia_webface/
```

The `.idx` file is auto-detected if it sits next to the `.rec` file. Without it, the script does a sequential scan (slightly slower but works fine).

### Verification

```bash
python -c "
from pathlib import Path
root = Path('data/casia_webface')
identities = sorted([d for d in root.iterdir() if d.is_dir()])
total_images = sum(len(list(d.glob('*.jpg'))) + len(list(d.glob('*.png'))) for d in identities)
print(f'Identities: {len(identities)}')     # ~10,575
print(f'Total images: {total_images}')       # ~494,414
print(f'First 5 IDs: {[d.name for d in identities[:5]]}')
"
```

### Config

Training config is already set up in `configs/stage1_casia.yaml`. Adjust `data.root` if your path differs:

```bash
# Default path
python scripts/train.py --config configs/stage1_casia.yaml --variant configs/variant_a.yaml

# Custom path
python scripts/train.py --config configs/stage1_casia.yaml --variant configs/variant_a.yaml \
    --override data.root=/path/to/your/casia_webface/
```

---

## LFW — Labeled Faces in the Wild (Stage 1 — Evaluation)

**Purpose**: Standard face verification benchmark. 6,000 pairs (3,000 genuine + 3,000 impostor) with 10-fold cross-validation.

### Option A: Download from Hugging Face (images)

```bash
pip install huggingface-hub  # if not already installed
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='<repo-id>',
    repo_type='dataset',
    local_dir='data/lfw/',
)
"
```

> **Note**: If the extracted images end up in `data/lfw/lfw/`, move them up one level:
> ```bash
> mv data/lfw/lfw/* data/lfw/ && rmdir data/lfw/lfw

### Option B: Convert from Hugging Face Parquet

If your HuggingFace download gave you `.parquet` files instead of images:

```bash
pip install pandas pyarrow Pillow
python scripts/convert_parquet.py --input data/lfw_parquet/ --output data/lfw/ --dataset lfw
```

The `--dataset lfw` flag preserves original filenames from the `filename` column, which is required for `pairs.txt` compatibility. The script also looks for `pairs.txt` in the parquet directory and copies it if found.

> **Note**: HuggingFace LFW downloads typically do **not** include `pairs.txt`. See [Downloading pairs.txt](#downloading-pairstxt) below.

### Downloading pairs.txt

The original UMass hosting is no longer available. Use the Figshare mirror (same file, used by scikit-learn):

```bash
curl -L -o data/lfw/pairs.txt https://ndownloader.figshare.com/files/5976006
```

Other Figshare mirrors for related protocol files:
- `pairsDevTrain.txt`: `https://ndownloader.figshare.com/files/5976012`
- `pairsDevTest.txt`: `https://ndownloader.figshare.com/files/5976009`

### Expected Layout

```
data/lfw/
├── pairs.txt              # Verification protocol (6,000 pairs)
├── Aaron_Eckhart/
│   ├── Aaron_Eckhart_0001.jpg
│   └── ...
├── Aaron_Guiel/
│   ├── Aaron_Guiel_0001.jpg
│   └── ...
└── ...
```

**Important**: The `pairs.txt` file must be at the root of the LFW directory. The images follow the naming convention `{Name}/{Name}_{NNNN}.jpg` (1-indexed, zero-padded to 4 digits).

### pairs.txt Format

```
10	300                          # 10 folds, 300 pairs per fold
Abel_Pacheco	1	4              # Same person: name, img_idx1, img_idx2
Akhmed_Zakayev	1	3
...
Abdel_Madi_Shabneh	1	Dean_Barker	1    # Different: name1, idx1, name2, idx2
Abdel_Nasser_Assidi	1	Sidney_Poitier	1
...
```

### Verification

```bash
python -c "
from pathlib import Path
root = Path('data/lfw')

# Check pairs.txt
pairs_file = root / 'pairs.txt'
assert pairs_file.exists(), 'pairs.txt not found!'
with open(pairs_file) as f:
    lines = f.readlines()
print(f'Pair file lines: {len(lines)}')  # 6,001 (1 header + 6,000 pairs)

# Check images
identities = sorted([d for d in root.iterdir() if d.is_dir()])
total_images = sum(len(list(d.glob('*.jpg'))) for d in identities)
print(f'Identities: {len(identities)}')     # ~5,749
print(f'Total images: {total_images}')       # ~13,233

# Verify our benchmark loader can parse the pairs
import sys; sys.path.insert(0, 'src')
from ppi.evaluation.benchmarks import LFWBenchmark
paths1, paths2, issame = LFWBenchmark(root).load_pairs()
print(f'Pairs loaded: {len(paths1)}')        # 6,000
print(f'Genuine pairs: {issame.sum()}')       # 3,000
print(f'Impostor pairs: {(~issame).sum()}')   # 3,000

# Spot-check that referenced images exist
import random
random.seed(42)
for i in random.sample(range(len(paths1)), 10):
    assert (root / paths1[i]).exists(), f'Missing: {paths1[i]}'
    assert (root / paths2[i]).exists(), f'Missing: {paths2[i]}'
print('Spot check passed: all sampled images exist')
"
```

### Running LFW Evaluation

```bash
# After training on CASIA-WebFace:
python scripts/evaluate.py \
    --checkpoint runs/<run_dir>/checkpoint_epoch28.pt \
    --config configs/stage1_casia.yaml \
    --variant configs/variant_a.yaml \
    --benchmark lfw

# Output:
# === LFW Pair Verification ===
#   P0:   accuracy=0.XXXX +/- 0.XXXX  TAR@FAR=1e-3=0.XXXX
#   P1:   accuracy=0.XXXX +/- 0.XXXX  TAR@FAR=1e-3=0.XXXX
#   ...
#   P012: accuracy=0.XXXX +/- 0.XXXX  TAR@FAR=1e-3=0.XXXX
```

---

## MS1MV2 (Stage 2 — Future)

**Purpose**: Large-scale confirmatory training. ~5.8M images, ~85k identities.

This dataset is used in Step 12 only — after the winning variant is identified on CASIA-WebFace. Setup instructions will be added when needed.

**Expected format**: Same ImageFolder layout as CASIA-WebFace. Config: `configs/stage2_ms1mv2.yaml`.

---

## Direction 2 — Boosting Data Setup

Direction 2 uses a different training script (`scripts/train_boosting.py`) and config (`configs/direction_2_base.yaml`). The data requirements are otherwise the same as Stage 1 — CASIA-WebFace for training, LFW for evaluation.

### Dataset options

`direction_2_base.yaml` supports three dataset values via `--dataset`:

| `--dataset` | What it loads | Use case |
|-------------|--------------|----------|
| `cifar100` | CIFAR-100 (auto-downloaded) | Smoke test — verifies boosting pipeline works before committing GPU hours |
| `casia_subset` | Random 2 000-identity subset of CASIA-WebFace | Fast iteration on CASIA without full 10k-identity cost |
| `casia` | Full CASIA-WebFace (~10.5k identities) | Full runs |

The `casia_subset` loader samples identities with a fixed seed (42 by default) and remaps labels to 0..N-1, so it is fully reproducible.

### Config data root

`configs/direction_2_base.yaml` defaults to `data/casia_webface/` (the same layout as Stage 1). If your CASIA-WebFace is elsewhere, edit `data.root` in the config directly — `train_boosting.py` does **not** expose a `--data-root` CLI flag (unlike `train.py`'s `--override`).

```yaml
# configs/direction_2_base.yaml
data:
  dataset: casia          # cifar100 | casia_subset | casia
  root: data/casia_webface/
```

### Running

```bash
# Smoke test (CIFAR-100, auto-downloaded)
python scripts/train_boosting.py \
    --config configs/direction_2_base.yaml \
    --dataset cifar100

# Fast CASIA subset (2k identities)
python scripts/train_boosting.py \
    --config configs/direction_2_base.yaml \
    --dataset casia_subset

# Full CASIA run
python scripts/train_boosting.py \
    --config configs/direction_2_base.yaml \
    --dataset casia

# Eval only (skip training, load existing checkpoint)
python scripts/train_boosting.py \
    --config configs/direction_2_base.yaml \
    --eval-only runs/direction_2/<run_dir>/
```

LFW evaluation runs automatically after training if `data/lfw/` and `data/lfw/pairs.txt` exist (see [LFW section](#lfw--labeled-faces-in-the-wild-stage-1--evaluation) above).

### Checkpoints

Phase checkpoints are written under `runs/direction_2/<run_name>/`:

```
runs/direction_2/<run_name>/
├── phase_0/
│   ├── backbone.pt
│   └── partition_0.pt
├── phase_1/
│   ├── backbone.pt
│   └── partition_1.pt
└── phase_2/
    ├── backbone.pt
    └── partition_2.pt
```

Each phase saves independently, so a crash in Phase 2 does not lose Phase 0/1 checkpoints.

---

## CFP-FP — Celebrities in Frontal-Profile (Direction 2 — Evaluation)

**Purpose**: Frontal-to-profile face verification. 7,000 pairs (3,500 genuine + 3,500 impostor), 10-fold cross-validation. Tests robustness to pose variation — a harder benchmark than LFW.

### Download

CFP-FP is available from the authors at [cfp-dataset.com](http://www.cfpw.io). Download the zip and extract to `data/cfp_fp/`:

```bash
unzip cfp-dataset.zip -d data/cfp_fp/

# The zip may land inside a nested folder — check and flatten if needed:
# ls data/cfp_fp/
# mv data/cfp_fp/cfp-dataset/* data/cfp_fp/ && rmdir data/cfp_fp/cfp-dataset
```

### Expected Layout

The loader reads the native CFP split format — **no conversion needed**:

```
data/cfp_fp/
├── Data/
│   └── Images/
│       ├── 001/
│       │   ├── frontal/
│       │   │   ├── 01.jpg ... 10.jpg   (10 frontal images per subject)
│       │   └── profile/
│       │       ├── 01.jpg ... 04.jpg   (4 profile images per subject)
│       ├── 002/
│       └── ...  (500 subjects total)
└── Protocol/
    └── Frontal-Profile/
        ├── split1/
        │   ├── same.txt   # genuine pairs (CSV: person_id,frontal_idx,profile_idx)
        │   └── diff.txt   # impostor pairs (CSV: id1,front_idx1,id2,prof_idx2)
        ├── split2/
        ...
        └── split10/
```

### pairs.txt format

`same.txt` — each line is a genuine pair (1-indexed):
```
1,1,1       # subject 001, frontal 01.jpg vs profile 01.jpg
1,2,1       # subject 001, frontal 02.jpg vs profile 01.jpg
```

`diff.txt` — each line is an impostor pair:
```
1,1,2,1     # subject 001 frontal 01.jpg vs subject 002 profile 01.jpg
```

### Verification

```bash
python -c "
from pathlib import Path
import sys; sys.path.insert(0, 'src')
from ppi.evaluation.benchmarks import CFPFPBenchmark

root = Path('data/cfp_fp')
paths1, paths2, issame = CFPFPBenchmark(root).load_pairs()
print(f'Pairs loaded:    {len(paths1)}')        # 7,000
print(f'Genuine pairs:   {issame.sum()}')        # 3,500
print(f'Impostor pairs:  {(~issame).sum()}')     # 3,500

# Spot-check that referenced images exist
import random; random.seed(42)
for i in random.sample(range(len(paths1)), 10):
    assert (root / paths1[i]).exists(), f'Missing: {paths1[i]}'
    assert (root / paths2[i]).exists(), f'Missing: {paths2[i]}'
print('Spot check passed: all sampled images exist')
"
```

### Config

Add `data/cfp_fp/` to `configs/direction_2_base.yaml` (already set as default):

```yaml
evaluation:
  cfp_fp:
    root: data/cfp_fp/
```

CFP-FP evaluation runs automatically alongside LFW after `train_boosting.py` completes, if the directory exists.

---

## AgeDB-30 — Age Database (Direction 2 — Evaluation)

**Purpose**: Age-gap face verification. 6,000 pairs (3,000 genuine + 3,000 impostor), 10-fold cross-validation. Each genuine pair has an age gap of ≤ 30 years — tests robustness to ageing.

### Download

AgeDB-30 requires signing the authors' license at [ibug.doc.ic.ac.uk/resources/agedb](https://ibug.doc.ic.ac.uk/resources/agedb/). After approval, extract to `data/agedb/`:

```bash
unzip agedb_30.zip -d data/agedb/

# Verify the flat image layout — images should be at data/agedb/*.jpg, not nested
ls data/agedb/*.jpg | head -5
# Expected: data/agedb/Aaron_Eckhart_36.jpg, Aaron_Eckhart_54.jpg, ...
```

> **Note**: The dataset is also available via some Hugging Face community uploads and via the InsightFace data collection — search for "AgeDB-30". The prepared download typically includes `pairs.txt` already.

### Expected Layout

Images are in a **flat directory** (no per-identity subdirs). Filenames follow `{name}_{age}.jpg`:

```
data/agedb/
├── pairs.txt                   # 6,000-pair protocol
├── Aaron_Eckhart_36.jpg
├── Aaron_Eckhart_54.jpg
├── Aamir_Khan_31.jpg
└── ...  (~16,488 images)
```

### pairs.txt Format

Space-separated, no header. Each line: `img1_filename img2_filename label`:

```
Aaron_Eckhart_36.jpg Aaron_Eckhart_54.jpg 1    # genuine (same person)
Aaron_Eckhart_36.jpg Aamir_Khan_31.jpg 0       # impostor (different person)
```

### Verification

```bash
python -c "
from pathlib import Path
import sys; sys.path.insert(0, 'src')
from ppi.evaluation.benchmarks import AgeDB30Benchmark

root = Path('data/agedb')
paths1, paths2, issame = AgeDB30Benchmark(root).load_pairs()
print(f'Pairs loaded:    {len(paths1)}')        # 6,000
print(f'Genuine pairs:   {issame.sum()}')        # 3,000
print(f'Impostor pairs:  {(~issame).sum()}')     # 3,000

# Spot-check that referenced images exist
import random; random.seed(42)
for i in random.sample(range(len(paths1)), 10):
    assert (root / paths1[i]).exists(), f'Missing: {paths1[i]}'
    assert (root / paths2[i]).exists(), f'Missing: {paths2[i]}'
print('Spot check passed: all sampled images exist')
"
```

### Config

Add `data/agedb/` to `configs/direction_2_base.yaml` (already set as default):

```yaml
evaluation:
  agedb:
    root: data/agedb/
```

AgeDB-30 evaluation runs automatically alongside LFW and CFP-FP after `train_boosting.py` completes, if the directory exists.

---

## Image Pre-processing Notes

All face datasets (CASIA, LFW, MS1MV2) should contain **aligned and cropped** images at roughly 112x112 resolution. The data pipeline applies:

- **Training**: `Resize(112)` + `RandomHorizontalFlip` + `Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])`
- **Evaluation**: `Resize(112)` + `Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])`

If your images are not pre-aligned, you'll need to run face detection + alignment first (e.g., using RetinaFace or MTCNN). This is outside the scope of the PPI codebase.

---

## Disk Space Summary

| Dataset | Download | On Disk | Required For |
|---------|----------|---------|--------------|
| CIFAR-100 | ~169 MB | ~346 MB | Stage 0 + Direction 2 smoke test |
| CASIA-WebFace | ~4 GB | ~6 GB | Stage 1 (variant comparison) + Direction 2 |
| LFW | ~180 MB | ~180 MB | Stage 1 + Direction 2 (evaluation) |
| CFP-FP | ~3 GB | ~3 GB | Direction 2 (evaluation — pose) |
| AgeDB-30 | ~1 GB | ~1 GB | Direction 2 (evaluation — age gap) |
| MS1MV2 | ~25 GB | ~40 GB | Stage 2 (confirmatory) |

---

## Troubleshooting

**"Dataset not found" error during training**
- Check that `data.root` in your config points to the directory containing identity subdirectories
- Run the verification snippet above for your dataset

**Images are `.png` but code expects `.jpg`**
- `ImageFolder` from torchvision handles both automatically — no action needed

**"pairs.txt not found" during LFW evaluation**
- Ensure `pairs.txt` is in the LFW root directory: `curl -L -o data/lfw/pairs.txt https://ndownloader.figshare.com/files/5976006`
- Check config: `evaluation.lfw.root` should point to the LFW root
- HuggingFace parquet downloads do **not** include `pairs.txt` — you must download it separately

**Identity folders have unexpected naming**
- ImageFolder sorts subdirectories alphabetically and assigns class indices
- The actual folder names don't matter as long as one-folder-per-identity is maintained
- Class indices won't match between runs if folders are renamed — this is fine for training, but keep layout consistent for checkpoint compatibility

**LFW image names don't match pairs.txt**
- Our `LFWBenchmark` expects `{Name}/{Name}_{NNNN}.jpg` format (e.g., `Aaron_Eckhart/Aaron_Eckhart_0001.jpg`)
- Some downloads use different naming. Verify by running the LFW verification snippet above

**Direction 2 training can't find CASIA ("No such file or directory: /data/casia/")**
- This was caused by the old `direction_2_base.yaml` default (`root: /data/casia/`), which has since been corrected to `root: data/casia_webface/`
- If you cloned before this fix, update the config: `data.root: data/casia_webface/`

**`train_boosting.py` ignores my `--override` flag**
- Unlike `train.py`, `train_boosting.py` does not support `--override`. Edit `data.root` in `configs/direction_2_base.yaml` directly, or copy the config and pass the modified copy via `--config`.

**CFP-FP: "protocol directory not found"**
- The loader expects `{root}/Protocol/Frontal-Profile/split1/` etc. If you extracted to a nested folder, flatten it: `mv data/cfp_fp/cfp-dataset/* data/cfp_fp/ && rmdir data/cfp_fp/cfp-dataset`
- Run the verification snippet to confirm the layout before training.

**AgeDB-30: images not found at runtime**
- Images must be in the **flat root** — not inside a subdirectory. If your download has them under `data/agedb/images/`, move them up: `mv data/agedb/images/* data/agedb/`
- The filenames in `pairs.txt` must match the flat filenames exactly (e.g. `Aaron_Eckhart_36.jpg`, not `Aaron_Eckhart/Aaron_Eckhart_36.jpg`).

**CFP-FP or AgeDB-30 skipped silently**
- `train_boosting.py` skips any benchmark whose `root` directory does not exist — no error is raised. If you expect the benchmark to run but see no output for it, check that the path exists: `ls data/cfp_fp/` / `ls data/agedb/`.
