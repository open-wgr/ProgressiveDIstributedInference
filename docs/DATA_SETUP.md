# Data Setup Guide

This document covers how to obtain, install, and verify each dataset used in the PPI project.

## Directory Structure

All datasets live under `data/` in the project root (gitignored). The expected layout after setup:

```
data/
├── cifar100/              # Auto-downloaded by torchvision
│   ├── cifar-100-python/
│   └── ...
├── casia_webface/         # Stage 1 training
│   ├── 0000045/           # One subdirectory per identity
│   │   ├── 001.jpg
│   │   └── ...
│   ├── 0000099/
│   └── ...  (~10,575 identity folders, ~494k images)
├── lfw/                   # Stage 1 evaluation
│   ├── pairs.txt          # 6,000-pair verification protocol
│   ├── Aaron_Eckhart/
│   │   ├── Aaron_Eckhart_0001.jpg
│   │   └── ...
│   ├── Aaron_Guiel/
│   └── ...  (~5,749 identities, ~13,233 images)
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

The `--dataset lfw` flag ensures the LFW naming convention (`{Name}/{Name}_{NNNN}.jpg`) is used, which is required for `pairs.txt` compatibility. The script also looks for `pairs.txt` in the parquet directory and copies it if found.

### Option C: Download from official LFW site

```bash
# Download aligned images
wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
tar -xzf lfw.tgz -C data/

# Download the pairs protocol
wget http://vis-www.cs.umass.edu/lfw/pairs.txt -O data/lfw/pairs.txt
> ```

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

## Image Pre-processing Notes

All face datasets (CASIA, LFW, MS1MV2) should contain **aligned and cropped** images at roughly 112x112 resolution. The data pipeline applies:

- **Training**: `Resize(112)` + `RandomHorizontalFlip` + `Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])`
- **Evaluation**: `Resize(112)` + `Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])`

If your images are not pre-aligned, you'll need to run face detection + alignment first (e.g., using RetinaFace or MTCNN). This is outside the scope of the PPI codebase.

---

## Disk Space Summary

| Dataset | Download | On Disk | Required For |
|---------|----------|---------|--------------|
| CIFAR-100 | ~169 MB | ~346 MB | Stage 0 (mechanics check) |
| CASIA-WebFace | ~4 GB | ~6 GB | Stage 1 (variant comparison) |
| LFW | ~180 MB | ~180 MB | Stage 1 (evaluation) |
| MS1MV2 | ~25 GB | ~40 GB | Stage 2 (confirmatory) |

---

## Troubleshooting

**"Dataset not found" error during training**
- Check that `data.root` in your config points to the directory containing identity subdirectories
- Run the verification snippet above for your dataset

**Images are `.png` but code expects `.jpg`**
- `ImageFolder` from torchvision handles both automatically — no action needed

**"pairs.txt not found" during LFW evaluation**
- Ensure `pairs.txt` is in the LFW root directory
- Check config: `evaluation.lfw.root` should point to the LFW root

**Identity folders have unexpected naming**
- ImageFolder sorts subdirectories alphabetically and assigns class indices
- The actual folder names don't matter as long as one-folder-per-identity is maintained
- Class indices won't match between runs if folders are renamed — this is fine for training, but keep layout consistent for checkpoint compatibility

**LFW image names don't match pairs.txt**
- Our `LFWBenchmark` expects `{Name}/{Name}_{NNNN}.jpg` format (e.g., `Aaron_Eckhart/Aaron_Eckhart_0001.jpg`)
- Some downloads use different naming. Verify by running the LFW verification snippet above
