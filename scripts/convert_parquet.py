"""Convert Hugging Face Parquet datasets to ImageFolder layout.

Many HuggingFace face-recognition datasets are distributed as Parquet files
with images stored as binary blobs. This script extracts them into the
ImageFolder structure expected by our data pipeline (one subdirectory per
identity).

Supports:
  - CASIA-WebFace (training) -> data/casia_webface/{identity_id}/NNNNNN.jpg
  - LFW (evaluation)         -> data/lfw/{Person_Name}/Person_Name_NNNN.jpg
                                 + data/lfw/pairs.txt

Usage:
  # Auto-detect dataset type and column names
  python scripts/convert_parquet.py --input /path/to/parquet_dir --output data/casia_webface/

  # Explicit column mapping
  python scripts/convert_parquet.py --input /path/to/parquet_dir --output data/casia_webface/ \
      --image-col image --label-col label

  # LFW with pairs.txt generation
  python scripts/convert_parquet.py --input /path/to/lfw_parquet --output data/lfw/ --dataset lfw

  # Single parquet file instead of directory
  python scripts/convert_parquet.py --input /path/to/train-00000-of-00004.parquet --output data/casia_webface/
"""

from __future__ import annotations

import argparse
import io
import sys
from collections import defaultdict
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


def _find_parquet_files(input_path: Path) -> list[Path]:
    """Return sorted list of parquet files from a path (file or directory)."""
    if input_path.is_file():
        return [input_path]
    files = sorted(input_path.glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {input_path}")
    return files


def _detect_columns(df, image_col: str | None, label_col: str | None) -> tuple[str, str]:
    """Auto-detect image and label columns if not specified."""
    cols = list(df.columns)
    print(f"  Columns found: {cols}")

    # Image column
    if image_col is None:
        candidates = ["image", "img", "face", "photo", "pixel_values"]
        for c in candidates:
            if c in cols:
                image_col = c
                break
        if image_col is None:
            raise ValueError(
                f"Could not auto-detect image column from {cols}. "
                f"Use --image-col to specify."
            )

    # Label column
    if label_col is None:
        candidates = ["label", "identity", "id", "class", "target", "name", "person"]
        for c in candidates:
            if c in cols:
                label_col = c
                break
        if label_col is None:
            raise ValueError(
                f"Could not auto-detect label column from {cols}. "
                f"Use --label-col to specify."
            )

    print(f"  Using image column: '{image_col}', label column: '{label_col}'")
    return image_col, label_col


def _extract_image_bytes(cell) -> bytes:
    """Extract raw image bytes from various HuggingFace image formats.

    HF datasets store images in several ways:
      - dict with {"bytes": b"...", "path": "..."} (most common)
      - dict with {"bytes": b"..."} only
      - raw bytes
      - PIL Image (if datasets library decoded it)
    """
    if isinstance(cell, dict):
        if "bytes" in cell and cell["bytes"] is not None:
            return cell["bytes"]
        elif "path" in cell and cell["path"] is not None:
            return Path(cell["path"]).read_bytes()
    elif isinstance(cell, bytes):
        return cell
    else:
        # Might be a PIL Image (if loaded via datasets library)
        try:
            from PIL import Image
            if isinstance(cell, Image.Image):
                buf = io.BytesIO()
                cell.save(buf, format="JPEG", quality=95)
                return buf.getvalue()
        except ImportError:
            pass

    raise ValueError(
        f"Cannot extract image bytes from type {type(cell).__name__}. "
        f"Value preview: {str(cell)[:200]}"
    )


def _save_image(image_bytes: bytes, output_path: Path) -> None:
    """Save image bytes to disk, converting to JPEG if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Detect format from magic bytes
    if image_bytes[:3] == b"\xff\xd8\xff":
        # Already JPEG
        output_path.write_bytes(image_bytes)
    elif image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        # PNG -> convert to JPEG to save space
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.save(output_path, format="JPEG", quality=95)
    else:
        # Unknown format, try PIL
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img.save(output_path, format="JPEG", quality=95)
        except Exception:
            # Last resort: save as-is
            output_path.write_bytes(image_bytes)


def convert_training_dataset(
    parquet_files: list[Path],
    output_dir: Path,
    image_col: str | None,
    label_col: str | None,
) -> dict[str, int]:
    """Convert training parquet files to ImageFolder layout.

    Returns dict of stats.
    """
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)

    # Track per-identity image counters for sequential naming
    identity_counters: dict[str, int] = defaultdict(int)
    total_images = 0
    total_identities = set()

    for file_idx, pf in enumerate(parquet_files):
        print(f"\nProcessing {pf.name} ({file_idx + 1}/{len(parquet_files)})...")
        df = pd.read_parquet(pf)

        if file_idx == 0:
            image_col, label_col = _detect_columns(df, image_col, label_col)

        for row_idx in range(len(df)):
            label = df.iloc[row_idx][label_col]
            image_cell = df.iloc[row_idx][image_col]

            # Normalise label to string directory name
            label_str = str(label).strip()
            # Sanitise for filesystem (replace problematic chars)
            label_str = label_str.replace("/", "_").replace("\\", "_").replace(" ", "_")

            identity_counters[label_str] += 1
            total_identities.add(label_str)
            img_num = identity_counters[label_str]

            image_bytes = _extract_image_bytes(image_cell)
            out_path = output_dir / label_str / f"{img_num:06d}.jpg"
            _save_image(image_bytes, out_path)

            total_images += 1
            if total_images % 10000 == 0:
                print(f"  {total_images} images extracted ({len(total_identities)} identities)...")

    return {
        "total_images": total_images,
        "total_identities": len(total_identities),
    }


def _parse_lfw_identity(filename: str) -> tuple[str, str]:
    """Extract identity name and original filename from an LFW path.

    Handles various filename formats:
      - "Aaron_Eckhart/Aaron_Eckhart_0001.jpg" -> ("Aaron_Eckhart", "Aaron_Eckhart_0001.jpg")
      - "Aaron_Eckhart_0001.jpg"               -> ("Aaron_Eckhart", "Aaron_Eckhart_0001.jpg")
      - "Aaron Eckhart/Aaron Eckhart_0001.jpg"  -> ("Aaron_Eckhart", "Aaron_Eckhart_0001.jpg")

    The identity is everything before the last _NNNN.ext suffix.
    """
    filename = filename.strip().replace("\\", "/")

    # If there's a directory component, use it as the identity
    if "/" in filename:
        parts = filename.rsplit("/", 1)
        identity = parts[0].replace(" ", "_")
        basename = parts[1].replace(" ", "_")
        return identity, basename

    # No directory — parse identity from filename (everything before _NNNN.ext)
    basename = filename.replace(" ", "_")
    stem = Path(basename).stem  # e.g. "Aaron_Eckhart_0001"
    # Find the last _NNNN suffix
    last_underscore = stem.rfind("_")
    if last_underscore > 0 and stem[last_underscore + 1:].isdigit():
        identity = stem[:last_underscore]
    else:
        identity = stem

    return identity, basename


def convert_lfw(
    parquet_files: list[Path],
    output_dir: Path,
    image_col: str | None,
    label_col: str | None,
) -> dict[str, int]:
    """Convert LFW parquet files to ImageFolder layout + generate pairs.txt.

    LFW uses a specific naming convention: {Name}/{Name}_{NNNN}.jpg

    If a ``filename`` column exists, the original filenames are preserved
    (critical for ``pairs.txt`` compatibility). The identity is parsed from
    the filename. A ``--label-col`` override is only needed if neither a
    standard label column nor ``filename`` is present.
    """
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)

    identities_seen: set[str] = set()
    total_images = 0
    filename_col: str | None = None  # detected on first file

    for file_idx, pf in enumerate(parquet_files):
        print(f"\nProcessing {pf.name} ({file_idx + 1}/{len(parquet_files)})...")
        df = pd.read_parquet(pf)

        if file_idx == 0:
            cols = list(df.columns)
            print(f"  Columns found: {cols}")

            # Detect image column
            if image_col is None:
                for c in ["image", "img", "face", "photo", "pixel_values"]:
                    if c in cols:
                        image_col = c
                        break
                if image_col is None:
                    raise ValueError(
                        f"Could not auto-detect image column from {cols}. "
                        f"Use --image-col to specify."
                    )

            # For LFW, prefer using the filename column to preserve original names
            if "filename" in cols:
                filename_col = "filename"
                print(f"  Using image column: '{image_col}', "
                      f"identity parsed from filename column: '{filename_col}'")
            elif "file_name" in cols:
                filename_col = "file_name"
                print(f"  Using image column: '{image_col}', "
                      f"identity parsed from filename column: '{filename_col}'")
            else:
                # Fall back to label column detection
                _, label_col = _detect_columns(df, image_col, label_col)
                print(f"  No filename column found — using label column '{label_col}' "
                      f"(image numbering will be sequential, may not match pairs.txt)")

        for row_idx in range(len(df)):
            image_cell = df.iloc[row_idx][image_col]
            image_bytes = _extract_image_bytes(image_cell)

            if filename_col is not None:
                # Preserve original LFW filename for pairs.txt compatibility
                raw_filename = str(df.iloc[row_idx][filename_col])
                identity, basename = _parse_lfw_identity(raw_filename)
                out_path = output_dir / identity / basename
            else:
                # Fallback: sequential numbering from label column
                label = df.iloc[row_idx][label_col]
                identity = str(label).strip().replace(" ", "_")
                # Count images per identity for sequential naming
                existing = list((output_dir / identity).glob("*.jpg")) if (output_dir / identity).exists() else []
                img_num = len(existing) + 1
                out_path = output_dir / identity / f"{identity}_{img_num:04d}.jpg"

            identities_seen.add(identity)
            _save_image(image_bytes, out_path)

            total_images += 1
            if total_images % 5000 == 0:
                print(f"  {total_images} images extracted ({len(identities_seen)} identities)...")

    # Check for existing pairs.txt
    pairs_path = output_dir / "pairs.txt"
    if pairs_path.exists():
        print(f"\npairs.txt already exists at {pairs_path} — keeping it.")
    else:
        # Check if pairs.txt was in the parquet directory
        for pf in parquet_files:
            candidate = pf.parent / "pairs.txt"
            if candidate.exists():
                import shutil
                shutil.copy2(candidate, pairs_path)
                print(f"\nCopied pairs.txt from {candidate}")
                break
        else:
            print(f"\nWARNING: No pairs.txt found. You need to download it separately:")
            print(f"  curl -L -o {pairs_path} https://ndownloader.figshare.com/files/5976006")
            print(f"  (Figshare mirror — the original UMass URL is no longer available)")

    return {
        "total_images": total_images,
        "total_identities": len(identities_seen),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace Parquet datasets to ImageFolder layout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CASIA-WebFace
  python scripts/convert_parquet.py --input data/casia_parquet/ --output data/casia_webface/

  # LFW
  python scripts/convert_parquet.py --input data/lfw_parquet/ --output data/lfw/ --dataset lfw

  # Explicit column names
  python scripts/convert_parquet.py --input data/ --output data/casia_webface/ \\
      --image-col image --label-col label
        """,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to directory containing .parquet files (or a single .parquet file)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory (will be created if needed)",
    )
    parser.add_argument(
        "--dataset", default="training", choices=["training", "lfw"],
        help="Dataset type: 'training' for CASIA/MS1MV2 (default), 'lfw' for LFW benchmark",
    )
    parser.add_argument(
        "--image-col", default=None,
        help="Name of the image column (auto-detected if omitted)",
    )
    parser.add_argument(
        "--label-col", default=None,
        help="Name of the label/identity column (auto-detected if omitted)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    try:
        import pandas  # noqa: F401
    except ImportError:
        print("pandas is required for Parquet conversion.")
        print("Install with: pip install pandas pyarrow")
        sys.exit(1)

    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print(f"Mode:   {args.dataset}")

    parquet_files = _find_parquet_files(input_path)
    print(f"Found {len(parquet_files)} parquet file(s)")

    if args.dataset == "lfw":
        stats = convert_lfw(parquet_files, output_dir, args.image_col, args.label_col)
    else:
        stats = convert_training_dataset(parquet_files, output_dir, args.image_col, args.label_col)

    print(f"\nDone!")
    print(f"  Images:     {stats['total_images']}")
    print(f"  Identities: {stats['total_identities']}")
    print(f"  Output:     {output_dir}")

    # Quick sanity check
    identity_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    if identity_dirs:
        sample = identity_dirs[0]
        sample_images = list(sample.glob("*.jpg"))
        print(f"\n  Sample identity: {sample.name}/ ({len(sample_images)} images)")


if __name__ == "__main__":
    main()
