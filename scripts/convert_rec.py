"""Convert MXNet RecordIO (.rec/.idx) datasets to ImageFolder layout.

Parses the RecordIO binary format directly — does NOT require the mxnet
package (which has numpy.bool compatibility issues on Python 3.12+).

The .rec format stores face images as JPEG blobs with an integer label
(identity ID) in a packed header. This script reads each record, extracts
the label and image bytes, and writes them to an ImageFolder tree.

Usage:
  python scripts/convert_rec.py --input /path/to/train.rec --output data/casia_webface/

  # With explicit .idx file (auto-detected by default)
  python scripts/convert_rec.py --input /path/to/train.rec --idx /path/to/train.idx --output data/casia_webface/

RecordIO binary format reference:
  Each record: [4-byte magic (0xCEDAEDFE)] [4-byte length] [4-byte flag] [payload]
  - If flag == 0: payload is the full record
  - If flag > 0: record is split across multiple chunks (flag = sequence number)
  Header within payload (IRHeader, 24 bytes):
    [4-byte flag] [4-byte float label] [8-byte id1] [8-byte id2]
  Image bytes follow immediately after the header.
"""

from __future__ import annotations

import argparse
import struct
import sys
from collections import defaultdict
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# RecordIO constants
_MAGIC = 0xCEDAEDFE
_HEADER_SIZE = 4 + 4  # magic + lrecord
_IR_HEADER_SIZE = 24   # flag(4) + label(4 float) + id1(8) + id2(8)


def _read_idx(idx_path: Path) -> list[int]:
    """Read a .idx file → list of byte offsets into the .rec file."""
    offsets = []
    with open(idx_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                # Format: record_index\tbyte_offset
                offsets.append(int(parts[1]))
    return offsets


def _iter_records(rec_path: Path, idx_path: Path | None = None):
    """Yield (label: int, image_bytes: bytes) for each record in a .rec file.

    If an .idx file is provided, uses it for seeking. Otherwise, reads
    sequentially.
    """
    # Auto-detect .idx
    if idx_path is None:
        candidate = rec_path.with_suffix(".idx")
        if candidate.exists():
            idx_path = candidate

    with open(rec_path, "rb") as f:
        if idx_path is not None:
            offsets = _read_idx(idx_path)
            for offset in offsets:
                f.seek(offset)
                record = _read_one_record(f)
                if record is not None:
                    yield record
        else:
            # Sequential scan
            while True:
                record = _read_one_record(f)
                if record is None:
                    break
                yield record


def _read_one_record(f) -> tuple[int, bytes] | None:
    """Read a single RecordIO record from current file position.

    Returns (label, image_bytes) or None at EOF.
    """
    # Read magic + lrecord (8 bytes)
    header_bytes = f.read(_HEADER_SIZE)
    if len(header_bytes) < _HEADER_SIZE:
        return None

    magic, lrecord = struct.unpack("<II", header_bytes)
    if magic != _MAGIC:
        return None

    # lrecord encodes: cflag in top 3 bits, length in remaining 29 bits
    cflag = (lrecord >> 29) & 7
    length = lrecord & ((1 << 29) - 1)

    if length == 0:
        return None

    # Read the payload
    data = f.read(length)
    if len(data) < length:
        return None

    # Handle multi-part records (cflag > 0 means continuation)
    if cflag > 0:
        # First part of a multi-part record — collect remaining chunks
        parts = [data]
        while True:
            chunk_header = f.read(_HEADER_SIZE)
            if len(chunk_header) < _HEADER_SIZE:
                break
            c_magic, c_lrecord = struct.unpack("<II", chunk_header)
            if c_magic != _MAGIC:
                break
            c_cflag = (c_lrecord >> 29) & 7
            c_length = c_lrecord & ((1 << 29) - 1)
            chunk = f.read(c_length)
            parts.append(chunk)
            if c_cflag == 0:
                break  # Last chunk
        data = b"".join(parts)

    # Skip padding (records are padded to 4-byte alignment)
    pad = (4 - (length % 4)) % 4
    if pad > 0:
        f.read(pad)

    # Parse the IR header (24 bytes)
    if len(data) < _IR_HEADER_SIZE:
        return None

    ir_flag, ir_label = struct.unpack("<If", data[:8])
    # id1, id2 = struct.unpack("<QQ", data[8:24])  # not needed

    label = int(ir_label)
    image_bytes = data[_IR_HEADER_SIZE:]

    # Sanity check: should start with JPEG magic or PNG magic
    if len(image_bytes) < 4:
        return None

    return label, image_bytes


def convert(
    rec_path: Path,
    output_dir: Path,
    idx_path: Path | None = None,
) -> dict[str, int]:
    """Convert a .rec file to ImageFolder layout.

    Returns dict of stats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    identity_counters: dict[int, int] = defaultdict(int)
    total_images = 0
    skipped = 0

    for label, image_bytes in _iter_records(rec_path, idx_path):
        # Skip header records (label often -1 or very large for metadata)
        if label < 0:
            skipped += 1
            continue

        identity_counters[label] += 1
        img_num = identity_counters[label]

        # Use zero-padded label as directory name
        label_dir = output_dir / f"{label:07d}"
        label_dir.mkdir(exist_ok=True)

        out_path = label_dir / f"{img_num:06d}.jpg"

        # Detect if image is JPEG or needs conversion
        if image_bytes[:3] == b"\xff\xd8\xff":
            out_path.write_bytes(image_bytes)
        else:
            # Try to convert via PIL (handles PNG, BMP, etc.)
            try:
                import io
                from PIL import Image
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img.save(out_path, format="JPEG", quality=95)
            except Exception:
                # Save raw and let the user sort it out
                out_path.write_bytes(image_bytes)

        total_images += 1
        if total_images % 10000 == 0:
            print(
                f"  {total_images} images extracted "
                f"({len(identity_counters)} identities)...",
                flush=True,
            )

    return {
        "total_images": total_images,
        "total_identities": len(identity_counters),
        "skipped_records": skipped,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert MXNet RecordIO to ImageFolder (no mxnet required)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/convert_rec.py --input data/casia/train.rec --output data/casia_webface/
  python scripts/convert_rec.py --input data/ms1mv2/train.rec --output data/ms1mv2/
        """,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to .rec file",
    )
    parser.add_argument(
        "--idx", default=None,
        help="Path to .idx file (auto-detected from .rec path if omitted)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output ImageFolder directory",
    )
    args = parser.parse_args()

    rec_path = Path(args.input)
    output_dir = Path(args.output)
    idx_path = Path(args.idx) if args.idx else None

    if not rec_path.exists():
        print(f"Error: {rec_path} does not exist")
        sys.exit(1)

    print(f"Input:  {rec_path}")
    if idx_path:
        print(f"Index:  {idx_path}")
    else:
        auto_idx = rec_path.with_suffix(".idx")
        if auto_idx.exists():
            print(f"Index:  {auto_idx} (auto-detected)")
        else:
            print(f"Index:  none (sequential scan)")
    print(f"Output: {output_dir}")

    stats = convert(rec_path, output_dir, idx_path)

    print(f"\nDone!")
    print(f"  Images:     {stats['total_images']}")
    print(f"  Identities: {stats['total_identities']}")
    print(f"  Skipped:    {stats['skipped_records']} (header/metadata records)")
    print(f"  Output:     {output_dir}")

    # Quick sanity check
    identity_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    if identity_dirs:
        sample = identity_dirs[0]
        sample_images = list(sample.glob("*.jpg"))
        print(f"\n  Sample identity: {sample.name}/ ({len(sample_images)} images)")


if __name__ == "__main__":
    main()
