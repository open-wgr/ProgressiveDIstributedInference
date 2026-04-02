"""Convert MXNet RecordIO datasets to image folder format.

Usage: python scripts/convert_rec.py --input /path/to/train.rec --output /path/to/images/
"""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Convert RecordIO to image folder")
    parser.add_argument("--input", required=True, help="Path to .rec file")
    parser.add_argument("--output", required=True, help="Output image folder")
    args = parser.parse_args()

    try:
        import mxnet as mx
    except ImportError:
        print("mxnet is required for RecordIO conversion.")
        print("Install with: pip install 'ppi[conversion]'")
        return

    raise NotImplementedError("RecordIO conversion not yet implemented")


if __name__ == "__main__":
    main()
