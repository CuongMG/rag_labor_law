"""
Ingest Script — CLI để chạy Indexing Pipeline.

Usage:
    python scripts/ingest.py                   # Load tất cả PDF trong docs/
    python scripts/ingest.py --file luat.pdf   # Load 1 file PDF
    python scripts/ingest.py --reset           # Xóa collection cũ, index lại
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.indexing.pipeline import run_ingestion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline nạp tài liệu Luật Lao Động vào Qdrant Vector DB."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Một file PDF cần index",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Xóa collection Qdrant hiện tại và index lại từ đầu",
    )

    args = parser.parse_args()
    run_ingestion(file_path=args.file, reset=args.reset)