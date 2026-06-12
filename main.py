"""
RAG Labor Law — Hệ thống hỏi đáp về Bộ Luật Lao Động Việt Nam.

Usage:
    python main.py ingest                 # Index dữ liệu
    python main.py ingest --reset         # Index lại từ đầu
    python main.py query "câu hỏi"        # Hỏi đáp
    python main.py query "câu hỏi" --mode dense
"""

import sys
import argparse

from src.indexing.pipeline import run_ingestion
from scripts.query import run_query


def main():
    parser = argparse.ArgumentParser(
        description="RAG Labor Law — Hỏi đáp về Bộ Luật Lao Động Việt Nam"
    )
    subparsers = parser.add_subparsers(dest="command", help="Lệnh")

    # Subcommand: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Nạp dữ liệu vào vector store")
    ingest_parser.add_argument(
        "--file", type=str, default=None, help="Một file PDF cần index"
    )
    ingest_parser.add_argument(
        "--reset", action="store_true", help="Xóa collection cũ và index lại"
    )

    # Subcommand: query
    query_parser = subparsers.add_parser("query", help="Hỏi đáp về Luật Lao Động")
    query_parser.add_argument("query", type=str, help="Câu hỏi")
    query_parser.add_argument(
        "--top-k", type=int, default=5, help="Số kết quả (mặc định: 5)"
    )
    query_parser.add_argument(
        "--mode",
        type=str,
        choices=["hybrid", "dense"],
        default="hybrid",
        help="Chế độ tìm kiếm (mặc định: hybrid)",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingestion(file_path=args.file, reset=args.reset)
    elif args.command == "query":
        run_query(query=args.query, top_k=args.top_k, mode=args.mode)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()