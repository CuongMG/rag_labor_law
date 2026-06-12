"""
Query Script — CLI để chạy Retrieval Pipeline.

Usage:
    python scripts/query.py "câu hỏi của bạn"
    python scripts/query.py "nghỉ phép năm bao nhiêu ngày?" --mode dense
    python scripts/query.py "thời giờ làm việc" --top-k 5
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.retriever import Retriever
from src.retrieval.generator import AnswerGenerator


def run_query(query: str, top_k: int = 3, mode: str = "hybrid"):
    """Chạy quy trình hỏi-đáp RAG."""
    print(f"🔍 Query: '{query}'")
    print(f"⚙️  Mode: {mode}, Top-K: {top_k}")

    try:
        # Bước 1: Retrieve
        print("\n📡 Đang tìm kiếm tài liệu liên quan...")
        retriever = Retriever()
        results = retriever.retrieve(query, top_k=top_k, mode=mode)
        print(f"  ✓ Tìm thấy {len(results)} kết quả")

        # Bước 2: Generate answer
        print("\n🤖 Đang sinh câu trả lời...")
        generator = AnswerGenerator()
        answer = generator.generate_answer(query, results)

        print("\n" + "=" * 60)
        print("📝 CÂU TRẢ LỜI:")
        print("=" * 60)
        print(answer)

        # In nguồn tham khảo
        if results:
            print("\n" + "=" * 60)
            print("📚 NGUỒN THAM KHẢO:")
            print("=" * 60)
            for i, res in enumerate(results):
                meta = res["metadata"]
                print(f"{i+1}. Điều {meta['article']} - {meta['title']}")
                print(f"   Chương {meta['chapter']}, Mục {meta['section']}")
                print(f"   Score: {res['score']:.4f}")

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hỏi đáp về Luật Lao Động Việt Nam (RAG)."
    )
    parser.add_argument(
        "query",
        type=str,
        help="Câu hỏi về Luật Lao Động",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Số kết quả tìm kiếm (mặc định: 3)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hybrid", "dense"],
        default="hybrid",
        help="Chế độ tìm kiếm (mặc định: hybrid)",
    )

    args = parser.parse_args()
    run_query(query=args.query, top_k=args.top_k, mode=args.mode)