"""
Indexing Pipeline — Chạy pipeline nạp dữ liệu hoàn chỉnh.

Các bước: load → chunk Chương/Điều → NLP → embed → Qdrant Cloud.
"""

import sys
from pathlib import Path

from src.indexing.loader import load_pdf, load_all_pdfs
from src.indexing.chunker import chunk_documents
from src.indexing.text_processor import preprocess_for_embedding, preprocess_for_bm25
from src.vectorstore import QdrantVectorStore


def run_ingestion(
    file_path: str | None = None,
    reset: bool = False,
):
    """Chạy quy trình nạp tài liệu."""
    print("🚀 Bắt đầu quy trình nạp tài liệu (Ingestion Pipeline)...")

    try:
        vector_store = QdrantVectorStore()
    except Exception as e:
        print(f"❌ Không thể kết nối tới Qdrant database: {e}")
        sys.exit(1)

    if reset:
        vector_store.recreate_collection()

    # Load — PDF
    if file_path:
        path = Path(file_path)
        print(f"📂 Đang load file: {path}")
        raw_docs = load_pdf(path)
    else:
        print("📂 Đang quét PDF trong docs/...")
        try:
            raw_docs = load_all_pdfs()
        except ValueError as e:
            print(f"❌ Lỗi: {e}")
            sys.exit(1)

    print("✂️  Đang phân tích cấu trúc và chia nhỏ văn bản...")
    chunks = chunk_documents(raw_docs)

    if not chunks:
        print("⚠️ Không tạo được chunk nào. Hủy bỏ pipeline.")
        return

    print("🧠 Đang xử lý text: clean (cho dense) + bỏ stopword (cho BM25)...")
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        chunk.page_content = preprocess_for_embedding(chunk.page_content)
        chunk.metadata["bm25_text"] = preprocess_for_bm25(chunk.page_content)
        processed_chunks.append(chunk)

        if (i + 1) % 50 == 0:
            print(f"  ✓ Đã xử lý xong {i + 1}/{len(chunks)} chunks")

    print(f"✓ Hoàn tất xử lý text cho {len(processed_chunks)} chunks.")

    try:
        vector_store.add_documents(processed_chunks)
        print("🎉 QUY TRÌNH NẠP DỮ LIỆU THÀNH CÔNG RỰC RỠ!")
    except Exception as e:
        print(f"❌ Lỗi khi index tài liệu vào Qdrant: {e}")
        sys.exit(1)
