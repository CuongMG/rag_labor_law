"""
Module Retriever — Điều phối luồng tìm kiếm (Retrieval Pipeline).

Module này đóng vai trò interface tìm kiếm duy nhất cho các phần khác của hệ thống
(như CLI script hay Web API). Nó cho phép chọn chế độ tìm kiếm:
  - 'hybrid': Kết hợp Dense + BM25 bằng RRF (Đề xuất dùng)
  - 'dense': Chỉ tìm kiếm ngữ nghĩa dùng Gemini embeddings
"""

from src.vectorstore import QdrantVectorStore
from src.config import TOP_K


class Retriever:
    """Điều phối toàn bộ quá trình tìm kiếm thông tin từ cơ sở dữ liệu."""

    def __init__(self, vector_store: QdrantVectorStore = None):
        """
        Khởi tạo Retriever.

        Args:
            vector_store: Đối tượng QdrantVectorStore. Nếu None, tự khởi tạo.
        """
        self.store = vector_store or QdrantVectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        mode: str = "hybrid",
    ) -> list[dict]:
        """
        Tìm kiếm các đoạn tài liệu liên quan đến query.

        Args:
            query: Câu truy vấn của người dùng.
            top_k: Số kết quả tối đa cần lấy.
            mode: Chế độ tìm kiếm ('hybrid', 'dense').

        Returns:
            List[dict]: Danh sách các đoạn văn liên quan nhất kèm score và metadata.
        """
        top_k = top_k or TOP_K
        query = query.strip()

        if not query:
            return []

        if mode == "hybrid":
            return self.store.search_hybrid(query, top_k=top_k)
        elif mode == "dense":
            return self.store.search_dense(query, top_k=top_k)
        else:
            raise ValueError(f"Chế độ tìm kiếm '{mode}' không hợp lệ. Chọn 'hybrid' hoặc 'dense'.")


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Module Retriever")
    print("=" * 60)

    try:
        retriever = Retriever()
        query = "Thời giờ làm việc bình thường"
        
        print(f"\n🔍 Thử tìm kiếm với query: '{query}'")
        
        # Test hybrid search
        print("\n--- HYBRID SEARCH RESULTS ---")
        results = retriever.retrieve(query, top_k=3, mode="hybrid")
        for i, res in enumerate(results):
            meta = res["metadata"]
            print(f"{i+1}. [RRF Score: {res['score']:.4f}] - Điều {meta['article']}: {meta['title']}")
            print(f"   Chapter: {meta['chapter']}")
            print(f"   Excerpt: {res['content'][:120]}...\n")

    except Exception as e:
        print(f"❌ Lỗi: {e}")