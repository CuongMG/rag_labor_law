"""
Module Vector Store — Quản lý cơ sở dữ liệu vector Qdrant Cloud.

Cấu hình Hybrid Search:
  - Dense Vector: Gemini Embedding (768 dimensions)
  - Sparse Vector: BM25 (via fastembed)
  - Search Fusion: Reciprocal Rank Fusion (RRF)
"""

import uuid

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from src import config
from src.embedder import GeminiEmbedder
from src.indexing.text_processor import preprocess_for_bm25, preprocess_for_embedding

# Namespace cố định — uuid5(source + chunk_id) → ingest lại ghi đè cùng point
_POINT_ID_NAMESPACE = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")


def _create_qdrant_client() -> QdrantClient:
    """Kết nối Qdrant Cloud — yêu cầu QDRANT_URL và QDRANT_API_KEY trong .env."""
    if not config.QDRANT_URL or not config.QDRANT_API_KEY:
        raise ValueError(
            "Thiếu cấu hình Qdrant Cloud. Đặt QDRANT_URL và QDRANT_API_KEY trong file .env."
        )
    print(f"🌐 Kết nối tới Qdrant Cloud tại: {config.QDRANT_URL}")
    return QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        cloud_inference=True,
    )


class QdrantVectorStore:
    """Quản lý dữ liệu trong Qdrant và thực hiện tìm kiếm hybrid."""

    def __init__(self, embedder: GeminiEmbedder = None):
        """
        Khởi tạo Qdrant client và load model sparse embedding.

        Args:
            embedder: Đối tượng GeminiEmbedder. Nếu None, tự khởi tạo.
        """
        self.embedder = embedder or GeminiEmbedder()
        self.client = _create_qdrant_client()

        # 2. Khởi tạo model Sparse Embedding (BM25) để phục vụ keyword search
        print("🤖 Loading BM25 sparse model (fastembed)...")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

        # 3. Tạo collection nếu chưa có
        self._init_collection()

    @staticmethod
    def _make_point_id(doc) -> str:
        """ID ổn định từ source + chunk_id — upsert lại không tạo duplicate."""
        source = doc.metadata.get("source", "")
        chunk_id = doc.metadata.get("chunk_id", "")
        sub_chunk = doc.metadata.get("sub_chunk", "")
        key = f"{source}|{chunk_id}|{sub_chunk}"
        return str(uuid.uuid5(_POINT_ID_NAMESPACE, key))

    def _init_collection(self):
        """Khởi tạo collection với cấu hình song song dense + sparse."""
        try:
            # Kiểm tra xem collection đã tồn tại chưa
            self.client.get_collection(config.COLLECTION_NAME)
            print(f"✓ Collection '{config.COLLECTION_NAME}' đã tồn tại.")
        except Exception:
            # Nếu chưa có, tiến hành tạo mới
            print(f"🆕 Đang tạo collection mới: '{config.COLLECTION_NAME}'...")
            self.client.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config={
                    "dense": models.VectorParams(
                        size=config.EMBEDDING_DIM,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,                           
                        ),
                        modifier=models.Modifier.IDF,  # Kích hoạt IDF modifier cho BM25
                    )
                },
            )
            print(f"✓ Tạo collection '{config.COLLECTION_NAME}' thành công.")

    def recreate_collection(self):
        """Xóa collection cũ và tạo lại từ đầu."""
        print(f"🧹 Đang xóa collection '{config.COLLECTION_NAME}'...")
        try:
            self.client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass
        self._init_collection()

    def get_sparse_vector(self, text: str) -> models.SparseVector:
        """
        Tạo sparse vector cho text dùng BM25.

        Args:
            text: Nội dung văn bản.

        Returns:
            models.SparseVector: Đối tượng vector thưa thớt của Qdrant.
        """
        # fastembed.embed trả về generator, lấy phần tử đầu tiên
        embeddings = list(self.sparse_model.embed([text]))
        emb = embeddings[0]
        
        return models.SparseVector(
            indices=emb.indices.tolist(),
            values=emb.values.tolist(),
        )

    def add_documents(self, documents: list):
        """
        Index danh sách LangChain Documents vào Qdrant.

        - dense: doc.page_content (đã chuẩn hóa / clean)
        - sparse: metadata bm25_text (clean + bỏ stopword)
        - payload page_content: bản clean để hiển thị cho LLM
        """
        if not documents:
            return

        print(f"⚡ Đang index {len(documents)} chunks vào Qdrant...")

        points = []

        dense_texts = [doc.page_content for doc in documents]
        sparse_texts = [
            doc.metadata.get("bm25_text") or preprocess_for_bm25(doc.page_content)
            for doc in documents
        ]

        dense_vectors = self.embedder.embed_documents(dense_texts)
        sparse_vectors = [self.get_sparse_vector(text) for text in sparse_texts]

        for i, doc in enumerate(documents):
            point_id = self._make_point_id(doc)

            payload = {
                "page_content": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "chapter": doc.metadata.get("chapter", ""),
                "section": doc.metadata.get("section", ""),
                "article": doc.metadata.get("article", ""),
                "title": doc.metadata.get("title", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vectors[i],
                        "sparse": sparse_vectors[i],
                    },
                    payload=payload,
                )
            )

        # Upsert lên Qdrant theo batches để tránh quá tải payload
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=config.COLLECTION_NAME,
                wait=True,
                points=batch,
            )
            print(f"  ✓ Đã upsert xong batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")

        print(f"✓ Index hoàn tất. Hiện có {self.client.count(config.COLLECTION_NAME).count} points trong collection.")

    def search_hybrid(self, query: str, top_k: int = None) -> list[dict]:
        """
        Thực hiện Hybrid Search: dense + sparse kết hợp RRF (Reciprocal Rank Fusion).

        RRF fusion kết hợp thứ hạng từ 2 bộ máy tìm kiếm (semantic + keyword)
        để cho ra kết quả xếp hạng tốt nhất.

        Args:
            query: Câu hỏi/từ khóa tìm kiếm.
            top_k: Số kết quả cần trả về.

        Returns:
            List[dict]: Danh sách kết quả tìm kiếm đã xếp hạng.
        """
        top_k = top_k or config.TOP_K

        # Query: dense = chuẩn hóa/clean; sparse = thêm bỏ stopword (khớp lúc index)
        query_dense = self.embedder.embed_query(preprocess_for_embedding(query))
        query_sparse = self.get_sparse_vector(preprocess_for_bm25(query))

        # 2. Thực hiện query hybrid sử dụng Prefetch API của Qdrant
        results = self.client.query_points(
            collection_name=config.COLLECTION_NAME,
            prefetch=[
                # Nhánh 1: Semantic search
                models.Prefetch(
                    query=query_dense,
                    using="dense",
                    limit=config.HYBRID_PREFETCH_LIMIT,
                ),
                # Nhánh 2: Keyword/BM25 search
                models.Prefetch(
                    query=query_sparse,
                    using="sparse",
                    limit=config.HYBRID_PREFETCH_LIMIT,
                ),
            ],
            # RRF Fusion xếp hạng lại
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
        )

        # 3. Format output
        output = []
        for point in results.points:
            output.append({
                "content": point.payload.get("page_content", ""),
                "score": point.score,  # Điểm RRF score
                "metadata": {
                    "source": point.payload.get("source", ""),
                    "chapter": point.payload.get("chapter", ""),
                    "section": point.payload.get("section", ""),
                    "article": point.payload.get("article", ""),
                    "title": point.payload.get("title", ""),
                    "chunk_id": point.payload.get("chunk_id", ""),
                }
            })
            
        return output

    def search_dense(self, query: str, top_k: int = None) -> list[dict]:
        """Chỉ tìm kiếm ngữ nghĩa (Dense vector)."""
        top_k = top_k or config.TOP_K
        query_dense = self.embedder.embed_query(preprocess_for_embedding(query))
        
        results = self.client.query_points(
            collection_name=config.COLLECTION_NAME,
            query=query_dense,
            using="dense",
            limit=top_k,
        )
        
        output = []
        for point in results.points:
            output.append({
                "content": point.payload.get("page_content", ""),
                "score": point.score,
                "metadata": point.payload
            })
        return output

    def get_collection_stats(self) -> dict:
        """Lấy thông tin thống kê về collection."""
        try:
            collection_info = self.client.get_collection(config.COLLECTION_NAME)
            count_info = self.client.count(config.COLLECTION_NAME)
            return {
                "status": collection_info.status,
                "vectors_count": collection_info.vectors_count,
                "points_count": count_info.count,
            }
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Module Vector Store")
    print("=" * 60)

    try:
        store = QdrantVectorStore()
        stats = store.get_collection_stats()
        print(f"✓ Thống kê collection hiện tại: {stats}")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
