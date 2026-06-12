"""
Module Embedding — Tạo vector biểu diễn (embeddings) cho text.

Sử dụng Google Gemini Embedding API qua LangChain langchain_google_genai.
Model sử dụng mặc định: models/gemini-embedding-exp-03-07 (768 dimensions).

Tự quản lý batch + retry khi gặp quota 429 (free tier: 100 req/min).
"""

import re
import time

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError

from src import config


class GeminiEmbedder:
    """Wrapper cho Google Gemini Embedding API."""

    def __init__(self, batch_size: int = 5):
        """
        Khởi tạo GoogleGenerativeAIEmbeddings với API Key từ config.

        Args:
            batch_size: Số text tối đa mỗi lần gọi API (mặc định 5).
                       Batch nhỏ → ít nguy cơ vượt quota free tier.
        """
        if not config.GEMINI_API_KEY:
            raise ValueError(
                "Không tìm thấy GEMINI_API_KEY. Vui lòng cấu hình trong file .env"
            )

        self.batch_size = batch_size
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            output_dimensionality=config.EMBEDDING_DIM,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Tạo embeddings cho danh sách các văn bản.

        Tự chia batch nhỏ + retry khi gặp 429 để tránh vượt quota free tier.

        Args:
            texts: Danh sách các đoạn text cần embed.

        Returns:
            List[List[float]]: Danh sách các vector.
        """
        all_vectors: list[list[float]] = []
        total = len(texts)

        for i in range(0, total, self.batch_size):
            batch = texts[i : i + self.batch_size]

            # Retry khi gặp 429
            max_retries = 5
            retry_delay = 4
            for attempt in range(max_retries):
                try:
                    vectors = self.embeddings.embed_documents(batch)
                    all_vectors.extend(vectors)
                    print(f"  📊 Embed batch {i // self.batch_size + 1}/{(total - 1) // self.batch_size + 1} ({len(batch)} texts)")
                    time.sleep(1)  # delay 1s giữa các batch
                    break
                except GoogleGenerativeAIError as e:
                    error_str = str(e)
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        m = re.search(r"retry in (\d+(?:\.\d+)?)s", error_str)
                        wait = float(m.group(1)) + 1 if m else retry_delay
                        print(f"  ⏳ Quota exceeded, đợi {wait:.0f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait)
                        retry_delay *= 2
                    else:
                        raise  # lỗi khác, không retry
            else:
                raise RuntimeError(f"Gemini API quota exceeded after {max_retries} retries")

        return all_vectors

    def embed_query(self, text: str) -> list[float]:
        """
        Tạo embedding cho 1 câu truy vấn (query).

        Args:
            text: Câu query cần tìm kiếm.

        Returns:
            List[float]: Vector biểu diễn câu query.
        """
        return self.embeddings.embed_query(text)


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Module Embedder")
    print("=" * 60)

    try:
        embedder = GeminiEmbedder()

        # Test embed query
        query = "quyền nghỉ phép năm của người lao động"
        vector = embedder.embed_query(query)

        print(f"✓ Embed query thành công!")
        print(f"  Query: '{query}'")
        print(f"  Vector length: {len(vector)}")
        print(f"  Sample values: {vector[:5]}...")

        # Test embed documents
        docs = ["Hợp đồng lao động vô hiệu", "Thời giờ làm việc bình thường"]
        vectors = embedder.embed_documents(docs)
        print(f"✓ Embed documents thành công!")
        print(f"  Loaded {len(vectors)} vectors")
        print(f"  Dimensions: {len(vectors[0])}")

    except Exception as e:
        print(f"❌ Lỗi: {e}")