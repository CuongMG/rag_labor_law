# ⚖️ RAG Labor Law — Hỏi đáp Bộ Luật Lao Động Việt Nam

Hệ thống **Retrieval-Augmented Generation (RAG)** cho phép hỏi đáp về **Bộ Luật Lao Động Việt Nam 2019** sử dụng:
- **Google Gemini Embedding** — vector hóa văn bản luật
- **Qdrant Cloud** — vector database với hybrid search (dense + BM25)
- **Google Gemini 3.5 Flash** — sinh câu trả lời

---

## Kiến trúc

```
rag_labor_law/
├── app.py                         # Gradio Chat UI
├── main.py                        # CLI entry point
├── scripts/
│   ├── ingest.py                  # CLI: index dữ liệu
│   └── query.py                   # CLI: hỏi đáp
├── src/
│   ├── config.py                  # Cấu hình chung
│   ├── embedder.py                # Gemini Embedding (batch + retry 429)
│   ├── vectorstore.py             # Qdrant client (hybrid search)
│   ├── indexing/                  # Phase 1: Indexing Pipeline
│   │   ├── loader.py              # Load PDF (PyMuPDF)
│   │   ├── chunker.py             # Chunk theo Điều + semantic split
│   │   ├── text_processor.py      # NLP: clean, stopword removal
│   │   └── pipeline.py            # Pipeline hoàn chỉnh
│   └── retrieval/                 # Phase 2: Retrieval Pipeline
│       ├── retriever.py           # Hybrid search
│       └── generator.py           # LLM sinh câu trả lời
└── docs/
    └── bo_luat_lao_dong_2019.pdf  # Bộ Luật Lao Động (nguồn dữ liệu)
```

### 2 Phase RAG

| Phase | Mô tả | Công nghệ |
|-------|-------|-----------|
| **Indexing** | Load PDF → chunk theo Điều → clean text → embed (dense + BM25) → lưu Qdrant | PyMuPDF, LangChain, Gemini Embedding, Qdrant |
| **Retrieval** | Query → hybrid search (dense + BM25, RRF fusion) → LLM sinh câu trả lời | Gemini Embedding, Qdrant, Gemini 3.5 Flash |

---

## Yêu cầu

- Python 3.11+
- Tài khoản [Google AI Studio](https://aistudio.google.com/) (Gemini API key)
- Tài khoản [Qdrant Cloud](https://cloud.qdrant.io/) (vector database)

## Cài đặt

```bash
# 1. Clone repo
git clone https://github.com/CuongMG/rag_labor_law.git
cd rag_labor_law

# 2. Tạo virtual environment
uv venv
source .venv/bin/activate

# 3. Cài dependencies
uv pip install -r pyproject.toml

# 4. Tạo file .env
cp .env.example .env
```

## Cấu hình `.env`

Tạo file `.env` với nội dung:

```env
# Google Gemini API Key (lấy từ https://aistudio.google.com/)
GEMINI_API_KEY=your_gemini_api_key_here

# Qdrant Cloud (lấy từ https://cloud.qdrant.io/)
QDRANT_URL=https://your-instance.us-west-2-0.aws.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here
```

### Lấy Gemini API Key
1. Truy cập [Google AI Studio](https://aistudio.google.com/)
2. Đăng nhập, click **Get API Key**
3. Tạo key mới → copy vào `.env`

> **Free tier:** 1000 embedding request/ngày, 1500 generation request/ngày.

### Lấy Qdrant Cloud
1. Truy cập [Qdrant Cloud](https://cloud.qdrant.io/)
2. Đăng ký tài khoản free
3. Tạo cluster → copy URL và API Key

---

## Cách chạy

### 1. Index dữ liệu (chạy 1 lần)

```bash
# Index tất cả PDF trong docs/
python main.py ingest

# Hoặc xóa collection cũ, index lại từ đầu
python main.py ingest --reset
```

Quá trình index:
1. Load file PDF → 50-70 trang
2. Parse cấu trúc: Chương, Mục, Điều → ~220 Điều
3. Chunk: giữ nguyên Điều ≤ 3000 ký tự, semantic split Điều dài hơn
4. Clean text + tạo BM25 text (bỏ stopword)
5. Embed (dense) + BM25 (sparse) → upsert Qdrant

### 2. Hỏi đáp (CLI)

```bash
# Hỏi về Luật Lao Động
python main.py query "nghỉ phép năm bao nhiêu ngày?"
python main.py query "thời giờ làm việc" --top-k 5
python main.py query "hợp đồng lao động có những loại nào?" --mode dense

# Hoặc dùng script trực tiếp
python scripts/query.py "câu hỏi của bạn"
```

### 3. Giao diện Chat (Gradio)

```bash
python app.py
# → Mở http://localhost:7860
```

Giao diện web với:
- Ô nhập tin nhắn
- Câu hỏi mẫu (click để dùng ngay)
- Hiển thị nguồn tham khảo sau câu trả lời

---

## Chiến lược Chunking

2 tầng, không overlap:

**Tầng 1 — Cấu trúc pháp luật:**
- Parse regex: Chương, Mục, Điều
- Mỗi Điều = 1 đơn vị, gắn metadata (chapter, section, article, title)
- Xử lý phần mở đầu (preamble) riêng

**Tầng 2 — Semantic split (chỉ khi Điều > 3000 ký tự):**
- `SemanticChunker` (Gemini embedding) cắt tại điểm khác nghĩa
- Đệ quy giảm dần percentile nếu chunk vẫn quá dài
- Fallback: chia theo khoản/đoạn nếu semantic không đủ

**Config (src/config.py):**
```python
CHUNK_SIZE = 3000                    # Ngưỡng kích hoạt semantic split
SEMANTIC_BREAKPOINT_PERCENTILE = 80  # Cắt tại 20% khoảng cách lớn nhất
SEMANTIC_BREAKPOINT_MIN = 75         # Tối đa 1 lần đệ quy
```

---

## Xử lý lỗi Gemini API Quota (429)

Gemini free tier giới hạn: **1000 embedding request/ngày**, **1500 generation request/ngày**.

Code tự động:
1. **Chia batch nhỏ** (5 texts/batch) cho embedding
2. **Delay 1s** giữa các batch
3. **Retry** khi gặp 429 — đọc `retryDelay` từ error message, đợi đúng thời gian rồi thử lại (tối đa 5 lần)

Nếu hết quota ngày → đợi ~14h giờ VN (0h PT) hoặc tạo API key mới.

---

## Cấu hình

### File `src/config.py`

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `CHUNK_SIZE` | 3000 | Kích thước chunk tối đa (ký tự) |
| `SEMANTIC_BREAKPOINT_PERCENTILE` | 80 | Ngưỡng cắt semantic |
| `SEMANTIC_BREAKPOINT_STEP` | 25 | Bước giảm percentile khi đệ quy |
| `SEMANTIC_BREAKPOINT_MIN` | 75 | Ngưỡng tối thiểu (fallback) |
| `EMBEDDING_MODEL` | gemini-embedding-2-preview | Model embedding |
| `GENERATION_MODEL` | gemini-3.5-flash | Model sinh câu trả lời |
| `TOP_K` | 5 | Số kết quả retrieval |
| `HYBRID_PREFETCH_LIMIT` | 7 | Số candidates cho hybrid search |

---

## Công nghệ sử dụng

| Công nghệ | Phiên bản | Mục đích |
|-----------|-----------|----------|
| Python | 3.13 | Ngôn ngữ |
| LangChain | 0.3+ | Framework RAG |
| Google Gemini | embedding-2, 2.0 flash | Embedding + Generation |
| Qdrant Cloud | - | Vector database |
| FastEmbed | - | BM25 sparse embedding |
| PyMuPDF | - | Load PDF |
| Underthesea | - | NLP tiếng Việt |
| Gradio | 6.18+ | Chat UI |