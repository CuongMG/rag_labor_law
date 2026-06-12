"""
Cấu hình tập trung cho hệ thống RAG Luật Lao Động.

Tất cả các hằng số, đường dẫn, tham số model được quản lý tại đây.
Khi cần thay đổi cấu hình, chỉ cần sửa file này — không cần sửa các module khác.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

# ============================================================
# PATHS - Đường dẫn thư mục
# ============================================================
# Thư mục gốc của project (parent của src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Thư mục chứa tài liệu PDF gốc
DOCS_DIR = PROJECT_ROOT / "docs"

# Thư mục lưu dữ liệu đã xử lý
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# ============================================================
# API KEYS & CLOUD CREDENTIALS
# ============================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Qdrant Cloud 
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# ============================================================
# CHUNKING - Tham số chia text
# ============================================================
# Kích thước chunk tối đa (ký tự) — chỉ áp dụng khi 1 Điều quá dài, chia tiếp bằng semantic
CHUNK_SIZE = 3000

# Ngưỡng percentile khi tìm điểm cắt semantic (cao → ít điểm cắt, giữ nhiều câu gần nghĩa chung)
SEMANTIC_BREAKPOINT_PERCENTILE = 80

# Bước hạ percentile khi cần chia nhỏ hơn (vẫn theo ngữ nghĩa, không overlap)
SEMANTIC_BREAKPOINT_STEP = 25
SEMANTIC_BREAKPOINT_MIN = 75

# Regex tách câu cho văn bản luật VN: kết thúc câu, khoản (1. 2.), điểm (a) b))
SEMANTIC_SENTENCE_REGEX = (
    r"(?<=\.)\s+(?=[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ])"
    r"|(?:\n)(?=\d+\.\s)"
    r"|(?:\n)(?=[a-zđ]\)\s)"
    r"|(?<=;)\s+"
)

# ============================================================
# MODELS - Cấu hình model AI
# ============================================================
# Google Gemini Embedding model (free tier)
EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIM = 768

# Google Gemini LLM cho generation
GENERATION_MODEL = "gemini-3.5-flash"

# ============================================================
# VECTOR STORE - Cấu hình Qdrant
# ============================================================
# Tên collection trong Qdrant
COLLECTION_NAME = "rag_labor_law"

# ============================================================
# SEARCH - Tham số tìm kiếm
# ============================================================
# Số kết quả trả về mặc định
TOP_K = 5

# Số candidates cho hybrid search (trước khi RRF fusion)
HYBRID_PREFETCH_LIMIT = 7
