"""
Module Load Tài Liệu — hai luồng tách biệt:

- PDF: PyMuPDFLoader (LangChain) — mặc định, nhanh, cho file quét từ PDF.
- HTML: BeautifulSoup — khi nguồn là dữ liệu có cấu trúc sẵn (web, export HTML),
  không trộn với luồng PDF.
"""

from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from src.config import DOCS_DIR


# ============================================================
# PDF — luồng mặc định
# ============================================================

def load_pdf(pdf_path: str | Path) -> list[Document]:
    """Load 1 file PDF. Mỗi trang → 1 Document."""
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file PDF: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"File không phải PDF: {pdf_path}")

    loader = PyMuPDFLoader(str(pdf_path))
    documents = loader.load()

    print(f"  ✓ Loaded PDF '{pdf_path.name}': {len(documents)} trang")
    return documents


def load_all_pdfs(docs_dir: str | Path | None = None) -> list[Document]:
    """Load tất cả PDF trong thư mục (mặc định docs/)."""
    docs_dir = Path(docs_dir) if docs_dir else DOCS_DIR

    if not docs_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {docs_dir}")

    pdf_files = sorted(docs_dir.glob("**/*.pdf"))
    if not pdf_files:
        raise ValueError(f"Không tìm thấy file PDF nào trong: {docs_dir}")

    print(f"📂 Tìm thấy {len(pdf_files)} file PDF trong '{docs_dir}':")
    
    all_documents: list[Document] = []
    for pdf_file in pdf_files:
        all_documents.extend(load_pdf(pdf_file))

    print(f"📄 Tổng cộng: {len(all_documents)} trang từ {len(pdf_files)} file PDF")
    return all_documents


def merge_pages(documents: list[Document], separator: str = "\n\n") -> str:
    """Gộp nội dung tất cả document thành 1 string."""
    return separator.join(doc.page_content for doc in documents)


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Module Loader (PDF)")
    print("=" * 60)

    docs = load_all_pdfs()
    print(f"\nMetadata trang đầu: {docs[0].metadata}")
    print(f"Content (200 ký tự đầu):\n{docs[0].page_content[:200]}")
    
    full_text = merge_pages(docs)
    print(f"\nTổng độ dài gộp: {len(full_text):,} ký tự")