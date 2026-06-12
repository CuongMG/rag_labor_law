"""
Module Chunking — Tài liệu Luật Lao động.

Chiến lược 2 tầng (không overlap):
  Tầng 1 — Cấu trúc pháp luật: tách theo Điều, gắn metadata Chương/Mục.
  Tầng 2 — Chỉ khi một Điều vượt CHUNK_SIZE: SemanticChunker (Gemini embedding)
            cắt tại chỗ các câu/khoản *khác nghĩa*, giữ các câu gần nghĩa trong cùng chunk.

Nguyên tắc: không bao giờ gộp/cắt xuyên ranh giới Điều; semantic chỉ chạy trong phạm vi 1 Điều.
"""

import re
import time
from functools import lru_cache

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from src.config import (
    CHUNK_SIZE,
    SEMANTIC_BREAKPOINT_MIN,
    SEMANTIC_BREAKPOINT_PERCENTILE,
    SEMANTIC_BREAKPOINT_STEP,
    SEMANTIC_SENTENCE_REGEX,
)
from src.embedder import GeminiEmbedder


def extract_structure(text: str) -> list[dict]:
    """
    Tầng 1: Parse Chương/Mục/Điều — đơn vị chia chính là Điều.

    Mỗi phần trả về giữ nguyên ranh giới Điều; Chương/Mục lưu trong metadata.
    """
    chunks = []
    current_chapter = ""
    current_section = ""

    chapter_pattern = re.compile(
        r"Chương\s+([IVXLCDM]+|\d+)[.\s]*\n?\s*([^\n]*)",
        re.MULTILINE,
    )
    section_pattern = re.compile(
        r"Mục\s+(\d+)[.\s]*\n?\s*([^\n]*)",
        re.MULTILINE,
    )
    article_pattern = re.compile(
        r"(Điều\s+(\d+)\.\s*([^\n]*))",
        re.MULTILINE,
    )

    article_matches = list(article_pattern.finditer(text))

    if not article_matches:
        return [{
            "content": text,
            "chapter": "",
            "section": "",
            "article": "",
            "title": "",
        }]
    
    preamble = text[: article_matches[0].start()].strip()
    if preamble:
        ch_matches = list(chapter_pattern.finditer(preamble))
        if ch_matches:
            current_chapter = ch_matches[-1].group(1)
        chunks.append({
            "content": preamble,
            "chapter": current_chapter,
            "section": "",
            "article": "preamble",
            "title": "Phần mở đầu",
        })

    for i, match in enumerate(article_matches):
        article_num = match.group(2)
        article_title = match.group(3).strip()
        start = match.start()
        end = (
            article_matches[i + 1].start()
            if i + 1 < len(article_matches)
            else len(text)
        )
        
        content = text[start:end].strip()
       
        preceding_text = text[:start]
        ch_matches = list(chapter_pattern.finditer(preceding_text))
        
        if ch_matches:
            current_chapter = ch_matches[-1].group(1)

        sec_matches = list(section_pattern.finditer(preceding_text))
        if sec_matches:
            current_section = sec_matches[-1].group(1)
        
        chunks.append({
            "content": content,
            "chapter": current_chapter,
            "section": current_section,
            "article": article_num,
            "title": article_title,
        })

    return chunks


@lru_cache(maxsize=1)
def _get_embeddings():
    return GeminiEmbedder().embeddings


def _make_semantic_splitter(percentile: float) -> SemanticChunker:
    return SemanticChunker(
        _get_embeddings(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=percentile,
        sentence_split_regex=SEMANTIC_SENTENCE_REGEX,
    )


def _split_by_khoan_fallback(text: str, chunk_size: int) -> list[str]:
    """
    Fallback cuối: chia theo khoản/đoạn (\\n\\n hoặc đầu dòng '1. ', '2. ').
    Không overlap; chỉ dùng khi semantic vẫn không đưa đoạn xuống dưới chunk_size.
    """
    blocks = re.split(r"(?=\n\d+\.\s)", text)
    if len(blocks) <= 1:
        blocks = re.split(r"\n{2,}", text)
    if len(blocks) <= 1:
        return [text]

    merged: list[str] = []
    current = ""

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        candidate = f"{current}\n\n{block}".strip() if current else block
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                merged.append(current)
                current = ""
            if len(block) > chunk_size:
                merged.append(block)
            else:
                current = block

    if current:
        merged.append(current)
    return merged or [text]


def _semantic_split_within_article(
    content: str,
    chunk_size: int,
    percentile: float | None = None,
) -> list[str]:
    """
    Chia nội dung *trong một Điều* bằng embedding — giữ các câu gần nghĩa chung.

    Cắt tại điểm cosine distance cao giữa các nhóm câu liền kề.
    Nếu vẫn quá dài: hạ percentile và thử lại (thêm điểm cắt ngữ nghĩa).
    """
    percentile = percentile if percentile is not None else SEMANTIC_BREAKPOINT_PERCENTILE
    time.sleep(1)  # delay 1s để tránh vượt Gemini API quota (free: 100 req/min)
    splitter = _make_semantic_splitter(percentile)
    parts = splitter.split_text(content)
    
    if not parts:
        return [content]

    final: list[str] = []
    for part in parts:
        if len(part) <= chunk_size:
            final.append(part)
        elif percentile > SEMANTIC_BREAKPOINT_MIN:
            final.extend(
                _semantic_split_within_article(
                    part, chunk_size, percentile - SEMANTIC_BREAKPOINT_STEP
                )
            )
        else:
            final.extend(_split_by_khoan_fallback(part, chunk_size))

    return final


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
) -> list[Document]:
    """
  Pipeline chunking cho Luật Lao động.

    Tầng 1: Mỗi Điều (hoặc phần mở đầu) = 1 đơn vị, metadata Chương/Mục/Điều.
    Tầng 2: Điều quá dài → semantic embedding trong phạm vi Điều đó.
    """
    chunk_size = chunk_size or CHUNK_SIZE
    full_text = "\n\n".join(doc.page_content for doc in documents)
    source = documents[0].metadata.get("source", "unknown") if documents else "unknown"

    structured_chunks = extract_structure(full_text)
    print(f"  📋 Tìm thấy {len(structured_chunks)} Điều/phần (tầng cấu trúc)")

    final_documents = []
    chunk_id = 0
    semantic_split_articles = 0

    for item in structured_chunks:
        content = item["content"]
        base_meta = {
            "source": source,
            "chapter": item["chapter"],
            "section": item["section"],
            "article": item["article"],
            "title": item["title"],
        }

        if len(content) <= chunk_size:
            final_documents.append(
                Document(
                    page_content=content,
                    metadata={
                        **base_meta,
                        "chunk_id": f"chunk_{chunk_id}",
                        "split_layer": "article",
                    },
                )
            )
            chunk_id += 1
            continue

        semantic_split_articles += 1
        sub_chunks = _semantic_split_within_article(content, chunk_size)
        for j, sub_content in enumerate(sub_chunks):
            final_documents.append(
                Document(
                    page_content=sub_content,
                    metadata={
                        **base_meta,
                        "chunk_id": f"chunk_{chunk_id}",
                        "sub_chunk": f"{j + 1}/{len(sub_chunks)}",
                        "split_layer": "semantic",
                    },
                )
            )
            chunk_id += 1

    kept_whole = len(structured_chunks) - semantic_split_articles
    print(f"  ✂️  Tổng chunks: {len(final_documents)}")
    print(f"     - Điều giữ nguyên: {kept_whole}")
    print(f"     - Điều chia semantic: {semantic_split_articles}")
    avg_len = sum(len(d.page_content) for d in final_documents) / max(
        len(final_documents), 1
    )
    print(f"  📏 Độ dài trung bình: {avg_len:.0f} ký tự/chunk")

    return final_documents


if __name__ == "__main__":
    from src.indexing.loader import load_all_pdfs

    print("=" * 60)
    print("TEST: Module Chunker")
    print("=" * 60)

    docs = load_all_pdfs()
    chunks = chunk_documents(docs)

    print(f"\n--- 3 Chunks đầu tiên ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n[Chunk {i}]")
        print(f"  Metadata: {chunk.metadata}")
        print(f"  Content ({len(chunk.page_content)} chars):")
        print(f"  {chunk.page_content[:150]}...")