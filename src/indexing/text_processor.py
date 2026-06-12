"""
Module Xử Lý Text NLP — Pipeline 6 bước cho tài liệu pháp luật tiếng Việt.

Pipeline:
  Raw Text → [1] Unicode Normalize
           → [2] Clean Text (noise removal)
           → [3] Word Segmentation (tách từ tiếng Việt)
           → [4] POS Tagging (gán nhãn từ loại)
           → [5] NER - Extract Legal Entities
           → [6] Stopword Filtering
           → Processed text + metadata

Mỗi bước là 1 kỹ thuật NLP quan trọng, có docstring giải thích.
"""

import re
import unicodedata
from underthesea import word_tokenize, pos_tag


# ============================================================
# BƯỚC 1: Unicode Normalization
# ============================================================

def normalize_unicode(text: str) -> str:
    """
    NLP Concept: Unicode Normalization (NFC).

    Tiếng Việt có 2 cách biểu diễn dấu trong Unicode:
    - Precomposed (NFC): "ă" = 1 ký tự (U+0103)
    - Decomposed (NFD): "ă" = "a" + "◌̆" = 2 ký tự (U+0061 + U+0306)

    Chuẩn hóa về NFC để:
    - So sánh text chính xác (search, dedup)
    - Regex hoạt động đúng với tiếng Việt
    - Embedding nhất quán

    Args:
        text: Text gốc có thể chứa Unicode không nhất quán.

    Returns:
        Text đã chuẩn hóa NFC.
    """
    return unicodedata.normalize("NFC", text)


# ============================================================
# BƯỚC 2: Text Cleaning (Noise Removal)
# ============================================================

def clean_text(text: str) -> str:
    """
    NLP Concept: Text Cleaning / Noise Removal.

    Loại bỏ noise từ quá trình extract PDF:
    - Số trang, header/footer lặp lại
    - Ký tự đặc biệt thừa từ PDF encoding
    - Khoảng trắng thừa, dòng trống liên tiếp
    - Sửa lỗi xuống dòng giữa câu (PDF line wrapping)

    Đây là bước quan trọng nhất vì "garbage in = garbage out".
    Text bẩn → embedding sai → search kém.

    Args:
        text: Text thô từ PDF loader.

    Returns:
        Text đã được làm sạch.
    """
    # --- Xóa số trang ---
    # Dạng "- 1 -", "— 5 —"
    text = re.sub(r"[-—]\s*\d+\s*[-—]", "", text)
    # Dạng "Trang 5", "trang 12"
    text = re.sub(r"[Tt]rang\s+\d+", "", text)

    # --- Sửa lỗi PDF line wrapping ---
    # Trường hợp từ bị ngắt giữa bởi dấu gạch ngang + xuống dòng
    # Ví dụ: "người lao\n-động" → "người lao động"  (chưa đúng lắm)
    # Nhưng: "lao-\nđộng" → "laođộng"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Gộp dòng bị ngắt giữa câu (dòng không kết thúc bằng dấu câu
    # và dòng tiếp theo không bắt đầu bằng chữ hoa/số/heading)
    text = re.sub(r"(?<![.!?:;\n])\n(?![A-ZĐ\d\n])", " ", text)

    # --- Chuẩn hóa khoảng trắng ---
    # Nhiều space/tab liên tiếp → 1 space
    text = re.sub(r"[ \t]+", " ", text)
    # Nhiều dòng trống liên tiếp → tối đa 2 dòng
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Xóa space đầu/cuối mỗi dòng
    text = "\n".join(line.strip() for line in text.split("\n"))

    return text.strip()


# ============================================================
# BƯỚC 3: Vietnamese Word Segmentation (Tách từ)
# ============================================================

def segment_words(text: str) -> str:
    """
    NLP Concept: Word Segmentation / Tokenization.

    Đặc thù tiếng Việt: 1 từ có thể gồm nhiều âm tiết (syllables).
    Ví dụ:
    - "người lao động" = 1 từ (3 âm tiết) → "người_lao_động"
    - "hợp đồng" = 1 từ (2 âm tiết) → "hợp_đồng"
    - "Bộ luật" = 1 từ (2 âm tiết) → "Bộ_luật"

    Nếu không tách từ, model sẽ hiểu sai:
    - "người" + "lao" + "động" (3 từ riêng lẻ, sai nghĩa)
    vs
    - "người_lao_động" (1 từ, đúng nghĩa)

    Library underthesea dùng model ML (CRF/BiLSTM) để nhận diện
    ranh giới từ dựa trên ngữ cảnh.

    Args:
        text: Text tiếng Việt chưa tách từ.

    Returns:
        Text đã tách từ, từ ghép nối bằng "_".
    """
    # Tách theo từng đoạn (paragraph) để giữ cấu trúc
    paragraphs = text.split("\n")
    segmented_paragraphs = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            segmented_paragraphs.append("")
            continue

        # word_tokenize với format="text" trả về string
        # với từ ghép nối bằng "_"
        try:
            segmented = word_tokenize(para, format="text")
            segmented_paragraphs.append(segmented)
        except Exception:
            # Fallback: giữ nguyên nếu tách từ lỗi
            segmented_paragraphs.append(para)

    return "\n".join(segmented_paragraphs)


# ============================================================
# BƯỚC 4: POS Tagging (Gán nhãn từ loại)
# ============================================================

def analyze_pos(text: str) -> list[tuple[str, str]]:
    """
    NLP Concept: Part-of-Speech (POS) Tagging.

    Gán nhãn từ loại cho mỗi từ trong câu:
    - N (Noun): Danh từ — "người lao động", "hợp đồng"
    - V (Verb): Động từ — "ký kết", "chấm dứt"
    - A (Adjective): Tính từ — "hợp pháp", "bắt buộc"
    - R (Adverb): Phó từ — "không", "đã", "sẽ"
    - E (Preposition): Giới từ — "của", "trong", "theo"
    - Np (Proper noun): Danh từ riêng — "Việt Nam"

    Ứng dụng trong RAG:
    - Trích xuất key noun phrases (N, Np) để làm keywords
    - Phân tích cấu trúc câu trong văn bản pháp luật
    - Hiểu context tốt hơn

    Args:
        text: Text tiếng Việt (nên là 1 câu hoặc đoạn ngắn).

    Returns:
        List of (word, tag) tuples.

    Example:
        >>> analyze_pos("Người lao động có quyền nghỉ phép")
        [('Người lao động', 'N'), ('có', 'V'),
         ('quyền', 'N'), ('nghỉ phép', 'V')]
    """
    try:
        return pos_tag(text)
    except Exception:
        return []


def extract_key_phrases(pos_tags: list[tuple[str, str]]) -> list[str]:
    """
    Trích xuất cụm danh từ quan trọng từ POS tags.

    Lọc ra các từ có tag N (danh từ), Np (danh từ riêng)
    — đây thường là các khái niệm pháp lý quan trọng.

    Args:
        pos_tags: Output từ analyze_pos().

    Returns:
        Danh sách các danh từ/cụm danh từ quan trọng.
    """
    noun_tags = {"N", "Np", "Nc", "Nu"}
    return [word for word, tag in pos_tags if tag in noun_tags]


# ============================================================
# BƯỚC 5: Named Entity Recognition (NER) cho văn bản pháp luật
# ============================================================

def extract_legal_entities(text: str) -> dict[str, list[str]]:
    """
    NLP Concept: Named Entity Recognition (NER).

    Trích xuất thực thể pháp luật bằng regex patterns.
    Đây là NER dựa trên luật (rule-based), khác với NER dựa trên
    ML (model-based) nhưng phù hợp với văn bản có cấu trúc rõ ràng.

    Các loại thực thể:
    - Điều khoản: "Điều 113", "Điều 5"
    - Chương: "Chương VII", "Chương I"
    - Mục: "Mục 1", "Mục 2"
    - Khoản: "khoản 1", "khoản 2"
    - Điểm: "điểm a", "điểm b"
    - Ngày tháng: "01/01/2021"
    - Số hiệu văn bản: "45/2019/QH14"

    Metadata này cực kỳ hữu ích cho:
    - Filtering kết quả search (chỉ tìm trong Chương III)
    - Hiển thị nguồn trích dẫn chính xác
    - Cross-reference giữa các điều khoản

    Args:
        text: Text pháp luật tiếng Việt.

    Returns:
        Dict với key là loại thực thể, value là danh sách giá trị.
    """
    entities = {
        "articles": re.findall(r"Điều\s+(\d+)", text),
        "chapters": re.findall(r"Chương\s+([IVXLCDM]+|\d+)", text),
        "sections": re.findall(r"Mục\s+(\d+)", text),
        "clauses": re.findall(r"[Kk]hoản\s+(\d+)", text),
        "points": re.findall(r"[Đđ]iểm\s+([a-zđ])", text),
        "dates": re.findall(r"\d{1,2}/\d{1,2}/\d{4}", text),
        "legal_refs": re.findall(r"\d+/\d{4}/[A-ZĐ\-]+\d*", text),
    }

    # Loại bỏ duplicates, giữ thứ tự
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))

    return entities


# ============================================================
# BƯỚC 6: Stopword Removal
# ============================================================

# Danh sách stopwords tiếng Việt phổ biến
# ⚠️ LƯU Ý: Trong pháp luật, "hoặc" và "và" có ý nghĩa khác nhau!
# "hoặc" = OR (một trong các điều kiện)
# "và" = AND (tất cả điều kiện)
# → Chỉ dùng stopword removal cho BM25, KHÔNG cho embedding
VIETNAMESE_STOPWORDS = {
    "là", "của", "các", "có", "được", "cho", "trong",
    "với", "này", "đã", "để", "từ", "về", "do", "khi",
    "tại", "theo", "đến", "bởi", "trên", "sau",
    "cũng", "như", "nhưng", "nếu", "thì", "mà", "sẽ",
    "còn", "vào", "ra", "lên", "rằng", "bị", "nên",
    "đó", "đây", "ấy", "mỗi", "vẫn", "rất", "lại",
    "đang", "chỉ", "hơn", "nào", "gì", "ai",
}


def remove_stopwords(
    text: str,
    stopwords: set[str] | None = None,
) -> str:
    """
    NLP Concept: Stopword Removal.

    Loại bỏ từ phổ biến không mang nhiều ngữ nghĩa phân biệt.

    Tác dụng:
    - Giảm noise cho BM25/keyword search
    - Giúp focus vào từ quan trọng

    ⚠️ KHÔNG dùng cho embedding input vì:
    - Embedding model đã học cách xử lý stopwords
    - Loại bỏ có thể làm mất ngữ nghĩa

    Args:
        text: Text input.
        stopwords: Set các stopwords. Nếu None, dùng default.

    Returns:
        Text đã loại bỏ stopwords.
    """
    if stopwords is None:
        stopwords = VIETNAMESE_STOPWORDS

    words = text.split()
    filtered = [w for w in words if w.lower() not in stopwords]
    return " ".join(filtered)


def preprocess_for_embedding(text: str) -> str:
    """Chuẩn hóa Unicode + làm sạch — dùng cho dense embedding."""
    return clean_text(normalize_unicode(text))


def preprocess_for_bm25(text: str) -> str:
    """Chuẩn hóa + làm sạch + bỏ stopword — dùng cho sparse BM25."""
    return remove_stopwords(preprocess_for_embedding(text))


# ============================================================
# PIPELINE TỔNG HỢP
# ============================================================

class TextProcessor:
    """
    Pipeline xử lý text NLP cho tài liệu pháp luật tiếng Việt.

    Kết hợp 6 bước NLP thành 1 pipeline hoàn chỉnh.
    Có thể bật/tắt từng bước tùy nhu cầu.
    """

    def __init__(
        self,
        do_segment: bool = True,
        do_pos: bool = True,
        do_ner: bool = True,
        do_stopwords: bool = True,
    ):
        """
        Args:
            do_segment: Bật tách từ tiếng Việt.
            do_pos: Bật POS tagging.
            do_ner: Bật trích xuất thực thể pháp luật.
            do_stopwords: Bật loại bỏ stopwords.
        """
        self.do_segment = do_segment
        self.do_pos = do_pos
        self.do_ner = do_ner
        self.do_stopwords = do_stopwords

    def process(self, text: str) -> dict:
        """
        Chạy full pipeline xử lý text.

        Returns:
            Dict chứa các phiên bản text đã xử lý:
            - cleaned_text: text sạch (cho embedding)
            - segmented_text: text đã tách từ
            - keywords_text: text không stopwords (cho BM25)
            - entities: dict thực thể pháp luật (cho metadata)
            - pos_tags: POS tags (cho phân tích)
            - stats: thống kê xử lý
        """
        result = {}

        # Bước 1: Unicode normalize (luôn chạy)
        text = normalize_unicode(text)

        # Bước 2: Clean text (luôn chạy)
        cleaned = clean_text(text)
        result["cleaned_text"] = cleaned

        # Bước 3: Word segmentation
        if self.do_segment:
            result["segmented_text"] = segment_words(cleaned)
        else:
            result["segmented_text"] = cleaned

        # Bước 4: POS tagging (chạy trên đoạn ngắn để demo)
        if self.do_pos:
            # Chỉ POS tag 3 câu đầu (tránh chậm với text dài)
            sample = ". ".join(cleaned.split(".")[:3])
            pos_tags = analyze_pos(sample)
            result["pos_tags"] = pos_tags
            result["key_phrases"] = extract_key_phrases(pos_tags)
        else:
            result["pos_tags"] = []
            result["key_phrases"] = []

        # Bước 5: NER - Extract legal entities
        if self.do_ner:
            result["entities"] = extract_legal_entities(cleaned)
        else:
            result["entities"] = {}

        # Bước 6: Stopword removal (cho BM25)
        if self.do_stopwords:
            result["keywords_text"] = remove_stopwords(cleaned)
        else:
            result["keywords_text"] = cleaned

        # Thống kê
        result["stats"] = {
            "original_length": len(text),
            "cleaned_length": len(cleaned),
            "reduction_pct": round(
                (1 - len(cleaned) / max(len(text), 1)) * 100, 1
            ),
            "num_entities": sum(
                len(v) for v in result.get("entities", {}).values()
            ),
        }

        return result


# ============================================================
# Chạy trực tiếp để test module
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Module Text Processor")
    print("=" * 60)

    # Text mẫu từ Bộ Luật Lao Động
    sample_text = """
    Chương VII
    THỜI GIỜ LÀM VIỆC, THỜI GIỜ NGHỈ NGƠI

    Mục 1
    THỜI GIỜ LÀM VIỆC

    Điều 105. Thời giờ làm việc bình thường
    1. Thời giờ làm việc bình thường không quá 08 giờ trong 01 ngày
    và không quá 48 giờ trong 01 tuần.
    2. Người sử dụng lao động có quyền quy định thời giờ làm việc
    theo ngày hoặc theo tuần nhưng phải thông báo cho người lao động
    biết; trường hợp theo tuần thì thời giờ làm việc bình thường
    không quá 10 giờ trong 01 ngày và không quá 48 giờ trong 01 tuần.
    Nhà nước khuyến khích người sử dụng lao động thực hiện tuần làm
    việc 40 giờ đối với người lao động.

    Điều 113. Nghỉ hằng năm
    1. Người lao động làm việc đủ 12 tháng cho một người sử dụng
    lao động thì được nghỉ hằng năm, hưởng nguyên lương theo hợp
    đồng lao động như sau:
    a) 12 ngày làm việc đối với người làm công việc trong điều kiện
    bình thường;
    b) 14 ngày làm việc đối với người lao động chưa thành niên,
    lao động là người khuyết tật, người làm nghề, công việc nặng
    nhọc, độc hại, nguy hiểm;
    """

    processor = TextProcessor()
    result = processor.process(sample_text)

    print("\n📊 THỐNG KÊ:")
    for key, value in result["stats"].items():
        print(f"  {key}: {value}")

    print(f"\n📝 TEXT ĐÃ CLEAN (200 ký tự đầu):")
    print(f"  {result['cleaned_text'][:200]}")

    print(f"\n🔤 TEXT ĐÃ TÁCH TỪ (200 ký tự đầu):")
    print(f"  {result['segmented_text'][:200]}")

    print(f"\n🏷️  POS TAGS (10 từ đầu):")
    for word, tag in result["pos_tags"][:10]:
        print(f"  {word:20s} → {tag}")

    print(f"\n🔑 KEY PHRASES:")
    print(f"  {result['key_phrases'][:10]}")

    print(f"\n📌 LEGAL ENTITIES:")
    for etype, values in result["entities"].items():
        if values:
            print(f"  {etype}: {values}")

    print(f"\n🚫 TEXT KHÔNG STOPWORDS (200 ký tự đầu):")
    print(f"  {result['keywords_text'][:200]}")