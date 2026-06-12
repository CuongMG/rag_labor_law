"""
Module Generator — Sử dụng LLM để sinh câu trả lời RAG.

Sử dụng Google Gemini (gemini-3.5-flash) thông qua langchain_google_genai.
Nhiệm vụ: Nhận query của người dùng + các đoạn văn bản luật liên quan
và tạo ra câu trả lời tiếng Việt chính xác, khách quan, có trích dẫn rõ ràng.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import GEMINI_API_KEY, GENERATION_MODEL

class AnswerGenerator:
    """Tạo câu trả lời RAG dựa trên thông tin bối cảnh pháp luật được trích xuất."""

    def __init__(self):
        """Khởi tạo Gemini Chat model."""
        if not GEMINI_API_KEY:
            raise ValueError(
                "Không tìm thấy GEMINI_API_KEY. Vui lòng cấu hình trong file .env"
            )

        # Sử dụng temperature thấp (ví dụ: 0.1 hoặc 0.2) để tránh LLM "sáng tạo" quá mức,
        # đảm bảo câu trả lời mang tính chính xác và tuân thủ chặt chẽ tài liệu gốc.
        self.llm = ChatGoogleGenerativeAI(
            model=GENERATION_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
        )

    def format_context(self, retrieved_chunks: list[dict]) -> str:
        """
        Định dạng danh sách các chunks thành 1 khối text context để đưa vào prompt.

        Args:
            retrieved_chunks: Danh sách kết quả từ retriever.

        Returns:
            str: Khối text chứa toàn bộ nội dung luật kèm nguồn Điều/Chương.
        """
        formatted = []
        for i, chunk in enumerate(retrieved_chunks):
            meta = chunk["metadata"]
            source_info = f"--- Đoạn bối cảnh {i+1} [Nguồn: Điều {meta['article']} - {meta['title']} | Chương {meta['chapter']}] ---"
            content = chunk["content"]
            formatted.append(f"{source_info}\n{content}\n")
            
        return "\n".join(formatted)

    def generate_answer(self, query: str, retrieved_chunks: list[dict]) -> str:
        """
        Sinh câu trả lời dựa trên query và các chunks liên quan đã tìm được.

        Args:
            query: Câu hỏi của người dùng.
            retrieved_chunks: Danh sách các đoạn văn bản luật liên quan nhất.

        Returns:
            str: Câu trả lời pháp lý hoàn chỉnh.
        """
        if not retrieved_chunks:
            return (
                "Xin lỗi, tôi không tìm thấy tài liệu luật nào liên quan đến câu hỏi của bạn "
                "để có thể trả lời chính xác."
            )

        # 1. Chuẩn bị context
        context_str = self.format_context(retrieved_chunks)

        # 2. Xây dựng prompt với system instruction chi tiết
        system_instruction = (
            "Bạn là một trợ lý pháp lý AI chuyên nghiệp, am hiểu sâu sắc về Bộ Luật Lao Động Việt Nam.\n"
            "Hãy trả lời câu hỏi của người dùng một cách chính xác, khách quan dựa trên THÔNG TIN BỐI CẢNH được cung cấp dưới đây.\n\n"
            "Quy tắc trả lời:\n"
            "1. Chỉ trả lời dựa vào thông tin có sẵn trong THÔNG TIN BỐI CẢNH. Không sử dụng kiến thức bên ngoài.\n"
            "2. Trích dẫn cơ sở pháp lý rõ ràng trong câu trả lời (ví dụ: 'Theo Khoản 1 Điều 105...').\n"
            "3. Nếu thông tin bối cảnh không đủ hoặc không liên quan để trả lời câu hỏi, hãy lịch sự phản hồi: "
            "'Dựa trên tài liệu luật được cung cấp, tôi chưa có đủ thông tin để trả lời chính xác câu hỏi này.' "
            "Tuyệt đối không được bịa đặt (hallucinate) điều khoản luật.\n"
            "4. Định dạng câu trả lời rõ ràng bằng markdown, có gạch đầu dòng các ý lớn để người dùng dễ đọc.\n\n"
            f"--- THÔNG TIN BỐI CẢNH ---\n{context_str}"
        )

        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=f"Câu hỏi: {query}"),
        ]

        # 3. Gọi model
        try:
            response = self.llm.invoke(messages)
            # → chuyển về string để app.py có thể append nguồn tham khảo
            if isinstance(response.content, list):
                parts = []
                for part in response.content:
                    if isinstance(part, dict) and "text" in part:
                        parts.append(part["text"])
                return "".join(parts)
            return str(response.content)
        except Exception as e:
            return f"❌ Đã xảy ra lỗi khi gọi Gemini API để sinh câu trả lời: {e}"


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Module Generator")
    print("=" * 60)

    try:
        generator = AnswerGenerator()
        
        # Giả lập kết quả search mẫu
        dummy_chunks = [
            {
                "content": "Điều 105. Thời giờ làm việc bình thường\n1. Thời giờ làm việc bình thường không quá 08 giờ trong 01 ngày và không quá 48 giờ trong 01 tuần.",
                "metadata": {"article": "105", "title": "Thời giờ làm việc bình thường", "chapter": "VII"}
            }
        ]
        
        query = "Thời giờ làm việc bình thường được quy định thế nào?"
        print(f"Query: {query}")
        
        answer = generator.generate_answer(query, dummy_chunks)
        print("\n--- GEMINI RAG ANSWER ---")
        print(answer)

    except Exception as e:
        print(f"❌ Lỗi: {e}")