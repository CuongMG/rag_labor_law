"""
Gradio Chat UI — Hỏi đáp về Bộ Luật Lao Động Việt Nam.

Chạy:
    python app.py
    # → Mở http://localhost:7860
"""

import gradio as gr
from src.retrieval.retriever import Retriever
from src.retrieval.generator import AnswerGenerator

retriever = Retriever()
generator = AnswerGenerator()


def respond(message, history):
    if not message.strip():
        return ""
    results = retriever.retrieve(message, top_k=3)
    answer = generator.generate_answer(message, results)

    if results:
        sources_list = [f"- Điều {r['metadata']['article']} ({r['metadata']['title']})" for r in results]
        answer += "\n\n📚 Nguồn: " + " | ".join(sources_list)
    return answer


demo = gr.ChatInterface(
    respond,
    title="⚖️ RAG Labor Law — Hỏi đáp Luật Lao Động",
    description="Hỏi về Bộ Luật Lao Động Việt Nam 2019.",
    examples=[
        "Thời giờ làm việc bình thường là bao nhiêu giờ?",
        "Nghỉ phép năm bao nhiêu ngày?",
        "Hợp đồng lao động có những loại nào?",
    ],
)

if __name__ == "__main__":
    demo.launch()