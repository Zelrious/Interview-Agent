"""LangChain Agent Tools for Interview Agent RAG Knowledge Harness.

Implements:
- search_interview_questions_tool: for Interviewer Agent (UC-02), locked to caller_role='interviewer'
- retrieve_grading_rubric_tool: for Grader Agent (UC-05/06), locked to caller_role='grader'
"""

from __future__ import annotations

import threading
from typing import Any

from langchain_core.tools import tool

from src.rag.hybrid_retriever import HybridKnowledgeRetriever
from src.rag.ingest import load_and_chunk_knowledge_base

_retriever_instance: HybridKnowledgeRetriever | None = None
_retriever_lock = threading.Lock()


def get_shared_retriever() -> HybridKnowledgeRetriever:
    """Khởi tạo singleton HybridKnowledgeRetriever từ dữ liệu data/knowledge."""
    global _retriever_instance
    with _retriever_lock:
        if _retriever_instance is None:
            chunks = load_and_chunk_knowledge_base(
                data_dir="data/knowledge", add_contextual_prefix=True
            )
            _retriever_instance = HybridKnowledgeRetriever(documents=chunks)
        return _retriever_instance


def set_shared_retriever(retriever: HybridKnowledgeRetriever) -> None:
    """Ghi đè retriever dùng trong tests hoặc custom environment."""
    global _retriever_instance
    with _retriever_lock:
        _retriever_instance = retriever


@tool
def search_interview_questions_tool(
    query: str,
    competency: str | None = None,
    level: str | None = None,
    top_k: int = 5,
) -> str:
    """Tìm kiếm câu hỏi phỏng vấn trong Ngân hàng câu hỏi cho Interviewer Agent (UC-02).

    Công cụ này được khóa cứng với vai trò 'interviewer': tuyệt đối KHÔNG trả về
    rubric chấm điểm mật hay các tín hiệu mong đợi (expected_signals) của câu hỏi.

    Args:
        query: Từ khóa hoặc mô tả nội dung câu hỏi cần tìm.
        competency: Mã năng lực cần lọc (ví dụ: BE_API_DESIGN, BE_DB_OPTIMIZATION) hoặc None.
        level: Cấp độ ứng viên (intern, junior, middle, senior) hoặc None.
        top_k: Số lượng câu hỏi tối đa cần lấy về (mặc định 5).
    """
    retriever = get_shared_retriever()
    metadata_filter: dict[str, Any] = {"source_type": "question", "status": "active"}
    if competency:
        metadata_filter["competency"] = competency
    if level:
        metadata_filter["level"] = level

    results = retriever.retrieve(
        query=query,
        top_n=top_k,
        caller_role="interviewer",  # KHÓA CỨNG VAI TRÒ INTERVIEWER
        metadata_filter=metadata_filter,
    )

    if not results:
        return "Không tìm thấy câu hỏi phù hợp trong ngân hàng câu hỏi."

    output_lines = [f"Tìm thấy {len(results)} câu hỏi phù hợp:"]
    for i, doc in enumerate(results, 1):
        cid = doc.metadata.get("chunk_id", "")
        item_id = doc.metadata.get("item_id", "")
        output_lines.append(
            f"\n--- [Câu hỏi {i} | Mã: {item_id} | Trích dẫn: [{cid}]] ---"
        )
        output_lines.append(doc.page_content)

    return "\n".join(output_lines)


@tool
def retrieve_grading_rubric_tool(
    competency_id: str,
    query: str | None = None,
) -> str:
    """Truy xuất Rubric chấm điểm mật dành riêng cho Grader Agent (UC-05 & UC-06).

    Công cụ này được cấp quyền vai trò 'grader': trả về thang điểm chi tiết 1-5,
    các tín hiệu mong đợi và ví dụ trả lời mẫu kèm mã trích dẫn [chunk_id].

    Args:
        competency_id: Mã năng lực cần lấy Rubric (ví dụ: BE_API_DESIGN, DA_SQL_ANALYTICS, QA_AUTOMATION).
        query: Chuỗi truy vấn bổ sung (nếu muốn tìm rubric theo ngữ cảnh cụ thể).
    """
    retriever = get_shared_retriever()
    q = query or f"Rubric {competency_id}"

    results = retriever.retrieve(
        query=q,
        top_n=3,
        caller_role="grader",  # KHÓA CỨNG VAI TRÒ GRADER
        metadata_filter={"source_type": "rubric", "competency": competency_id},
    )

    if not results:
        # Fallback thử tìm không lọc cứng competency
        results = retriever.retrieve(
            query=f"Rubric {competency_id}",
            top_n=1,
            caller_role="grader",
            metadata_filter={"source_type": "rubric"},
        )

    if not results:
        return f"Không tìm thấy rubric mật cho năng lực '{competency_id}'."

    best_rubric = results[0]
    cid = best_rubric.metadata.get("chunk_id", "")
    return f"=== RUBRIC CHẤM ĐIỂM [{cid}] ===\n{best_rubric.page_content}"
