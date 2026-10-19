"""LangChain Agent Tools for Interview Agent RAG Knowledge Harness.

Implements:
- search_interview_questions_tool: for Interviewer Agent (UC-02), locked to caller_role='interviewer'
- retrieve_grading_rubric_tool: for Grader Agent (UC-05/06), locked to caller_role='grader'
- rewrite_query_for_retrieval: query normalization & keyword expansion for robust retrieval
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

from langchain_core.tools import tool

from src.rag.hybrid_retriever import HybridKnowledgeRetriever
from src.rag.ingest import load_and_chunk_knowledge_base

logger = logging.getLogger(__name__)

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


def rewrite_query_for_retrieval(
    query: str,
    competency: str | None = None,
    level: str | None = None,
) -> str:
    """Chuẩn hóa và làm giàu câu truy vấn cho RAG retriever.

    1. Loại bỏ các tiền tố hội thoại rác (conversational fillers).
    2. Chuẩn hóa các từ viết tắt kỹ thuật thông dụng (db, mq, cache, k8s, api).
    3. Bổ sung mã năng lực và cấp độ để tăng độ nhạy cho cả Dense Vector và BM25.
    """
    clean_q = query.strip()

    # 1. Loại bỏ tiền tố và từ dừng hội thoại
    filler_patterns = [
        r"^(hãy\s+)?(tìm|cho tôi|lấy|cần tìm|cần hỏi|đưa ra)\s+(câu hỏi|ngân hàng câu hỏi)?\s*(về|liên quan đến)?\s*",
        r"\bcâu hỏi phỏng vấn\s*(về)?\s*",
        r"\bcho (ứng viên|vị trí|bạn)\s*",
    ]
    for pattern in filler_patterns:
        clean_q = re.sub(pattern, "", clean_q, flags=re.IGNORECASE).strip()

    # 2. Chuẩn hóa từ viết tắt kỹ thuật
    synonyms = {
        r"\bdb\b": "database cơ sở dữ liệu",
        r"\bmq\b": "message queue hàng đợi",
        r"\bcache\b": "caching redis bộ nhớ đệm",
        r"\bk8s\b": "kubernetes",
        r"\bapi\b": "api restful",
    }
    for pattern, replacement in synonyms.items():
        clean_q = re.sub(pattern, replacement, clean_q, flags=re.IGNORECASE)

    # 3. Làm giàu ngữ cảnh với competency và level
    context_tokens = []
    if competency and competency.lower() not in clean_q.lower():
        context_tokens.append(competency)
    if level and level.lower() not in clean_q.lower():
        context_tokens.append(f"cấp {level}")

    if context_tokens:
        clean_q = f"{' '.join(context_tokens)} {clean_q}".strip()

    return clean_q or query


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
        competency: Mã năng lực cần lọc (ví dụ: BE_API_DESIGN, BE_DB_OPTIMIZATION, BE_CACHING_SYSTEMS) hoặc None.
        level: Cấp độ ứng viên (intern, junior, middle, senior) hoặc None.
        top_k: Số lượng câu hỏi tối đa cần lấy về (mặc định 5).
    """
    retriever = get_shared_retriever()
    metadata_filter: dict[str, Any] = {"source_type": "question", "status": "active"}
    if competency:
        metadata_filter["competency"] = competency
    if level:
        metadata_filter["level"] = level

    # BƯỚC 1: Chuẩn hóa câu truy vấn (Query Rewriting)
    normalized_query = rewrite_query_for_retrieval(
        query=query, competency=competency, level=level
    )

    # BƯỚC 2: Truy xuất với chốt chặn khóa cứng caller_role="interviewer"
    results = retriever.retrieve(
        query=normalized_query,
        top_n=top_k,
        caller_role="interviewer",  # KHÓA CỨNG VAI TRÒ INTERVIEWER
        metadata_filter=metadata_filter,
    )

    if not results:
        return "Không tìm thấy câu hỏi phù hợp trong ngân hàng câu hỏi."

    output_lines = [f"Tìm thấy {len(results)} câu hỏi phù hợp:"]
    for i, doc in enumerate(results, 1):
        # BƯỚC 3: Kiểm tra phòng hộ thứ cấp (Defensive Security Check)
        meta = doc.metadata or {}
        if meta.get("is_confidential") is True or meta.get("source_type") == "rubric":
            continue

        cid = meta.get("chunk_id", "")
        item_id = meta.get("item_id", "")

        # Bảo đảm tuyệt đối không lọt expected_signals ra nội dung trả về
        clean_content = doc.page_content
        if "expected_signals" in clean_content:
            clean_content = clean_content.split("expected_signals")[0].strip()

        output_lines.append(
            f"\n--- [Câu hỏi {i} | Mã: {item_id} | Trích dẫn: [{cid}]] ---"
        )
        output_lines.append(clean_content)

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
        competency_id: Mã năng lực cần lấy Rubric (ví dụ: BE_API_DESIGN, BE_CACHING_SYSTEMS, BE_SYSTEM_DESIGN).
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
