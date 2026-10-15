"""Unit Tests for RAG Knowledge Harness (Buổi 06 - Đề tài 05).

Covers:
1. upsert_file_idempotent: Idempotency of document chunking
2. verify_embedding_lock: Model mismatch detection
3. Security boundary: Role-based confidentiality filtering (interviewer vs grader)
4. Reciprocal Rank Fusion (RRF): Consensus ranking
5. verify_citations: Grounding and hallucination detection
6. Agent Tools: search_interview_questions_tool and retrieve_grading_rubric_tool
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.rag.agent_tools import (
    retrieve_grading_rubric_tool,
    search_interview_questions_tool,
)
from src.rag.grounding import ESCAPE_HATCH_TEXT, verify_citations
from src.rag.hybrid_retriever import (
    HybridKnowledgeRetriever,
    reciprocal_rank_fusion,
)
from src.rag.ingest import (
    upsert_file_idempotent,
    verify_embedding_lock,
)

# ==============================================================================
# Test 1: upsert_file_idempotent
# ==============================================================================


def test_upsert_file_idempotent() -> None:
    """Nạp 2 lần cùng một file/doc_id không làm tăng số lượng chunk trong kho."""
    mock_store: list[Document] = []
    doc_id = "doc123456789"

    chunk1 = Document(
        page_content="Chunk 1 content",
        metadata={"doc_id": doc_id, "chunk_id": f"{doc_id}:0"},
    )
    chunk2 = Document(
        page_content="Chunk 2 content",
        metadata={"doc_id": doc_id, "chunk_id": f"{doc_id}:1"},
    )

    # Nạp lần 1
    upsert_file_idempotent(mock_store, doc_id, [chunk1, chunk2])
    assert len(mock_store) == 2

    # Nạp lần 2 với cùng doc_id
    updated_chunk1 = Document(
        page_content="Chunk 1 updated",
        metadata={"doc_id": doc_id, "chunk_id": f"{doc_id}:0"},
    )
    updated_chunk2 = Document(
        page_content="Chunk 2 updated",
        metadata={"doc_id": doc_id, "chunk_id": f"{doc_id}:1"},
    )
    upsert_file_idempotent(mock_store, doc_id, [updated_chunk1, updated_chunk2])

    # Số lượng chunk vẫn là 2, không bị nhân bản
    assert len(mock_store) == 2
    assert mock_store[0].page_content == "Chunk 1 updated"


# ==============================================================================
# Test 2: verify_embedding_lock
# ==============================================================================


def test_verify_embedding_lock() -> None:
    """Truyền sai tên embedding model phải ném AssertionError."""
    valid_meta = {"embedding_model": "local-deterministic-minilm"}

    # Khớp model -> Thành công
    verify_embedding_lock(valid_meta, "local-deterministic-minilm")

    # Không khớp model -> Bắt buộc ném AssertionError
    with pytest.raises(AssertionError) as exc_info:
        verify_embedding_lock(valid_meta, "text-embedding-3-small")
    assert "Embedding Lock Mismatch" in str(exc_info.value)


# ==============================================================================
# Test 3: Role-based Security Boundary (Interviewer vs Grader)
# ==============================================================================


def test_security_boundary_interviewer_vs_grader() -> None:
    """Interviewer tuyệt đối không nhận rubric hay expected_signals; Grader được phép."""
    doc_question = Document(
        page_content="Bạn hãy giải thích cơ chế GIL trong Python?",
        metadata={
            "chunk_id": "q1:0",
            "item_id": "Q_PY_01",
            "source_type": "question",
            "is_confidential": False,
            "expected_signals_json": '["GIL ngăn nhiều thread chạy song song bytecode", "Sử dụng multiprocessing thay thế"]',
        },
    )
    doc_rubric = Document(
        page_content="Rubric chấm điểm mật cho GIL: Mức 1: Không biết GIL...",
        metadata={
            "chunk_id": "r1:0",
            "item_id": "RUBRIC_PY_01",
            "source_type": "rubric",
            "is_confidential": True,
        },
    )

    retriever = HybridKnowledgeRetriever(documents=[doc_question, doc_rubric])

    # 1. Truy vấn với vai trò INTERVIEWER
    interviewer_results = retriever.retrieve(
        query="GIL Python", caller_role="interviewer", top_n=5
    )

    # Không có bất kỳ tài liệu mật nào
    for doc in interviewer_results:
        assert doc.metadata.get("is_confidential") is not True
        assert doc.metadata.get("source_type") != "rubric"
        # Bị bóc tách expected_signals
        assert "expected_signals_json" not in doc.metadata
        assert "expected_signals" not in doc.metadata

    # 2. Truy vấn với vai trò GRADER
    grader_results = retriever.retrieve(
        query="GIL Python", caller_role="grader", top_n=5
    )
    rubric_found = any(
        doc.metadata.get("source_type") == "rubric" for doc in grader_results
    )
    assert rubric_found is True


# ==============================================================================
# Test 4: Reciprocal Rank Fusion (RRF)
# ==============================================================================


def test_reciprocal_rank_fusion_consensus() -> None:
    """Tài liệu được cả Dense và BM25 đồng thuận đưa lên top phải đạt điểm RRF cao nhất."""
    dense_ranking = ["doc_consensus", "doc_dense_only", "doc_other"]
    bm25_ranking = ["doc_consensus", "doc_bm25_only", "doc_dense_only"]

    fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking], k=60, top_n=5)

    assert len(fused) > 0
    # doc_consensus đứng thứ 1 ở cả 2 list -> điểm: 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.03278
    top_doc_id, top_score = fused[0]
    assert top_doc_id == "doc_consensus"

    expected_score = (1.0 / 61) + (1.0 / 61)
    assert pytest.approx(top_score, rel=1e-4) == expected_score


# ==============================================================================
# Test 5: verify_citations (Grounding & Hallucination Detection)
# ==============================================================================


def test_verify_citations_grounding_and_hallucinations() -> None:
    """Xác thực 100% mã trích dẫn [chunk_id] và phát hiện mã bịa đặt."""
    valid_ids = {"8a921a205f8c:0", "0cb6031ac460:1", "2d9be11c1d23:2"}

    # 1. Trích dẫn hợp lệ 100%
    valid_answer = (
        "Phương thức PUT yêu cầu thay thế toàn bộ resource [8a921a205f8c:0], "
        "trong khi API thanh toán bắt buộc dùng Idempotency Key [0cb6031ac460:1]."
    )
    res_valid = verify_citations(valid_answer, valid_ids)
    assert res_valid["citation_existence_rate"] == 1.0
    assert res_valid["is_grounded"] is True
    assert res_valid["invalid_citations"] == []

    # 2. Trích dẫn bịa đặt (hallucinated chunk_id)
    fake_answer = (
        "Ứng viên được đánh giá mức 5 [fake12345678:99] căn cứ theo [8a921a205f8c:0]."
    )
    res_fake = verify_citations(fake_answer, valid_ids)
    assert res_fake["citation_existence_rate"] == 0.5
    assert res_fake["is_grounded"] is False
    assert "fake12345678:99" in res_fake["invalid_citations"]

    # 3. Kích hoạt Escape Hatch
    res_escape = verify_citations(ESCAPE_HATCH_TEXT, valid_ids)
    assert res_escape["is_escape_hatch"] is True
    assert res_escape["is_grounded"] is True
    assert res_escape["citation_existence_rate"] == 1.0


# ==============================================================================
# Test 6: Agent Tools (Interviewer vs Grader)
# ==============================================================================


def test_agent_tools_execution() -> None:
    """Kiểm tra 2 tool @tool chuẩn LangChain hoạt động an toàn và chính xác."""
    # 1. Tool tìm kiếm câu hỏi cho Interviewer
    q_output = search_interview_questions_tool.invoke(
        {"query": "API RESTful PUT PATCH", "top_k": 2}
    )
    assert isinstance(q_output, str)
    assert "Tìm thấy" in q_output
    # Đảm bảo không để lộ rubric hay expected_signals
    assert "expected_signals" not in q_output
    assert "Rubric mật" not in q_output

    # 2. Tool truy xuất Rubric cho Grader
    rubric_output = retrieve_grading_rubric_tool.invoke(
        {"competency_id": "BE_API_DESIGN"}
    )
    assert isinstance(rubric_output, str)
    assert "RUBRIC CHẤM ĐIỂM" in rubric_output
    assert "Thang điểm" in rubric_output
    assert "Điểm 4/5" in rubric_output or "mức 4" in rubric_output.lower()
