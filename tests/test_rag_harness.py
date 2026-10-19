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
    DeterministicHashEmbeddings,
    get_embedding_model,
    get_ingestion_stats,
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


# ==============================================================================
# Test 7: Security Boundary for Candidate Role & Hidden Tests
# ==============================================================================


def test_security_boundary_candidate_role() -> None:
    """Ứng viên (candidate) bị loại bỏ tài liệu mật và stripped expected_signals/hidden_tests."""
    doc_task = Document(
        page_content="Bài tập code rate limiter",
        metadata={
            "chunk_id": "c1:0",
            "item_id": "CODE_001",
            "source_type": "coding_task",
            "is_confidential": False,
            "hidden_tests_json": '[{"input": "hidden", "expected": "ok"}]',
        },
    )
    doc_rubric = Document(
        page_content="Rubric chấm điểm năng lực thuật toán",
        metadata={
            "chunk_id": "r1:0",
            "item_id": "RUBRIC_ALGO",
            "source_type": "rubric",
            "is_confidential": True,
        },
    )

    retriever = HybridKnowledgeRetriever(documents=[doc_task, doc_rubric])
    candidate_results = retriever.retrieve(
        query="rate limiter", caller_role="candidate", top_n=5
    )

    for doc in candidate_results:
        assert doc.metadata.get("is_confidential") is not True
        assert doc.metadata.get("source_type") != "rubric"
        assert "hidden_tests_json" not in doc.metadata
        assert "expected_signals_json" not in doc.metadata


# ==============================================================================
# Test 8: Upsert Idempotent with Object Store (delete & add_documents)
# ==============================================================================


def test_upsert_idempotent_object_store() -> None:
    """upsert_file_idempotent hỗ trợ store có phương thức delete và add_documents."""

    class MockVectorStore:
        def __init__(self) -> None:
            self.docs: list[Document] = []

        def delete(self, where: dict) -> None:
            doc_id = where.get("doc_id")
            self.docs = [d for d in self.docs if d.metadata.get("doc_id") != doc_id]

        def add_documents(self, documents: list[Document]) -> None:
            self.docs.extend(documents)

    store = MockVectorStore()
    d1 = Document(page_content="Doc 1 v1", metadata={"doc_id": "file_a"})
    upsert_file_idempotent(store, "file_a", [d1])
    assert len(store.docs) == 1

    d2 = Document(page_content="Doc 1 v2", metadata={"doc_id": "file_a"})
    upsert_file_idempotent(store, "file_a", [d2])
    assert len(store.docs) == 1
    assert store.docs[0].page_content == "Doc 1 v2"


# ==============================================================================
# Test 9: Cascade Reranker Ranking Logic
# ==============================================================================


def test_cascade_reranker_logic() -> None:
    """CascadeReranker ưu tiên document có mã định danh và ngữ cảnh khớp truy vấn."""
    from src.rag.reranker import CascadeReranker

    reranker = CascadeReranker(model_name="lightweight")
    docs = [
        Document(
            page_content="Khái niệm cơ bản về kiến trúc phần mềm",
            metadata={"chunk_id": "c1", "item_id": "GENERAL_01"},
        ),
        Document(
            page_content="Tối ưu hóa cơ sở dữ liệu và đánh chỉ mục B-Tree BE_DB_OPTIMIZATION",
            metadata={
                "chunk_id": "c2",
                "item_id": "BE_DB_OPTIMIZATION",
                "competency": "BE_DB_OPTIMIZATION",
            },
        ),
    ]

    reranked = reranker.rerank("Năng lực BE_DB_OPTIMIZATION cấp middle", docs, top_n=2)
    assert len(reranked) == 2
    assert reranked[0].metadata["item_id"] == "BE_DB_OPTIMIZATION"


# ==============================================================================
# Test 10: questions_sv1_be_core.json — Schema Validation (SV1 Task 1.1)
# ==============================================================================


def test_sv1_questions_schema_and_count() -> None:
    """Kiểm tra questions_sv1_be_core.json có đúng >= 30 câu hỏi hợp lệ, đủ trường bắt buộc."""
    import json
    from pathlib import Path

    q_path = Path("data/knowledge/questions/questions_sv1_be_core.json")
    assert q_path.exists(), f"File không tồn tại: {q_path}"

    with open(q_path, encoding="utf-8") as f:
        questions = json.load(f)

    # Kiểm tra tổng số câu hỏi
    assert len(questions) >= 30, f"Cần >= 30 câu hỏi, hiện có {len(questions)}"

    # Các trường bắt buộc theo schema
    required_fields = {
        "question_id",
        "competency",
        "role",
        "level",
        "type",
        "difficulty",
        "language",
        "text",
        "follow_up_hints",
        "expected_signals",
        "status",
    }
    valid_levels = {"intern", "junior", "middle", "senior"}
    valid_statuses = {"active", "superseded", "draft"}
    valid_languages = {"vi", "en"}

    for q in questions:
        qid = q.get("question_id", "UNKNOWN")

        # Kiểm tra đủ trường bắt buộc
        missing = required_fields - q.keys()
        assert not missing, f"{qid}: thiếu trường {missing}"

        # Kiểm tra giá trị hợp lệ
        assert q["level"] in valid_levels, f"{qid}: level '{q['level']}' không hợp lệ"
        assert (
            q["status"] in valid_statuses
        ), f"{qid}: status '{q['status']}' không hợp lệ"
        assert (
            q["language"] in valid_languages
        ), f"{qid}: language '{q['language']}' không hợp lệ"
        assert (
            isinstance(q["difficulty"], int) and 1 <= q["difficulty"] <= 5
        ), f"{qid}: difficulty phải là int 1-5, nhận {q['difficulty']}"
        assert (
            isinstance(q["follow_up_hints"], list) and len(q["follow_up_hints"]) >= 2
        ), f"{qid}: follow_up_hints phải có >= 2 gợi ý"
        assert (
            isinstance(q["expected_signals"], list) and len(q["expected_signals"]) >= 3
        ), f"{qid}: expected_signals phải có >= 3 tín hiệu"

    # Phải có ít nhất 2 câu superseded (phục vụ test lọc metadata phiên bản)
    superseded = [q for q in questions if q["status"] == "superseded"]
    assert (
        len(superseded) >= 2
    ), f"Cần >= 2 câu 'superseded' để test metadata filtering, hiện có {len(superseded)}"

    # Phải bao phủ cả 2 ngôn ngữ
    langs = {q["language"] for q in questions}
    assert "vi" in langs, "Phải có câu hỏi tiếng Việt (vi)"
    assert "en" in langs, "Phải có câu hỏi tiếng Anh (en)"

    # Phải bao phủ cả 4 cấp độ
    levels = {q["level"] for q in questions}
    assert (
        levels == valid_levels
    ), f"Phải bao phủ đủ 4 cấp độ {valid_levels}, hiện có {levels}"


# ==============================================================================
# Test 11: get_embedding_model() fallback khi không có API key hợp lệ
# ==============================================================================


def test_get_embedding_model_fallback_without_api_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_embedding_model() fallback an toàn về DeterministicHashEmbeddings khi không có API key."""
    # Đảm bảo KHÔNG có API key hợp lệ trong môi trường
    monkeypatch.setenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

    embed_model = get_embedding_model(preferred_provider="auto")

    # Phải fallback về DeterministicHashEmbeddings, KHÔNG crash
    assert isinstance(
        embed_model, DeterministicHashEmbeddings
    ), f"Mong đợi DeterministicHashEmbeddings, nhận {type(embed_model).__name__}"
    assert embed_model.model_name == "deterministic-hash-v1"

    # Thử embed thực tế để đảm bảo không crash
    vectors = embed_model.embed_documents(["test fallback", "another text"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 384

    # Kiểm tra offline_mode=True cũng cho kết quả tương tự
    embed_offline = get_embedding_model(offline_mode=True)
    assert isinstance(embed_offline, DeterministicHashEmbeddings)
    assert embed_offline.model_name == "deterministic-hash-v1"


# ==============================================================================
# Test 12: upsert_file_idempotent — double run không nhân bản chunk
# ==============================================================================


def test_upsert_idempotent_double_run_fresh_store() -> None:
    """Chạy upsert_file_idempotent 2 lần liên tiếp trên store mới không nhân bản chunk."""
    from src.rag.ingest import _generate_doc_id

    fresh_store: list[Document] = []
    source_key = "question:Q_BE_TEST_IDEMPOTENT"
    doc_id = _generate_doc_id(source_key)

    def make_chunks(version: str) -> list[Document]:
        return [
            Document(
                page_content=f"Chunk 0 - {version}",
                metadata={"doc_id": doc_id, "source_type": "question"},
            ),
            Document(
                page_content=f"Chunk 1 - {version}",
                metadata={"doc_id": doc_id, "source_type": "question"},
            ),
            Document(
                page_content=f"Chunk 2 - {version}",
                metadata={"doc_id": doc_id, "source_type": "question"},
            ),
        ]

    # Lần 1: nạp 3 chunk
    upsert_file_idempotent(fresh_store, doc_id, make_chunks("v1"))
    assert (
        len(fresh_store) == 3
    ), f"Sau lần 1: mong đợi 3 chunk, nhận {len(fresh_store)}"

    # Kiểm tra chunk_id được sinh tất định: {doc_id}:0, {doc_id}:1, {doc_id}:2
    chunk_ids = [c.metadata["chunk_id"] for c in fresh_store]
    assert chunk_ids == [f"{doc_id}:0", f"{doc_id}:1", f"{doc_id}:2"]

    # Lần 2: nạp lại với dữ liệu cập nhật
    upsert_file_idempotent(fresh_store, doc_id, make_chunks("v2"))
    assert (
        len(fresh_store) == 3
    ), f"Sau lần 2: tổng chunk KHÔNG được tăng gấp đôi, mong đợi 3, nhận {len(fresh_store)}"

    # Nội dung phải là version mới nhất
    assert all(
        "v2" in c.page_content for c in fresh_store
    ), "Nội dung phải được cập nhật sang v2"

    # Kiểm tra get_ingestion_stats() hoạt động đúng
    stats = get_ingestion_stats(fresh_store)
    assert stats["total"] == 3
    assert stats["by_source_type"].get("question") == 3
    assert stats["confidential_count"] == 0
