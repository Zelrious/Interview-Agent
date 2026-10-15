"""Hybrid Knowledge Retriever (Dense Vector + Sparse BM25 + Reciprocal Rank Fusion).

Implements:
- Reciprocal Rank Fusion (RRF) with configurable k (default 60)
- BM25 tokenizer tailored for technical identifiers (e.g. Q_BE_001, BE_DB_OPTIMIZATION)
- Dense Vector semantic similarity search
- Role-based security boundary (caller_role: interviewer, grader, candidate)
- Metadata filtering (competency, level, language, source_type, status)
"""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from src.rag.ingest import get_embeddings


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize giữ nguyên các mã định danh kỹ thuật (Q_BE_001, BE_DB_OPTIMIZATION, v.v.)."""
    if not text:
        return []
    # Khớp các từ tiếng Việt/tiếng Anh, giữ nguyên dấu gạch dưới và số
    return re.findall(r"[A-Za-z0-9_]+|[\u00C0-\u1EF9]+", text.lower())


def reciprocal_rank_fusion(
    rank_lists: list[list[str]], k: int = 60, top_n: int = 20
) -> list[tuple[str, float]]:
    """Tính điểm Reciprocal Rank Fusion (RRF) từ nhiều danh sách xếp hạng.

    Công thức: RRF(d) = sum(1 / (k + rank_m(d))) với rank_m(d) là thứ hạng 1-based.
    """
    scores: dict[str, float] = {}

    for rank_list in rank_lists:
        for rank_0, item_id in enumerate(rank_list):
            rank_1 = rank_0 + 1
            scores[item_id] = scores.get(item_id, 0.0) + (1.0 / (k + rank_1))

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:top_n]


class HybridKnowledgeRetriever:
    """Retriever lai ghép Dense Vector và Sparse BM25 với chốt chặn phân quyền bảo mật."""

    def __init__(
        self,
        documents: list[Document] | None = None,
        embeddings: Embeddings | None = None,
        vectorstore: Any | None = None,
    ) -> None:
        self.embeddings = embeddings or get_embeddings()
        self.vectorstore = vectorstore
        self.documents: list[Document] = []
        self.doc_map: dict[str, Document] = {}
        self.bm25: BM25Okapi | None = None
        self.bm25_ids: list[str] = []

        # In-memory dense index cache
        self.dense_vectors: list[list[float]] = []
        self.dense_ids: list[str] = []

        if documents:
            self.build_index(documents)

    def build_index(self, documents: list[Document]) -> None:
        """Xây dựng chỉ mục tìm kiếm Dense và Sparse từ danh sách Document."""
        self.documents = list(documents)
        self.doc_map = {
            doc.metadata.get("chunk_id", str(i)): doc for i, doc in enumerate(documents)
        }

        # 1. Chỉ mục Sparse BM25
        tokenized_corpus = []
        self.bm25_ids = []
        for i, doc in enumerate(self.documents):
            cid = doc.metadata.get("chunk_id", str(i))
            self.bm25_ids.append(cid)
            # Ghép cả page_content và item_id vào để tăng độ nhạy cho mã định danh
            combined_text = f"{doc.metadata.get('item_id', '')} {doc.page_content}"
            tokenized_corpus.append(tokenize_for_bm25(combined_text))

        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Chỉ mục Dense Vector
        self.dense_ids = [
            doc.metadata.get("chunk_id", str(i)) for i, doc in enumerate(self.documents)
        ]
        texts = [doc.page_content for doc in self.documents]

        if hasattr(self.vectorstore, "add_documents"):
            self.vectorstore.add_documents(self.documents)

        # Lưu dense vector in-memory để truy vấn cực nhanh và độc lập
        if texts:
            self.dense_vectors = self.embeddings.embed_documents(texts)
        else:
            self.dense_vectors = []

    def _cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 < 1e-12 or norm2 < 1e-12:
            return 0.0
        return dot / (norm1 * norm2)

    def search_dense(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Tìm kiếm tương đồng ngữ nghĩa bằng Dense Vector."""
        if not self.dense_vectors or not self.dense_ids:
            return []

        query_vec = self.embeddings.embed_query(query)
        scored: list[tuple[str, float]] = []

        for cid, doc_vec in zip(self.dense_ids, self.dense_vectors):
            sim = self._cosine_similarity(query_vec, doc_vec)
            scored.append((cid, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_bm25(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Tìm kiếm từ khóa chính xác bằng BM25."""
        if self.bm25 is None or not self.bm25_ids:
            return []

        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        scored = [
            (cid, float(score))
            for cid, score in zip(self.bm25_ids, scores)
            if score > 0.0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def retrieve(
        self,
        query: str,
        top_n: int = 5,
        caller_role: Literal["interviewer", "grader", "candidate"] = "interviewer",
        metadata_filter: dict[str, Any] | None = None,
        fusion_k: int = 60,
    ) -> list[Document]:
        """Truy vấn kết hợp Dense + BM25 qua RRF kèm chốt chặn bảo mật theo vai trò."""
        candidate_pool_size = max(top_n * 4, 30)

        # 1. Thu thập ứng viên từ Dense Vector
        dense_results = self.search_dense(query, top_k=candidate_pool_size)
        dense_ids = [cid for cid, _ in dense_results]

        # 2. Thu thập ứng viên từ Sparse BM25
        bm25_results = self.search_bm25(query, top_k=candidate_pool_size)
        bm25_ids = [cid for cid, _ in bm25_results]

        # 3. Hợp nhất thứ hạng bằng RRF
        fused = reciprocal_rank_fusion(
            [dense_ids, bm25_ids], k=fusion_k, top_n=candidate_pool_size
        )

        # 4. Lọc kết quả và áp dụng chốt chặn bảo mật
        filtered_docs: list[Document] = []

        for cid, rrf_score in fused:
            doc = self.doc_map.get(cid)
            if doc is None:
                continue

            meta = dict(doc.metadata)
            meta["rrf_score"] = rrf_score

            # Kiểm tra bộ lọc metadata_filter (nếu có)
            if metadata_filter:
                match = True
                for k, expected_v in metadata_filter.items():
                    actual_v = meta.get(k)
                    if (
                        actual_v is not None
                        and actual_v != expected_v
                        and actual_v != "all"
                    ):
                        match = False
                        break
                if not match:
                    continue

            # CHỐT CHẶN BẢO MẬT:
            # Nếu người gọi là interviewer hoặc candidate:
            # - Tuyệt đối không trả về tài liệu mật (rubric hoặc hidden_tests)
            # - Xóa expected_signals khỏi metadata để tránh rò rỉ đáp án
            if caller_role in ("interviewer", "candidate"):
                if meta.get("is_confidential") is True:
                    continue
                # Bóc tách trường expected_signals ra khỏi metadata trả về
                meta.pop("expected_signals_json", None)
                meta.pop("expected_signals", None)
                meta.pop("hidden_tests_json", None)

            clean_doc = Document(page_content=doc.page_content, metadata=meta)
            filtered_docs.append(clean_doc)

            if len(filtered_docs) >= top_n:
                break

        return filtered_docs
