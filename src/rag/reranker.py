"""Cascade Reranker (Cross-Encoder & Lightweight Cross-Attention Scorer).

Implements:
- 2-phase retrieval cascade (Hybrid RRF Top-20 -> Reranker Top-5)
- Deterministic LightweightCrossAttentionScorer for millisecond offline testing
- sentence_transformers.CrossEncoder support when online model is available
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class LightweightCrossAttentionScorer:
    """Bộ chấm điểm chéo nhẹ (Lightweight Cross-Attention Scorer).

    Đánh giá mức độ liên quan ngữ nghĩa và định danh kỹ thuật giữa query và document:
    1. Khớp chính xác mã định danh (Q_BE_001, BE_DB_OPTIMIZATION, CODE_001, ...)
    2. Tỉ lệ trùng lặp token có ý nghĩa (Token Overlap)
    3. Trùng khớp cụm từ (N-gram phrase matching)
    4. Mức độ tương đồng thuộc tính metadata (vai trò, cấp độ, năng lực)
    """

    @staticmethod
    def score(query: str, doc: Document) -> float:
        q_lower = query.lower()
        d_lower = doc.page_content.lower()
        meta = doc.metadata

        score = 0.0

        # 1. Khớp chính xác mã định danh (Technical Identifier Match)
        query_identifiers = re.findall(
            r"\b(?:q_[a-z0-9_]+|be_[a-z0-9_]+|da_[a-z0-9_]+|qa_[a-z0-9_]+|code_[a-z0-9_]+|rubric_[a-z0-9_]+)\b",
            q_lower,
        )
        item_id = str(meta.get("item_id", "")).lower()
        competency = str(meta.get("competency", "")).lower()

        for ident in query_identifiers:
            if ident == item_id or ident in d_lower or ident == competency:
                score += 15.0

        # 2. Khớp cụm từ nguyên văn (Exact Phrase Match)
        # Bóc tách các cụm từ quan trọng trong query (từ 2 đến 4 từ)
        q_words = [
            w
            for w in re.findall(r"[A-Za-z0-9_]+|[\u00C0-\u1EF9]+", q_lower)
            if len(w) >= 2
        ]
        if len(q_words) >= 2:
            for n in [3, 2]:
                for i in range(len(q_words) - n + 1):
                    phrase = " ".join(q_words[i : i + n])
                    if phrase in d_lower:
                        score += 3.0 * n

        # 3. Độ phủ Token (Token Overlap)
        d_words = set(re.findall(r"[A-Za-z0-9_]+|[\u00C0-\u1EF9]+", d_lower))
        matching_tokens = [w for w in q_words if w in d_words]
        if q_words:
            overlap_ratio = len(matching_tokens) / len(q_words)
            score += overlap_ratio * 8.0

        # 4. Tương đồng Metadata (Role & Level)
        target_role = str(meta.get("role", "")).lower()
        if target_role and target_role in q_lower:
            score += 2.0

        target_level = str(meta.get("level", "")).lower()
        if target_level and target_level in q_lower:
            score += 2.0

        # 5. Điểm số RRF ban đầu (nếu có)
        rrf_prior = float(meta.get("rrf_score", 0.0))
        score += rrf_prior * 10.0

        return score


class CascadeReranker:
    """Bộ xếp hạng lại theo tầng (Cascade Reranker) lọc từ Top-20 xuống Top-5."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("RERANKER_MODEL_NAME", "lightweight")
        self.cross_encoder: Any = None

        if self.model_name != "lightweight":
            try:
                from sentence_transformers import CrossEncoder

                self.cross_encoder = CrossEncoder(self.model_name)
                logger.info("Đã tải CrossEncoder model: %s", self.model_name)
            except (ImportError, RuntimeError, ValueError, OSError) as e:
                logger.warning(
                    "Không thể tải CrossEncoder (%s). Chuyển sang Lightweight mode.", e
                )
                self.cross_encoder = None

    def rerank(
        self, query: str, documents: list[Document], top_n: int = 5
    ) -> list[Document]:
        """Sắp xếp lại danh sách tài liệu và chọn ra top_n tài liệu phù hợp nhất."""
        if not documents:
            return []

        scored_docs: list[tuple[Document, float]] = []

        # 1. Nếu có model CrossEncoder của sentence_transformers
        if self.cross_encoder is not None:
            pairs = [[query, doc.page_content] for doc in documents]
            try:
                scores = self.cross_encoder.predict(pairs)
                for doc, s in zip(documents, scores):
                    scored_docs.append((doc, float(s)))
            except (RuntimeError, ValueError) as e:
                logger.warning(
                    "Lỗi suy luận CrossEncoder: %s. Chuyển sang Lightweight scorer.", e
                )
                self.cross_encoder = None

        # 2. Sử dụng LightweightCrossAttentionScorer
        if not scored_docs:
            for doc in documents:
                score = LightweightCrossAttentionScorer.score(query, doc)
                scored_docs.append((doc, score))

        # Sắp xếp giảm dần theo điểm số rerank
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        reranked_results: list[Document] = []
        for doc, score in scored_docs[:top_n]:
            updated_meta = dict(doc.metadata)
            updated_meta["rerank_score"] = round(score, 4)
            reranked_results.append(
                Document(page_content=doc.page_content, metadata=updated_meta)
            )

        return reranked_results
