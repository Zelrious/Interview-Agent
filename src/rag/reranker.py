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
import unicodedata
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_ID_PATTERN = re.compile(
    r"\b(?:q_[a-z0-9_]+|be_[a-z0-9_]+|da_[a-z0-9_]+|qa_[a-z0-9_]+"
    r"|code_[a-z0-9_]+|rubric_[a-z0-9_]+|behavioral_[a-z0-9_]+)\b"
)
# \w hiểu Unicode nên giữ nguyên từ tiếng Việt có dấu (không tách vụn)
_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = frozenset(
    {
        "và", "là", "của", "có", "cho", "với", "một", "các", "những", "được", "trong",
        "the", "and", "for", "with", "that", "this", "are", "how", "what",
    }
)
_SUPERSEDED_PENALTY = 5.0


def _norm(text: str) -> str:
    """Chuẩn hóa NFC + lowercase để so khớp tiếng Việt ổn định."""
    return unicodedata.normalize("NFC", text).lower()


def _tokens(text: str) -> list[str]:
    return [
        w for w in _TOKEN_PATTERN.findall(text) if len(w) >= 2 and w not in _STOPWORDS
    ]


class LightweightCrossAttentionScorer:
    """Bộ chấm điểm chéo nhẹ (Lightweight Cross-Attention Scorer).

    Đánh giá mức độ liên quan giữa query và document:
    1. Khớp chính xác mã định danh (theo token nguyên vẹn, không khớp substring)
    2. Trùng khớp cụm từ (N-gram phrase matching)
    3. Tỉ lệ trùng lặp token có ý nghĩa (Token Overlap)
    4. Tương đồng metadata (vai trò, cấp độ)
    5. Phạt tài liệu superseded (trừ khi query gọi đích danh ID)
    6. Điểm RRF ban đầu
    """

    @staticmethod
    def score(query: str, doc: Document) -> float:
        q_lower = _norm(query)
        d_lower = _norm(doc.page_content)
        meta = doc.metadata

        score = 0.0

        # 1. Khớp chính xác mã định danh (dedupe + so theo token nguyên vẹn)
        d_all_tokens = set(_TOKEN_PATTERN.findall(d_lower))
        item_id = str(meta.get("item_id", "")).lower()
        competency = str(meta.get("competency", "")).lower()
        explicit_id_hit = False

        for ident in set(_ID_PATTERN.findall(q_lower)):
            if ident in (item_id, competency) or ident in d_all_tokens:
                score += 15.0
                explicit_id_hit = True

        # 2. Khớp cụm từ nguyên văn (từ 2 đến 3 từ)
        q_words = _tokens(q_lower)
        if len(q_words) >= 2:
            for n in (3, 2):
                for i in range(len(q_words) - n + 1):
                    if " ".join(q_words[i : i + n]) in d_lower:
                        score += 3.0 * n

        # 3. Độ phủ token
        d_words = set(_tokens(d_lower))
        if q_words:
            matching = sum(1 for w in q_words if w in d_words)
            score += (matching / len(q_words)) * 8.0

        # 4. Tương đồng metadata (role, level) với word boundary
        q_plain = q_lower.replace("_", " ")
        target_role = str(meta.get("role", "")).lower().replace("_", " ")
        if target_role and target_role in q_plain:
            score += 2.0

        target_level = str(meta.get("level", "")).lower()
        if target_level and re.search(rf"\b{re.escape(target_level)}\b", q_plain):
            score += 2.0

        # 5. Phạt câu superseded, trừ khi query gọi đích danh mã của nó
        if str(meta.get("status", "")).lower() == "superseded" and not explicit_id_hit:
            score -= _SUPERSEDED_PENALTY

        # 6. Điểm RRF ban đầu (an toàn với None)
        score += float(meta.get("rrf_score") or 0.0) * 10.0

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

        # Sắp xếp giảm dần (sort ổn định: điểm bằng nhau giữ thứ tự của retriever)
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        reranked_results: list[Document] = []
        for doc, score in scored_docs[:top_n]:
            updated_meta = dict(doc.metadata)
            updated_meta["rerank_score"] = round(score, 4)
            reranked_results.append(
                Document(page_content=doc.page_content, metadata=updated_meta)
            )

        return reranked_results