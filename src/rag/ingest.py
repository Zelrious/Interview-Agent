"""Document Ingestion & Chunking Pipeline for Interview Agent Knowledge Base.

Implements:
- Deterministic hash-based embeddings (offline fallback) & cloud embeddings
- Embedding lock verification
- Structure-aware chunking (questions, competencies, rubrics, coding tasks, policies)
- Coordinate metadata tagging (doc_id, chunk_id, is_confidential, etc.)
- Idempotent upserting
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "local-deterministic-minilm")


class DeterministicHashEmbeddings(Embeddings):
    """Deterministic, zero-dependency embedding generator for offline testing & benchmark.

    Generates dense vectors (default 384 dimensions) using a combination of
    token hashing, character n-grams, and technical identifier salience,
    followed by L2 normalization.
    """

    def __init__(
        self, model_name: str = "local-deterministic-minilm", dimension: int = 384
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        if not text:
            return vector

        # Tokenize preserving identifiers like Q_BE_001, BE_DB_OPTIMIZATION
        tokens = re.findall(r"[A-Za-z0-9_]+|[\u00C0-\u1EF9]+", text.lower())

        # 1. Unigram & Bigram hashing
        for i, token in enumerate(tokens):
            # Higher weight for technical identifiers and keywords
            weight = 3.0 if ("_" in token or any(c.isdigit() for c in token)) else 1.0

            # Hash token to dimension
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            vector[idx] += sign * weight

            # Bigram
            if i > 0:
                bigram = f"{tokens[i - 1]}_{token}"
                hb = int(hashlib.sha256(bigram.encode("utf-8")).hexdigest(), 16)
                idx_b = hb % self.dimension
                sign_b = 1.0 if ((hb >> 8) & 1) == 0 else -1.0
                vector[idx_b] += sign_b * 1.5

        # 2. Character trigrams for morphological robustness
        clean_text = " ".join(tokens)
        for i in range(len(clean_text) - 2):
            trigram = clean_text[i : i + 3]
            ht = int(hashlib.md5(trigram.encode("utf-8")).hexdigest(), 16)
            idx_t = ht % self.dimension
            sign_t = 1.0 if ((ht >> 4) & 1) == 0 else -1.0
            vector[idx_t] += sign_t * 0.3

        # L2 normalization
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 1e-12:
            return [v / norm for v in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)


def get_embeddings(model_name: str | None = None) -> Embeddings:
    """Khởi tạo Embedding model dựa trên cấu hình môi trường."""
    selected = model_name or EMBEDDING_MODEL_NAME

    if selected.startswith("text-embedding") and os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=selected)
        except (ImportError, RuntimeError) as e:
            logger.debug("Không thể khởi tạo OpenAIEmbeddings: %s", e)

    if "models/embedding" in selected and os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            return GoogleGenerativeAIEmbeddings(model=selected)
        except (ImportError, RuntimeError) as e:
            logger.debug("Không thể khởi tạo GoogleGenerativeAIEmbeddings: %s", e)

    return DeterministicHashEmbeddings(model_name=selected, dimension=384)


def verify_embedding_lock(store_meta: dict[str, Any], expected_model: str) -> None:
    """Xác nhận mô hình embedding dùng lúc truy vấn khớp 100% với lúc build index."""
    actual_model = store_meta.get("embedding_model")
    assert (
        actual_model == expected_model
    ), f"Embedding Lock Mismatch: Store được build bằng '{actual_model}', nhưng query dùng '{expected_model}'!"


def _generate_doc_id(key: str) -> str:
    """Sinh mã định danh doc_id 12 ký tự hex từ key."""
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def load_and_chunk_knowledge_base(
    data_dir: str | Path = "data/knowledge",
    add_contextual_prefix: bool = True,
) -> list[Document]:
    """Tải toàn bộ kho tri thức và phân mảnh theo cấu trúc chuyên biệt."""
    base_path = Path(data_dir)
    chunks: list[Document] = []

    # 1. Câu hỏi phỏng vấn (questions/*.json)
    q_dir = base_path / "questions"
    if q_dir.exists():
        for q_file in sorted(q_dir.glob("*.json")):
            with open(q_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for idx, q in enumerate(data):
                qid = q["question_id"]
                comp = q.get("competency", "GENERAL")
                lvl = q.get("level", "junior")
                status = q.get("status", "active")
                hints = q.get("follow_up_hints", [])

                prefix = ""
                if add_contextual_prefix:
                    prefix = f"[Tài liệu: Ngân hàng câu hỏi | Mã: {qid} | Năng lực: {comp} | Cấp: {lvl} | Trạng thái: {status}] "

                content_parts = [
                    f"{prefix}{q['text']}",
                    "Gợi ý hỏi đào sâu:\n- " + "\n- ".join(hints) if hints else "",
                ]
                content = "\n\n".join(part for part in content_parts if part)

                doc_id = _generate_doc_id(f"question:{qid}")
                meta = {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}:{idx}",
                    "item_id": qid,
                    "source_type": "question",
                    "competency": comp,
                    "role": q.get("role", "general"),
                    "level": lvl,
                    "type": q.get("type", "technical"),
                    "language": q.get("language", "vi"),
                    "difficulty": q.get("difficulty", 3),
                    "status": status,
                    "version": q.get("version", "v2.0"),
                    "path": str(q_file.relative_to(base_path)).replace("\\", "/"),
                    "is_confidential": False,
                    "expected_signals_json": json.dumps(
                        q.get("expected_signals", []), ensure_ascii=False
                    ),
                }
                chunks.append(Document(page_content=content, metadata=meta))

    # 2. Khung năng lực (competencies/*.json)
    comp_dir = base_path / "competencies"
    if comp_dir.exists():
        for c_file in sorted(comp_dir.glob("*.json")):
            with open(c_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for idx, c in enumerate(data):
                cid = c["competency_id"]
                role = c.get("role", "general")
                prefix = ""
                if add_contextual_prefix:
                    prefix = (
                        f"[Tài liệu: Khung năng lực | Mã: {cid} | Vai trò: {role}] "
                    )

                levels_str = "\n".join(
                    f"- Cấp {k.upper()}: {v}" for k, v in c.get("levels", {}).items()
                )
                content = f"{prefix}Năng lực: {c['name']} ({cid})\nNguồn tham chiếu: {c.get('source_reference', '')}\n\nMô tả hành vi theo cấp độ:\n{levels_str}"

                doc_id = _generate_doc_id(f"competency:{cid}")
                meta = {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}:{idx}",
                    "item_id": cid,
                    "source_type": "competency",
                    "competency": cid,
                    "role": role,
                    "level": "all",
                    "status": "active",
                    "version": "v2.0",
                    "path": str(c_file.relative_to(base_path)).replace("\\", "/"),
                    "is_confidential": False,
                }
                chunks.append(Document(page_content=content, metadata=meta))

    # 3. Rubric chấm điểm mật (rubrics/*.json)
    r_dir = base_path / "rubrics"
    if r_dir.exists():
        for r_file in sorted(r_dir.glob("*.json")):
            with open(r_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for idx, r in enumerate(data):
                rid = r["rubric_id"]
                cid = r["competency_id"]
                prefix = ""
                if add_contextual_prefix:
                    prefix = (
                        f"[Tài liệu: Rubric mật | Mã rubric: {rid} | Năng lực: {cid}] "
                    )

                scale_str = "\n".join(
                    f"- Điểm {k}/5: {v}" for k, v in sorted(r.get("scale", {}).items())
                )
                samples = r.get("sample_answers", {})
                sample_str = f"Ví dụ câu trả lời mức 2:\n{samples.get('level_2_example', '')}\n\nVí dụ câu trả lời mức 4:\n{samples.get('level_4_example', '')}"

                content = f"{prefix}Rubric Version: {r.get('rubric_version', 'v2.0')}\nThang điểm đánh giá năng lực {cid}:\n{scale_str}\n\n{sample_str}"

                doc_id = _generate_doc_id(f"rubric:{rid}")
                meta = {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}:{idx}",
                    "item_id": rid,
                    "source_type": "rubric",
                    "competency": cid,
                    "role": "all",
                    "level": "all",
                    "status": "active",
                    "version": r.get("rubric_version", "v2.0"),
                    "path": str(r_file.relative_to(base_path)).replace("\\", "/"),
                    "is_confidential": True,  # Rubric là tài liệu mật
                }
                chunks.append(Document(page_content=content, metadata=meta))

    # 4. Bài tập Code (coding_tasks/*.json)
    code_dir = base_path / "coding_tasks"
    if code_dir.exists():
        for cp_file in sorted(code_dir.glob("*.json")):
            with open(cp_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for idx, p in enumerate(data):
                pid = p["problem_id"]
                comp = p.get("competency", "GENERAL")
                lvl = p.get("level", "middle")
                prefix = ""
                if add_contextual_prefix:
                    prefix = f"[Tài liệu: Bài tập code | Mã: {pid} | Năng lực: {comp} | Cấp: {lvl}] "

                tests_str = json.dumps(p.get("public_tests", []), ensure_ascii=False)
                content = f"{prefix}{p['title']} ({pid})\n\nĐề bài:\n{p['prompt']}\n\nPublic Tests:\n{tests_str}"

                doc_id = _generate_doc_id(f"coding_task:{pid}")
                meta = {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}:{idx}",
                    "item_id": pid,
                    "source_type": "coding_task",
                    "competency": comp,
                    "level": lvl,
                    "status": "active",
                    "version": "v2.0",
                    "path": str(cp_file.relative_to(base_path)).replace("\\", "/"),
                    "is_confidential": False,
                    "hidden_tests_json": json.dumps(
                        p.get("hidden_tests", []), ensure_ascii=False
                    ),
                }
                chunks.append(Document(page_content=content, metadata=meta))

    # 5. Chính sách doanh nghiệp (policies/*.md)
    pol_dir = base_path / "policies"
    if pol_dir.exists():
        for pol_file in sorted(pol_dir.glob("*.md")):
            text = pol_file.read_text(encoding="utf-8")
            # Cắt theo heading Markdown (#, ##, ###)
            sections = re.split(r"(?m)^(?=#{1,3}\s)", text)
            for idx, sec in enumerate(sections):
                clean_sec = sec.strip()
                if not clean_sec:
                    continue
                # Trích xuất dòng tiêu đề
                header_line = clean_sec.split("\n", 1)[0].strip()
                doc_id = _generate_doc_id(f"policy:{header_line}")
                meta = {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}:{idx}",
                    "item_id": f"POL_{idx}",
                    "source_type": "policy",
                    "competency": "policy",
                    "role": "all",
                    "level": "all",
                    "status": "active",
                    "version": "v2.0",
                    "heading_path": header_line,
                    "path": str(pol_file.relative_to(base_path)).replace("\\", "/"),
                    "is_confidential": False,
                }
                chunks.append(Document(page_content=clean_sec, metadata=meta))

    return chunks


def chunk_documents_fixed_size(
    documents: list[Document], chunk_size: int = 300, chunk_overlap: int = 50
) -> list[Document]:
    """Phân mảnh văn bản fixed-size 300 ký tự (dùng làm Baseline Cấu hình A)."""
    fixed_chunks: list[Document] = []
    for doc in documents:
        text = doc.page_content
        start = 0
        idx = 0
        while start < len(text):
            end = start + chunk_size
            sub = text[start:end]
            meta = dict(doc.metadata)
            meta["chunk_id"] = f"{meta.get('doc_id', 'doc')}:fixed_{idx}"
            fixed_chunks.append(Document(page_content=sub, metadata=meta))
            start += max(1, chunk_size - chunk_overlap)
            idx += 1
    return fixed_chunks


def upsert_file_idempotent(store: Any, doc_id: str, chunks: list[Document]) -> None:
    """Cập nhật dữ liệu idempotent: xóa toàn bộ chunk cũ cùng doc_id trước khi thêm chunk mới."""
    # Nếu store là Chroma VectorStore
    if hasattr(store, "delete") and hasattr(store, "add_documents"):
        try:
            store.delete(where={"doc_id": doc_id})
        except (KeyError, ValueError, RuntimeError) as e:
            logger.debug("Lỗi khi xóa chunk cũ trong store: %s", e)
        if chunks:
            store.add_documents(chunks)
    # Nếu store là in-memory list (ví dụ trong mock/test store)
    elif isinstance(store, list):
        store[:] = [c for c in store if c.metadata.get("doc_id") != doc_id]
        store.extend(chunks)
    elif hasattr(store, "chunks") and isinstance(store.chunks, list):
        store.chunks = [c for c in store.chunks if c.metadata.get("doc_id") != doc_id]
        store.chunks.extend(chunks)
