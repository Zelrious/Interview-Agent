"""Document Ingestion & Chunking Pipeline for Interview Agent Knowledge Base.

Implements:
- Deterministic hash-based embeddings (offline fallback) & cloud embeddings
- Factory `get_embedding_model()` với auto-detect Gemini / OpenAI / DeterministicHash
- Embedding lock verification
- Structure-aware chunking (questions, competencies, rubrics, coding tasks, policies)
- Coordinate metadata tagging (doc_id, chunk_id, is_confidential, etc.)
- Idempotent upserting
- Ingestion statistics via `get_ingestion_stats()`
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

# Tiền tố mẫu cho API key chưa được cấu hình thực
_PLACEHOLDER_PREFIXES = ("your_", "YOUR_", "sk-xxx", "<your", "<YOUR")


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


def _is_valid_api_key(value: str | None) -> bool:
    """Kiểm tra giá trị API key có thực sự được cấu hình (không phải placeholder)."""
    if not value:
        return False
    stripped = value.strip()
    if not stripped:
        return False
    # Loại bỏ các giá trị mẫu / placeholder phổ biến
    lower = stripped.lower()
    for prefix in _PLACEHOLDER_PREFIXES:
        if lower.startswith(prefix.lower()):
            return False
    return True


def get_embedding_model(
    preferred_provider: str = "auto",
    offline_mode: bool = False,
) -> Embeddings:
    """Factory khởi tạo Embedding model với fallback tự động.

    Ưu tiên theo thứ tự:
    1. Gemini (nếu GEMINI_API_KEY hoặc GOOGLE_API_KEY hợp lệ và preferred_provider != 'openai')
    2. OpenAI (nếu OPENAI_API_KEY hợp lệ)
    3. DeterministicHashEmbeddings (fallback offline — không crash, ghi log cảnh báo)

    Args:
        preferred_provider: 'auto' | 'gemini' | 'openai' | 'local'
        offline_mode: Khi True, bỏ qua kiểm tra API key và dùng local ngay lập tức.

    Returns:
        Embeddings instance với thuộc tính model_name được gán rõ ràng.
    """
    if offline_mode or preferred_provider == "local":
        logger.info(
            "[EmbeddingFactory] offline_mode=True hoặc preferred_provider='local'. "
            "Sử dụng DeterministicHashEmbeddings (deterministic-hash-v1)."
        )
        model = DeterministicHashEmbeddings(
            model_name="deterministic-hash-v1", dimension=384
        )
        return model

    # --- Thử Gemini ---
    if preferred_provider in ("auto", "gemini"):
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if _is_valid_api_key(gemini_key):
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings

                gemini_model_name = os.getenv(
                    "EMBEDDING_MODEL_NAME", "models/gemini-embedding-001"
                )
                embed = GoogleGenerativeAIEmbeddings(
                    model=gemini_model_name,
                    google_api_key=gemini_key,
                )
                object.__setattr__(
                    embed, "model_name", "gemini-embedding"
                )  # Pydantic v2 compat
                logger.info(
                    "[EmbeddingFactory] Sử dụng Gemini embedding: %s", gemini_model_name
                )
                return embed
            except ImportError as e:
                logger.warning(
                    "[EmbeddingFactory] Không thể khởi tạo Gemini embedding (ImportError): %s. "
                    "Thử OpenAI hoặc fallback local.",
                    e,
                )
            except RuntimeError as e:
                logger.warning(
                    "[EmbeddingFactory] Không thể khởi tạo Gemini embedding (RuntimeError): %s. "
                    "Thử OpenAI hoặc fallback local.",
                    e,
                )
            except OSError as e:
                logger.warning(
                    "[EmbeddingFactory] Không thể khởi tạo Gemini embedding (OSError): %s. "
                    "Thử OpenAI hoặc fallback local.",
                    e,
                )

    # --- Thử OpenAI ---
    if preferred_provider in ("auto", "openai"):
        openai_key = os.getenv("OPENAI_API_KEY")
        if _is_valid_api_key(openai_key):
            try:
                from langchain_openai import OpenAIEmbeddings

                embed = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    openai_api_key=openai_key,
                )
                embed.model_name = "text-embedding-3-small"  # type: ignore[attr-defined]
                logger.info(
                    "[EmbeddingFactory] Sử dụng OpenAI embedding: text-embedding-3-small"
                )
                return embed
            except ImportError as e:
                logger.warning(
                    "[EmbeddingFactory] Không thể khởi tạo OpenAI embedding (ImportError): %s. "
                    "Fallback về DeterministicHashEmbeddings.",
                    e,
                )
            except RuntimeError as e:
                logger.warning(
                    "[EmbeddingFactory] Không thể khởi tạo OpenAI embedding (RuntimeError): %s. "
                    "Fallback về DeterministicHashEmbeddings.",
                    e,
                )
            except OSError as e:
                logger.warning(
                    "[EmbeddingFactory] Không thể khởi tạo OpenAI embedding (OSError): %s. "
                    "Fallback về DeterministicHashEmbeddings.",
                    e,
                )

    # --- Fallback mặc định: DeterministicHashEmbeddings ---
    logger.warning(
        "[EmbeddingFactory] KHÔNG tìm thấy API key hợp lệ cho Gemini hoặc OpenAI. "
        "Tự động fallback về DeterministicHashEmbeddings (model_name='deterministic-hash-v1'). "
        "Hệ thống vẫn hoạt động bình thường ở chế độ offline/test."
    )
    model = DeterministicHashEmbeddings(
        model_name="deterministic-hash-v1", dimension=384
    )
    return model


def get_embeddings(model_name: str | None = None) -> Embeddings:
    """Khởi tạo Embedding model dựa trên cấu hình môi trường (legacy wrapper).

    Prefer sử dụng `get_embedding_model()` cho code mới.
    """
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
    """Xác nhận mô hình embedding dùng lúc truy vấn khớp 100% với lúc build index.

    Bắt buộc gọi hàm này trước mọi thao tác query để đảm bảo tính nhất quán
    của vector space. Nếu model không khớp, ném AssertionError ngay lập tức.

    Args:
        store_meta: Dict metadata của store, phải chứa key 'embedding_model'.
        expected_model: Tên model embedding đang dùng để truy vấn.

    Raises:
        AssertionError: Khi model lúc ingest khác model lúc query.
    """
    actual_model = store_meta.get("embedding_model")
    assert actual_model == expected_model, (
        f"Embedding Lock Mismatch: Store được build bằng '{actual_model}', "
        f"nhưng query dùng '{expected_model}'! "
        "Hãy rebuild index với đúng model hoặc dùng lại model cũ."
    )


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
    """Cập nhật dữ liệu idempotent: xóa toàn bộ chunk cũ cùng doc_id trước khi thêm chunk mới.

    Đảm bảo chạy lại nhiều lần (idempotent) — tổng số chunk không bị nhân bản.
    chunk_id cho mỗi chunk được sinh tất định theo dạng `{doc_id}:{idx}`.

    Args:
        store: Vector store (Chroma-like), in-memory list, hoặc object có thuộc tính `.chunks`.
        doc_id: Định danh tài liệu nguồn (12-char hex từ _generate_doc_id).
        chunks: Danh sách Document chunks cần ghi vào store.
    """
    # Sinh chunk_id tất định cho mỗi chunk trước khi upsert
    for idx, chunk in enumerate(chunks):
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["chunk_id"] = f"{doc_id}:{idx}"

    # Nếu store là Chroma VectorStore (có delete và add_documents)
    if hasattr(store, "delete") and hasattr(store, "add_documents"):
        try:
            store.delete(where={"doc_id": doc_id})
        except (KeyError, ValueError, RuntimeError) as e:
            logger.debug("Lỗi khi xóa chunk cũ trong store: %s", e)
        if chunks:
            store.add_documents(chunks)
    # Nếu store là in-memory list (dùng trong mock/test)
    elif isinstance(store, list):
        store[:] = [c for c in store if c.metadata.get("doc_id") != doc_id]
        store.extend(chunks)
    # Nếu store là object có thuộc tính .chunks (custom mock)
    elif hasattr(store, "chunks") and isinstance(store.chunks, list):
        store.chunks = [c for c in store.chunks if c.metadata.get("doc_id") != doc_id]
        store.chunks.extend(chunks)
    else:
        logger.warning(
            "[upsert_file_idempotent] Không nhận dạng được kiểu store: %s. "
            "Bỏ qua upsert cho doc_id='%s'.",
            type(store).__name__,
            doc_id,
        )


def get_ingestion_stats(chunks: list[Document]) -> dict[str, Any]:
    """Thống kê số lượng chunk theo source_type và status để in log kiểm chứng.

    Args:
        chunks: Danh sách Document chunks đã được load và phân mảnh.

    Returns:
        Dict chứa:
        - total: tổng số chunk
        - by_source_type: dict đếm chunk theo source_type
        - by_status: dict đếm chunk theo status
        - confidential_count: số chunk is_confidential=True
    """
    stats: dict[str, Any] = {
        "total": len(chunks),
        "by_source_type": {},
        "by_status": {},
        "confidential_count": 0,
    }

    for chunk in chunks:
        meta = chunk.metadata

        # Đếm theo source_type
        src = meta.get("source_type", "unknown")
        stats["by_source_type"][src] = stats["by_source_type"].get(src, 0) + 1

        # Đếm theo status
        status = meta.get("status", "unknown")
        stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

        # Đếm chunk mật
        if meta.get("is_confidential", False):
            stats["confidential_count"] += 1

    logger.info(
        "[IngestionStats] Tổng chunk: %d | Theo loại: %s | Theo trạng thái: %s | Mật: %d",
        stats["total"],
        stats["by_source_type"],
        stats["by_status"],
        stats["confidential_count"],
    )
    return stats
