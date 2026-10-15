"""Grounding Rules, Citation Verifier, and Escape Hatch Mechanisms.

Implements:
- 5 Grounding Prompt Rules to prevent LLM hallucinations
- Standard Escape Hatch: 'Không đủ căn cứ trong tài liệu.'
- Algorithmic citation verification: ensures 100% of [chunk_id] exist in the retrieval pool
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document

ESCAPE_HATCH_TEXT = "Không đủ căn cứ trong tài liệu."

GROUNDING_SYSTEM_RULES = """\
BẠN LÀ MỘT TRỢ LÝ THỰC THI 5 NGUYÊN TẮC BÁM NGUỒN (GROUNDING RULES) TUYỆT ĐỐI:
1. NGUYÊN TẮC PHẠM VI: Chỉ sử dụng thông tin được cung cấp rõ ràng trong các trích đoạn tài liệu (Context) dưới đây. Không tự ý suy diễn hoặc lấy kiến thức ngoài.
2. NGUYÊN TẮC TRÍCH DẪN: Mọi khẳng định hoặc thông tin nghiệp vụ bắt buộc phải trích dẫn mã chunk nguồn tương ứng theo cú pháp [chunk_id] ngay cuối câu hoặc mệnh đề (ví dụ: [3a9b1c7d2e4f:0]).
3. NGUYÊN TẮC TRUNG THỰC: Không được bịa đặt (hallucinate) bất kỳ mã trích dẫn nào không có trong Context.
4. NGUYÊN TẮC ESCAPE HATCH: Nếu trong Context không có đủ thông tin để trả lời câu hỏi, hoặc thông tin mơ hồ, BẮT BUỘC chỉ trả lời chính xác câu: "Không đủ căn cứ trong tài liệu." Không giải thích thêm.
5. NGUYÊN TẮC BẢO MẬT: Tuyệt đối không tiết lộ các nội dung đánh dấu mật (Rubric nội bộ, Hidden test cases) cho ứng viên hoặc người phỏng vấn.
"""

CITATION_PATTERN = re.compile(r"\[([a-zA-Z0-9_-]+:[a-zA-Z0-9_-]+)\]")


def format_grounded_context(documents: list[Document]) -> str:
    """Định dạng danh sách Document thành chuỗi Context có đánh dấu mã trích dẫn."""
    if not documents:
        return "Không có tài liệu nào trong ngữ cảnh."

    context_blocks = []
    for doc in documents:
        cid = doc.metadata.get("chunk_id", "unknown:0")
        src_type = doc.metadata.get("source_type", "document")
        item_id = doc.metadata.get("item_id", "")
        header = f"=== TRÍCH ĐOẠN [{cid}] | Loại: {src_type} | Mã: {item_id} ==="
        context_blocks.append(f"{header}\n{doc.page_content}")

    return "\n\n".join(context_blocks)


def is_escape_hatch(text: str) -> bool:
    """Kiểm tra xem câu trả lời có phải là Escape Hatch không."""
    clean = text.strip().lower()
    return (
        clean == ESCAPE_HATCH_TEXT.lower() or "không đủ căn cứ trong tài liệu" in clean
    )


def verify_citations(answer_text: str, valid_chunk_ids: set[str]) -> dict[str, Any]:
    """Kiểm tra tính có thật (Existence Rate) của mọi mã trích dẫn trong câu trả lời.

    Trả về:
    - total_citations: Tổng số lần trích dẫn
    - valid_citations: Số trích dẫn hợp lệ (có trong valid_chunk_ids)
    - invalid_citations: Danh sách mã trích dẫn bịa đặt (hallucinated)
    - citation_existence_rate: Tỉ lệ mã hợp lệ (1.0 = 100% hợp lệ)
    - is_grounded: True nếu không có mã bịa và có căn cứ hợp lệ
    - is_escape_hatch: True nếu kích hoạt escape hatch hợp lệ
    """
    if is_escape_hatch(answer_text):
        return {
            "total_citations": 0,
            "valid_citations": 0,
            "invalid_citations": [],
            "citation_existence_rate": 1.0,
            "is_grounded": True,
            "is_escape_hatch": True,
        }

    raw_matches = CITATION_PATTERN.findall(answer_text)
    total = len(raw_matches)

    if total == 0:
        return {
            "total_citations": 0,
            "valid_citations": 0,
            "invalid_citations": [],
            "citation_existence_rate": 0.0,
            "is_grounded": False,
            "is_escape_hatch": False,
        }

    valid_count = 0
    invalid_ids: list[str] = []

    for cid in raw_matches:
        if cid in valid_chunk_ids:
            valid_count += 1
        else:
            invalid_ids.append(cid)

    rate = valid_count / total if total > 0 else 0.0
    grounded = total > 0 and len(invalid_ids) == 0

    return {
        "total_citations": total,
        "valid_citations": valid_count,
        "invalid_citations": invalid_ids,
        "citation_existence_rate": round(rate, 4),
        "is_grounded": grounded,
        "is_escape_hatch": False,
    }
