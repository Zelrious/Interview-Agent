"""PII Redactor and Scanner for Interview Agent.

Ensures candidate data protection and compliance with anti-bias policies
by masking 11 protected attributes before transmitting text to LLM agents.
Also provides deterministic validation to ensure memory and reports are free of raw PII.
"""

from __future__ import annotations

import re

# 11 loại thuộc tính được bảo vệ
PII_FIELD_NAMES = {
    "name": "[REDACTED_NAME]",
    "gender": "[REDACTED_GENDER]",
    "age_dob": "[REDACTED_AGE_DOB]",
    "hometown": "[REDACTED_HOMETOWN]",
    "ethnicity": "[REDACTED_ETHNICITY]",
    "religion": "[REDACTED_RELIGION]",
    "marital": "[REDACTED_MARITAL]",
    "appearance": "[REDACTED_APPEARANCE]",
    "phone": "[REDACTED_PHONE]",
    "email": "[REDACTED_EMAIL]",
    "address": "[REDACTED_ADDRESS]",
}

# Regex nhận diện email & số điện thoại
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(
    r"(?:(?:\+84|0)(?:\s*|\.?|-?)(?:3[2-9]|5[2689]|7[06-9]|8[1-9]|9\d)(?:\s*|\.?|-?)\d{3}(?:\s*|\.?|-?)\d{3,4}\b)"
    r"|(?:(?:tel|phone|sđt|điện thoại|mobile)\s*[:=]\s*(\+?\d[\d\s\-().]{7,15}\d)\b)",
    re.IGNORECASE,
)

# Regex các thuộc tính có nhãn (labeled attributes)
LABELED_PATTERNS: dict[str, list[re.Pattern]] = {
    "gender": [
        re.compile(
            r"(?i)((?:giới tính|gender|sex)\s*[:=]\s*)(nam|nữ|khác|male|female|other|non-binary)\b"
        ),
    ],
    "age_dob": [
        re.compile(
            r"(?i)((?:ngày sinh|sinh ngày|date of birth|dob)\s*[:=]\s*)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
        ),
        re.compile(r"(?i)((?:năm sinh|birth year|sinh năm)\s*[:=]\s*)(\d{4})"),
        re.compile(r"(?i)((?:tuổi|age)\s*[:=]\s*)(\d{1,2})\b"),
        re.compile(r"(?i)\b(\d{1,2})\s*(?:tuổi|years?\s*old)\b"),
    ],
    "hometown": [
        re.compile(
            r"(?i)((?:quê quán|nguyên quán|nơi sinh|hometown|place of birth)\s*[:=]\s*)([^\n,;\r]+)"
        ),
    ],
    "ethnicity": [
        re.compile(r"(?i)((?:dân tộc|ethnicity)\s*[:=]\s*)([^\n,;\r]+)"),
    ],
    "religion": [
        re.compile(r"(?i)((?:tôn giáo|religion)\s*[:=]\s*)([^\n,;\r]+)"),
    ],
    "marital": [
        re.compile(
            r"(?i)((?:tình trạng hôn nhân|hôn nhân|marital status)\s*[:=]\s*)([^\n,;\r]+)"
        ),
        re.compile(
            r"(?i)(?:^|[\s,;])(độc thân|đã kết hôn|ly hôn|ly thân|single|married|divorced)(?:$|[\s,;])"
        ),
    ],
    "appearance": [
        re.compile(
            r"(?i)((?:ngoại hình|chiều cao|cân nặng|chiều cao/cân nặng|height|weight|appearance)\s*[:=]\s*)([^\n,;\r]+)"
        ),
        re.compile(
            r"(?i)((?:ảnh đại diện|ảnh chân dung|avatar|photo|profile picture)\s*[:=]\s*)([^\n,;\r]+)"
        ),
        re.compile(r"(?i)\b(?:cao\s*)?(?:1m[5-9]\d|\d{3}\s*cm)\b"),
        re.compile(r"(?i)\b(?:nặng\s*)?\d{2,3}\s*kg\b"),
    ],
    "address": [
        re.compile(
            r"(?i)((?:địa chỉ|nơi ở|thường trú|tạm trú|address|residence)\s*[:=]\s*)([^\n\r]+)"
        ),
    ],
    "name": [
        re.compile(
            r"(?i)((?:họ và tên|họ tên|full name|tên ứng viên|ứng viên)\s*[:=]\s*)([A-ZÀ-Ỹa-zà-ỹ\s]{2,40})"
        ),
    ],
}


def redact_pii(
    text: str, extra_names: list[str] | None = None
) -> dict[str, str | list[str]]:
    """Che 11 loại thuộc tính được bảo vệ theo nguyên tắc bảo vệ dữ liệu & chống thiên lệch.

    Thay thế thông tin nhạy cảm bằng placeholder tương ứng:
    - [REDACTED_NAME], [REDACTED_GENDER], [REDACTED_AGE_DOB], [REDACTED_HOMETOWN],
      [REDACTED_ETHNICITY], [REDACTED_RELIGION], [REDACTED_MARITAL], [REDACTED_APPEARANCE],
      [REDACTED_PHONE], [REDACTED_EMAIL], [REDACTED_ADDRESS].

    Trả về:
    {"status": "ok", "redacted_text": ..., "redacted_fields": [...]}
    (Chỉ lưu danh sách loại trường đã che, tuyệt đối không lưu dữ liệu gốc).
    """
    if not text:
        return {"status": "ok", "redacted_text": "", "redacted_fields": []}

    redacted_fields: set[str] = set()
    result = text

    # 1. Che extra_names (nếu có)
    if extra_names:
        for name in extra_names:
            clean_name = name.strip()
            if len(clean_name) >= 2:
                pattern = re.compile(rf"(?i)\b{re.escape(clean_name)}\b")
                if pattern.search(result):
                    result = pattern.sub("[REDACTED_NAME]", result)
                    redacted_fields.add("name")

    # 2. Che Email
    if EMAIL_PATTERN.search(result):
        result = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", result)
        redacted_fields.add("email")

    # 3. Che Phone
    def replace_phone(match: re.Match) -> str:
        # Nếu match có group 1 (từ cú pháp phone: ...)
        if match.group(1):
            start = match.start(1) - match.start(0)
            prefix = match.group(0)[:start]
            return f"{prefix}[REDACTED_PHONE]"
        return "[REDACTED_PHONE]"

    if PHONE_PATTERN.search(result):
        result = PHONE_PATTERN.sub(replace_phone, result)
        redacted_fields.add("phone")

    # 4. Che các trường nhãn (labeled fields)
    for field_name, patterns in LABELED_PATTERNS.items():
        placeholder = PII_FIELD_NAMES[field_name]
        for pattern in patterns:

            def repl(m: re.Match, ph: str = placeholder, fn: str = field_name) -> str:
                redacted_fields.add(fn)
                # Nếu pattern có group 1 là prefix (ví dụ 'Giới tính: ')
                if m.lastindex and m.lastindex >= 2:
                    return f"{m.group(1)}{ph}"
                elif m.lastindex == 1:
                    # Nếu pattern chỉ có 1 group (ví dụ inline age hoặc phone)
                    return ph
                return ph

            if pattern.search(result):
                result = pattern.sub(repl, result)

    return {
        "status": "ok",
        "redacted_text": result,
        "redacted_fields": sorted(redacted_fields),
    }


def contains_unredacted_pii(text: str) -> list[str]:
    """Quét văn bản để phát hiện các thuộc tính PII chưa được che.

    Dùng làm cảm biến kiểm chứng (computational sensor) cho báo cáo UC-08
    và bộ nhớ UC-09. Bỏ qua các placeholder đã che dạng [REDACTED_*].

    Trả về danh sách các trường PII còn sót lại (rỗng nếu hoàn toàn sạch).
    """
    if not text:
        return []

    # Tạm thời loại bỏ các placeholder đã che hợp lệ để tránh false positive
    sanitized = text
    for placeholder in PII_FIELD_NAMES.values():
        sanitized = sanitized.replace(placeholder, " ")

    detected: set[str] = set()

    # Kiểm tra Email
    if EMAIL_PATTERN.search(sanitized):
        detected.add("email")

    # Kiểm tra Phone
    if PHONE_PATTERN.search(sanitized):
        detected.add("phone")

    # Kiểm tra các trường nhãn
    for field_name, patterns in LABELED_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(sanitized)
            if match:
                # Kiểm tra nếu match không chỉ chứa khoảng trắng
                matched_val = match.group(0).strip()
                if len(matched_val) > 0:
                    detected.add(field_name)
                    break

    return sorted(detected)
