"""Computational Verification Sensors (Harness Lõi).

Implements deterministic algorithmic verification checks to enforce
groundedness, prevent hallucinations, and guarantee system constraints
without delegating checks back to LLM prose.
"""

from __future__ import annotations

import re
from typing import Any

from src.core.schemas import (
    CandidateLongTermMemory,
    CompetencyProfile,
    CompetencyScore,
    InterviewScenario,
    QuestionCoachingFeedback,
    TurnDecision,
)
from src.harness.pii_redactor import contains_unredacted_pii


def normalize_text(text: str) -> str:
    """Chuẩn hóa khoảng trắng, xuống dòng, và dấu câu cơ bản để so khớp chuỗi con."""
    if not text:
        return ""
    # Chuẩn hóa các dạng dấu ngoặc kép, nháy đơn, gạch ngang Unicode
    t = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("–", "-")
        .replace("—", "-")
        .replace("…", "...")
    )
    # Thu gọn mọi chuỗi khoảng trắng và xuống dòng thành một khoảng trắng đơn
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def is_exact_substring_quote(quote: str, source_text: str) -> bool:
    """Trả về True nếu quote (độ dài >= 3 ký tự) là chuỗi con có thật trong source_text.

    Áp dụng chuẩn hóa khoảng trắng trước khi kiểm tra nhằm loại bỏ sai lệch
    do xuống dòng hoặc khoảng cách định dạng của LLM.
    """
    if not quote or not source_text:
        return False

    norm_quote = normalize_text(quote)
    if len(norm_quote) < 3:
        return False

    norm_source = normalize_text(source_text)

    # Khớp chuỗi chính xác sau chuẩn hóa khoảng trắng
    if norm_quote in norm_source:
        return True

    # Khớp chuỗi trong trường hợp LLM vô tình bọc quote trong dấu nháy kép hoặc đơn
    stripped_quote = norm_quote.strip("\"'")
    return bool(len(stripped_quote) >= 3 and stripped_quote in norm_source)


def verify_cv_evidence_quotes(
    profile: CompetencyProfile, normalized_cv_text: str
) -> tuple[CompetencyProfile, list[str]]:
    """Kiểm tra mọi evidence_in_cv.

    Trích dẫn nào bịa (không phải chuỗi con có thật trong CV) thì tự động loại khỏi
    evidence_in_cv và thêm cảnh báo vào danh sách warnings.
    """
    valid_evidences = []
    warnings: list[str] = []
    norm_cv = normalize_text(normalized_cv_text)

    for ev in profile.evidence_in_cv:
        if is_exact_substring_quote(ev.quote, norm_cv):
            valid_evidences.append(ev)
        else:
            warnings.append(
                f"Phát hiện trích dẫn bịa / không tồn tại trong CV cho năng lực '{ev.competency_id}': '{ev.quote}'. Đã loại bỏ."
            )

    updated_profile = profile.model_copy(update={"evidence_in_cv": valid_evidences})
    return updated_profile, warnings


def verify_scenario_constraints(
    scenario: InterviewScenario,
    valid_question_ids: set[str],
    required_comp_ids: set[str],
) -> list[str]:
    """Kiểm tra tính hợp lệ và toàn vẹn của kịch bản phỏng vấn.

    - 100% câu hỏi có source nằm trong valid_question_ids, hoặc nếu
      source == 'generated' thì based_on phải nằm trong valid_question_ids.
    - Mọi năng lực trong required_comp_ids đều có ít nhất 1 câu hỏi.
    - Tổng time_budget_min của các section lệch không quá ±10% so với duration_min.

    Trả về danh sách lỗi (rỗng nếu kịch bản hợp lệ 100%).
    """
    errors: list[str] = []

    # 1. Kiểm tra nguồn gốc câu hỏi
    for sec_idx, section in enumerate(scenario.sections):
        for q_idx, q in enumerate(section.questions):
            loc_desc = f"Section {sec_idx + 1} ({section.competency_id}), Câu hỏi {q_idx + 1} ({q.question_id})"
            if q.source == "generated":
                if not q.based_on or q.based_on not in valid_question_ids:
                    errors.append(
                        f"{loc_desc}: Câu hỏi sinh tự động có based_on '{q.based_on}' không nằm trong ngân hàng câu hỏi hợp lệ"
                    )
            else:
                if q.source not in valid_question_ids:
                    errors.append(
                        f"{loc_desc}: Mã câu hỏi '{q.source}' không nằm trong ngân hàng câu hỏi hợp lệ"
                    )

    # 2. Kiểm tra độ phủ năng lực bắt buộc
    covered_comps: set[str] = {
        sec.competency_id for sec in scenario.sections if len(sec.questions) > 0
    }
    missing_comps = sorted(required_comp_ids - covered_comps)
    for comp_id in missing_comps:
        errors.append(
            f"Năng lực bắt buộc '{comp_id}' chưa có câu hỏi nào trong kịch bản phỏng vấn"
        )

    # 3. Kiểm tra tổng ngân sách thời gian không lệch quá ±10% duration_min
    total_budget = sum(sec.time_budget_min for sec in scenario.sections)
    min_allowed = 0.90 * scenario.duration_min
    max_allowed = 1.10 * scenario.duration_min
    if total_budget < min_allowed - 1e-6 or total_budget > max_allowed + 1e-6:
        errors.append(
            f"Tổng thời lượng các section ({total_budget:.1f} phút) lệch quá ±10% so với duration_min ({scenario.duration_min:.1f} phút, khoảng cho phép [{min_allowed:.1f}, {max_allowed:.1f}])"
        )

    return errors


def verify_turn_decision_quote(
    decision: TurnDecision, latest_candidate_answer: str
) -> bool:
    """Xác thực trích dẫn dẫn chứng khi Interviewer Agent đưa ra follow-up.

    Nếu action == 'follow_up', bắt buộc referenced_answer_span phải là chuỗi con có thật
    trong câu trả lời gần nhất của ứng viên.
    """
    if decision.action != "follow_up":
        return True

    if not decision.referenced_answer_span:
        return False

    return is_exact_substring_quote(
        decision.referenced_answer_span, latest_candidate_answer
    )


def verify_competency_score_quotes(
    score_obj: CompetencyScore, transcript_turns: dict[str, str]
) -> bool:
    """Xác thực 100% trích dẫn dẫn chứng chấm điểm của Grader Agent.

    Mọi evidence.quote phải là chuỗi con có thật trong transcript_turns[evidence.turn_id].
    """
    if not score_obj.evidence:
        # Nếu không có evidence, chỉ hợp lệ khi điểm là insufficient_evidence
        return score_obj.score == "insufficient_evidence"

    for ev in score_obj.evidence:
        turn_text = transcript_turns.get(ev.turn_id)
        if turn_text is None:
            return False
        if not is_exact_substring_quote(ev.quote, turn_text):
            return False

    return True


def verify_no_hidden_test_leak(utterance: str, hidden_test_strings: list[str]) -> bool:
    """Đảm bảo không rò rỉ bất kỳ chuỗi test ẩn nào vào tin nhắn gửi ứng viên."""
    norm_utterance = normalize_text(utterance).lower()
    for test_str in hidden_test_strings:
        norm_test = normalize_text(test_str).lower()
        if norm_test and norm_test in norm_utterance:
            return False
    return True


def verify_compaction_five_sections(summary_dict: dict[str, Any]) -> bool:
    """Kiểm tra sự hiện diện và tính hợp lệ của đủ 5 mục bắt buộc trong CompactionSummary.

    5 mục bắt buộc:
    1. session_goal (str, len >= 5)
    2. evaluated_competencies (list, len >= 1)
    3. current_open_question (str, len >= 2)
    4. constraints (dict chứa remaining_time_min, language, candidate_notes)
    5. next_step (str, len >= 3)
    """
    if not isinstance(summary_dict, dict):
        return False

    required_fields = [
        "session_goal",
        "evaluated_competencies",
        "current_open_question",
        "constraints",
        "next_step",
    ]
    for field in required_fields:
        if field not in summary_dict:
            return False

    session_goal = summary_dict.get("session_goal")
    if not isinstance(session_goal, str) or len(session_goal.strip()) < 5:
        return False

    evaluated_competencies = summary_dict.get("evaluated_competencies")
    if not isinstance(evaluated_competencies, list) or len(evaluated_competencies) == 0:
        return False

    current_open_question = summary_dict.get("current_open_question")
    if (
        not isinstance(current_open_question, str)
        or len(current_open_question.strip()) < 2
    ):
        return False

    next_step = summary_dict.get("next_step")
    if not isinstance(next_step, str) or len(next_step.strip()) < 3:
        return False

    constraints = summary_dict.get("constraints")
    if not isinstance(constraints, dict):
        return False

    required_constraint_keys = {"remaining_time_min", "language", "candidate_notes"}
    return required_constraint_keys.issubset(constraints.keys())


# ==============================================================================
# UC-09: Validators cho Hướng A (App luyện phỏng vấn & Bộ nhớ dài hạn)
# ==============================================================================


def verify_coaching_feedback(
    feedback: QuestionCoachingFeedback,
    candidate_answer: str,
    expected_signals: list[str],
) -> bool:
    """Xác thực phản hồi huấn luyện cho câu hỏi luyện tập (UC-09).

    - Kiểm tra mọi mục trong missing_signals đều nằm trong expected_signals của câu hỏi.
    - Kiểm tra strengths_with_quotes có trích dẫn chuỗi con thực sự từ candidate_answer
      (hoặc trực tiếp là chuỗi con, hoặc chứa đoạn trích dẫn trong ngoặc kép/đơn).
    """
    # 1. Kiểm tra missing_signals phải thuộc expected_signals
    norm_expected = {normalize_text(s).lower() for s in expected_signals}
    for sig in feedback.missing_signals:
        if normalize_text(sig).lower() not in norm_expected:
            return False

    # 2. Kiểm tra strengths_with_quotes phải trích xuất thật từ candidate_answer
    def _item_has_valid_quote(strength_item: str, source_text: str) -> bool:
        if is_exact_substring_quote(strength_item, source_text):
            return True
        # Tìm các chuỗi trích dẫn nằm trong dấu ngoặc nháy đơn hoặc kép: "...", '...', “...”, ‘...’
        quoted_spans = re.findall(r'["\'“‘]([^"\'”’]{3,})["\'”’]', strength_item)
        return bool(
            quoted_spans
            and any(
                is_exact_substring_quote(span, source_text) for span in quoted_spans
            )
        )

    for item in feedback.strengths_with_quotes:
        if not _item_has_valid_quote(item, candidate_answer):
            return False

    return True


def verify_memory_no_pii(memory: CandidateLongTermMemory) -> bool:
    """Đảm bảo bộ nhớ dài hạn của ứng viên tuyệt đối không chứa PII (UC-09).

    Quét toàn bộ văn bản trong memory (ngoại trừ candidate_code ẩn danh)
    bằng contains_unredacted_pii để bảo đảm 0 trường PII nào xuất hiện.
    """
    text_parts: list[str] = [
        memory.target_role,
        memory.target_level,
        *memory.weak_competencies,
        *memory.improvement_plan,
        memory.updated_at,
        memory.expires_at,
    ]
    for sess in memory.sessions:
        text_parts.append(sess.session_id)
        text_parts.append(sess.date_iso)
        text_parts.extend(sess.scores.keys())

    combined_text = " ".join(text_parts)
    unredacted = contains_unredacted_pii(combined_text)
    return len(unredacted) == 0


def verify_second_session_focus(
    scenario: InterviewScenario, weak_competencies: list[str]
) -> float:
    """Tính tỉ lệ câu hỏi trong scenario tập trung vào các năng lực còn yếu (UC-09).

    Tiêu chí chấp nhận UC-09 yêu cầu tỉ lệ này >= 0.60 ở phiên luyện thứ hai.
    Trả về tỉ lệ (float từ 0.0 đến 1.0).
    """
    total_questions = sum(len(sec.questions) for sec in scenario.sections)
    if total_questions == 0:
        return 0.0

    weak_set = set(weak_competencies)
    weak_questions = sum(
        len(sec.questions) for sec in scenario.sections if sec.competency_id in weak_set
    )
    return weak_questions / total_questions
