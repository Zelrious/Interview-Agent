"""Core Pydantic v2 schemas for Interview Agent (SE373 - Đề tài 05).

All models strictly enforce ConfigDict(extra="forbid") to guarantee
schema conformance and prevent hallucinated or unexpected fields.
"""

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ==============================================================================
# UC-01: CompetencyProfile Schemas
# ==============================================================================


class RequiredCompetency(BaseModel):
    """Năng lực bắt buộc được trích xuất từ Job Description (JD)."""

    model_config = ConfigDict(extra="forbid")

    competency_id: str
    weight: float = Field(ge=0.0, le=1.0)
    source_quote_jd: str


class CVEvidence(BaseModel):
    """Dẫn chứng năng lực trích xuất từ CV của ứng viên."""

    model_config = ConfigDict(extra="forbid")

    competency_id: str
    quote: str
    strength: Literal["weak", "moderate", "strong"]


class CompetencyProfile(BaseModel):
    """Hồ sơ năng lực tổng hợp đối sánh giữa JD và CV ứng viên."""

    model_config = ConfigDict(extra="forbid")

    role: str
    target_level: Literal["intern", "junior", "middle", "senior"]
    required_competencies: list[RequiredCompetency]
    evidence_in_cv: list[CVEvidence]
    gaps_to_probe: list[str]
    red_flags: list[str]
    redacted_fields: list[str]
    injection_detected: bool


# ==============================================================================
# UC-02: InterviewScenario Schemas
# ==============================================================================


class ScenarioQuestion(BaseModel):
    """Câu hỏi trong kịch bản phỏng vấn."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    text: str
    type: Literal["technical", "behavioral", "situational", "code"]
    level: Literal["intern", "junior", "middle", "senior"]
    source: str  # Mã câu hỏi ngân hàng hoặc "generated"
    based_on: str | None = None  # Bắt buộc khi source == "generated"

    @model_validator(mode="after")
    def validate_source_and_based_on(self) -> Self:
        if self.source == "generated" and (
            not self.based_on or not self.based_on.strip()
        ):
            raise ValueError("based_on is required when source is 'generated'")
        return self


class ScenarioSection(BaseModel):
    """Phần phỏng vấn tương ứng với một năng lực."""

    model_config = ConfigDict(extra="forbid")

    competency_id: str
    time_budget_min: float = Field(gt=0)
    questions: list[ScenarioQuestion]


class InterviewScenario(BaseModel):
    """Kịch bản phỏng vấn hoàn chỉnh được sinh từ hồ sơ năng lực."""

    model_config = ConfigDict(extra="forbid")

    role: str
    level: Literal["intern", "junior", "middle", "senior"]
    duration_min: float = Field(gt=0)
    sections: list[ScenarioSection]
    missing_bank_competencies: list[str] = Field(default_factory=list)


# ==============================================================================
# UC-03: TurnDecision Schemas
# ==============================================================================


class TurnDecision(BaseModel):
    """Quyết định điều phối lượt hội thoại của Interviewer Agent."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["follow_up", "next_question", "next_section", "clarify", "end"]
    reason: str
    utterance: str
    referenced_answer_span: str | None = None

    @model_validator(mode="after")
    def validate_follow_up(self) -> Self:
        if self.action == "follow_up" and (
            not self.referenced_answer_span or not self.referenced_answer_span.strip()
        ):
            raise ValueError("follow_up bắt buộc phải có referenced_answer_span")
        return self


# ==============================================================================
# UC-04: SandboxExecutionResult Schemas
# ==============================================================================


class SandboxExecutionResult(BaseModel):
    """Kết quả chạy code an toàn trong Docker Sandbox."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "compile_error", "runtime_error", "timeout", "system_error"]
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    errors: list[str] = Field(default_factory=list)
    timeout: bool = False
    stdout_truncated: str = ""
    runtime_ms: float = Field(ge=0.0)


# ==============================================================================
# UC-05 & UC-06: CompetencyScore & PanelModerationResult Schemas
# ==============================================================================


class AnswerEvidence(BaseModel):
    """Dẫn chứng câu trả lời từ transcript phỏng vấn."""

    model_config = ConfigDict(extra="forbid")

    turn_id: str
    quote: str


class CompetencyScore(BaseModel):
    """Điểm số đánh giá cho một năng lực dựa trên Rubric mật."""

    model_config = ConfigDict(extra="forbid")

    competency_id: str
    score: Literal[1, 2, 3, 4, 5, "insufficient_evidence"]
    rubric_level_matched: str | None = None
    evidence: list[AnswerEvidence] = Field(default_factory=list)
    signals_found: list[str] = Field(default_factory=list)
    signals_missing: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rubric_version: str

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.score != "insufficient_evidence" and len(self.evidence) == 0:
            raise ValueError("Chấm điểm bắt buộc phải có ít nhất 1 dẫn chứng evidence")
        return self


class GraderEvaluation(BaseModel):
    """Đánh giá độc lập từ một Grader Agent trong Panel."""

    model_config = ConfigDict(extra="forbid")

    grader_id: str
    grader_role: str
    scores: list[CompetencyScore]


class PanelModerationResult(BaseModel):
    """Kết quả hòa giải điểm số giữa các Grader trong Panel."""

    model_config = ConfigDict(extra="forbid")

    competency_id: str
    final_score: float | Literal["insufficient_evidence"]
    grader_scores: dict[str, int | str]
    max_disagreement: float
    explanation_round_triggered: bool
    needs_human_review: bool
    moderator_rationale: str


# ==============================================================================
# UC-07: CompactionSummary Schemas
# ==============================================================================


class CompactionSummary(BaseModel):
    """Tóm tắt nén ngữ cảnh phục vụ duy trì cửa sổ nhớ 5 mục cốt lõi."""

    model_config = ConfigDict(extra="forbid")

    session_goal: str = Field(min_length=5)
    evaluated_competencies: list[dict[str, Any]] = Field(min_length=1)
    current_open_question: str = Field(min_length=2)
    constraints: dict[str, Any]
    next_step: str = Field(min_length=3)

    @model_validator(mode="after")
    def validate_constraints(self) -> Self:
        required_keys = {"remaining_time_min", "language", "candidate_notes"}
        missing = required_keys - set(self.constraints.keys())
        if missing:
            raise ValueError(
                f"constraints thiếu các trường bắt buộc: {sorted(missing)}"
            )
        return self


# ==============================================================================
# UC-08: InterviewReport & HumanDecisionRecord Schemas
# ==============================================================================


class InterviewReport(BaseModel):
    """Báo cáo phỏng vấn tổng hợp trình duyệt cho Recruiter (HITL)."""

    model_config = ConfigDict(extra="forbid")

    candidate_code: str  # Chỉ dùng mã ứng viên, cấm chứa tên thật/PII
    summary: str
    competency_scores: list[CompetencyScore]
    strengths: list[str]
    areas_to_clarify: list[str]
    suggested_next_round_questions: list[str]
    recommend_level: Literal["strong", "moderate", "weak", "insufficient"]
    flags: list[
        Literal[
            "needs_human_review", "possible_ai_generated_answer", "injection_attempt"
        ]
    ]
    rubric_version: str
    model_version: str
    workflow_status: Literal["awaiting_human_decision"] = "awaiting_human_decision"


class HumanDecisionRecord(BaseModel):
    """Bản ghi quyết định nhân sự từ người có vai trò recruiter (HITL)."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    actor_id: str
    actor_role: str
    decision: Literal["advance", "reject", "hold"]
    notes: str
    differs_from_agent_recommendation: bool
    timestamp_iso: str


# ==============================================================================
# UC-09: Candidate Practice & Long-Term Memory (Hướng A: App luyện phỏng vấn)
# ==============================================================================


class QuestionCoachingFeedback(BaseModel):
    """Phản hồi huấn luyện chi tiết cho từng câu hỏi sau phiên luyện tập."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    turn_id: str
    strengths_with_quotes: list[
        str
    ]  # Những điểm làm tốt kèm trích dẫn câu trả lời của ứng viên
    missing_signals: list[str]  # Các ý còn thiếu (bắt buộc thuộc expected_signals)
    improved_sample_answer: str  # Câu trả lời mẫu (theo khung STAR nếu là behavioral)


class SessionPracticeSummary(BaseModel):
    """Tóm tắt kết quả một phiên luyện tập trong lịch sử ứng viên."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    date_iso: str
    scores: dict[str, int | Literal["insufficient_evidence"]]


class CandidateLongTermMemory(BaseModel):
    """Bộ nhớ dài hạn qua nhiều phiên luyện tập của ứng viên (UC-09)."""

    model_config = ConfigDict(extra="forbid")

    candidate_code: str  # Chỉ lưu mã định danh ẩn danh, cấm lưu PII hay nguyên văn CV
    target_role: str
    target_level: Literal["intern", "junior", "middle", "senior"]
    weak_competencies: list[str]
    sessions: list[SessionPracticeSummary] = Field(default_factory=list)
    improvement_plan: list[str] = Field(default_factory=list)
    updated_at: str
    expires_at: str  # Thời hạn lưu trữ dữ liệu theo quy định mục 1.5
