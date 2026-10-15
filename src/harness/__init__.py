"""Harness package containing validation, PII redaction, and document parsing."""

from src.harness.pdf_parser import parse_document
from src.harness.pii_redactor import contains_unredacted_pii, redact_pii
from src.harness.validators import (
    is_exact_substring_quote,
    normalize_text,
    verify_coaching_feedback,
    verify_compaction_five_sections,
    verify_competency_score_quotes,
    verify_cv_evidence_quotes,
    verify_memory_no_pii,
    verify_no_hidden_test_leak,
    verify_scenario_constraints,
    verify_second_session_focus,
    verify_turn_decision_quote,
)

__all__ = [
    "contains_unredacted_pii",
    "is_exact_substring_quote",
    "normalize_text",
    "parse_document",
    "redact_pii",
    "verify_coaching_feedback",
    "verify_compaction_five_sections",
    "verify_competency_score_quotes",
    "verify_cv_evidence_quotes",
    "verify_memory_no_pii",
    "verify_no_hidden_test_leak",
    "verify_scenario_constraints",
    "verify_second_session_focus",
    "verify_turn_decision_quote",
]
