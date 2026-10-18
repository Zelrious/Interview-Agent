"""RAG Knowledge Harness module for Interview Agent."""

from src.rag.agent_tools import (
    get_shared_retriever,
    retrieve_grading_rubric_tool,
    rewrite_query_for_retrieval,
    search_interview_questions_tool,
    set_shared_retriever,
)
from src.rag.grounding import (
    ESCAPE_HATCH_TEXT,
    GROUNDING_SYSTEM_RULES,
    format_grounded_context,
    is_escape_hatch,
    verify_citations,
)
from src.rag.hybrid_retriever import (
    HybridKnowledgeRetriever,
    reciprocal_rank_fusion,
    tokenize_for_bm25,
)
from src.rag.ingest import (
    DeterministicHashEmbeddings,
    chunk_documents_fixed_size,
    get_embeddings,
    load_and_chunk_knowledge_base,
    upsert_file_idempotent,
    verify_embedding_lock,
)
from src.rag.reranker import CascadeReranker, LightweightCrossAttentionScorer

__all__ = [
    "ESCAPE_HATCH_TEXT",
    "GROUNDING_SYSTEM_RULES",
    "CascadeReranker",
    "DeterministicHashEmbeddings",
    "HybridKnowledgeRetriever",
    "LightweightCrossAttentionScorer",
    "chunk_documents_fixed_size",
    "format_grounded_context",
    "get_embeddings",
    "get_shared_retriever",
    "is_escape_hatch",
    "load_and_chunk_knowledge_base",
    "reciprocal_rank_fusion",
    "retrieve_grading_rubric_tool",
    "rewrite_query_for_retrieval",
    "search_interview_questions_tool",
    "set_shared_retriever",
    "tokenize_for_bm25",
    "upsert_file_idempotent",
    "verify_citations",
    "verify_embedding_lock",
]
