"""Shared LangGraph state for the agent.

Every node gets this dict and returns a *partial* update (LangGraph merges it for me).
One dict I thread through the whole flow instead of passing a dozen args around.

The rewriter bumps `iteration`; the conditional edges bail out with `confidence="low"`
once it goes past RETRIEVAL_LOOP_BUDGET (2), so retrieval can't loop forever.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from src.models.schemas import LegalAdvice
from src.retrieval.hybrid import RetrievedChunk

Route = Literal["criminal", "out_of_scope", "needs_clarification"]


class AgentState(TypedDict, total=False):
    # --- input ---
    query: str  # original user query, verbatim

    # --- fast path (bypasses the rest of the graph on a hit) ---
    fast_path_hit: bool  # regex matched an exact section -> direct lookup
    fast_path_answer: LegalAdvice | None

    # --- routing ---
    route: Route

    # --- intent expansion ---
    sub_queries: list[str]  # 1 narrative -> 3-5 parallel offence sub-queries

    # --- retrieval ---
    retrieved: list[RetrievedChunk]  # base ranking plus bounded doctrine hints and siblings
    ood: bool  # out-of-domain gate tripped -> "not in corpus"

    # --- grading ---
    relevant_chunks: list[RetrievedChunk]  # chunks the grader marked "yes"
    grade_pass: bool  # at least one relevant chunk

    # --- generation ---
    answer: LegalAdvice | None

    # --- self-correction bookkeeping ---
    citation_valid: bool  # deterministic validator verdict
    faithful: bool  # LLM hallucination-checker verdict
    invalid_citations: list[str]  # sections cited but not in retrieved set
    iteration: int  # bounded rewrite or production-repair count
    confidence: Literal["high", "low"]  # "low" when loop budget exhausted

    # --- observability ---
    trace_notes: list[str]  # node-by-node breadcrumbs, handy for the README/demo
