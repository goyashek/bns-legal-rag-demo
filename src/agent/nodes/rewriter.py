"""Query rewriter (HyDE-style). The re-retrieval arm of the self-correction loop.

Fires when the grader finds no relevant chunks or the citation validator
rejects a made-up citation. It rewrites/expands the query (a HyDE-style
hypothetical answer, or a targeted rephrase aimed at the bad citation) and then
retrieval runs again.

Bumps `iteration` each time. The graph owns the loop budget (2): once that's
spent it returns with confidence="low" instead of spinning forever.

The rewrite goes into `sub_queries` (what the next retrieve pass searches on), NOT
`query` — the original user query is preserved so the final answer still carries it
and retrieve_node keeps reranking against the user's actual intent. Client is
injected (defaults to shared Flash) so tests run at zero quota.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.agent.llm import get_client, load_prompt
from src.agent.state import AgentState

# Why the rewriter fired, which changes how the prompt steers the rewrite.
Reason = Literal["low_relevance", "invalid_citation", "unfaithful_answer"]


class RewrittenQuery(BaseModel):
    """Structured rewrite output. `instructor` forces the model to fill this."""

    query: str = Field(description="A single focused query for the next retrieval pass")


def rewrite_query(
    query: str,
    *,
    reason: Reason,
    invalid_citations: list[str] | None = None,
    client: object | None = None,
) -> str:
    """Produce a better query for the next retrieval pass.

    reason identifies whether retrieval missed, a citation was fabricated, or an
    answer overreached its cited text.
    invalid_citations are the sections that got fabricated last round, so I can
    nudge retrieval toward the right neighborhood instead of the wrong one. Falls
    back to the original query if the model returns nothing usable.
    """
    client = client or get_client("easy")
    prompt = load_prompt("rewriter").format(
        query=query,
        reason=reason,
        invalid_citations=", ".join(invalid_citations or []) or "(none)",
    )
    result: RewrittenQuery = client.create(  # type: ignore[attr-defined]
        messages=[{"role": "user", "content": prompt}],
        response_model=RewrittenQuery,
        temperature=0,
    )
    return result.query.strip() or query


def _reason_from_state(state: AgentState) -> Reason:
    """Infer why we're rewriting. Citation rejection takes priority over other failures."""
    if state.get("citation_valid") is False:
        return "invalid_citation"
    if state.get("faithful") is False:
        return "unfaithful_answer"
    return "low_relevance"


def rewriter_node(state: AgentState, *, client: object | None = None) -> AgentState:
    """LangGraph node. Sets fresh `sub_queries` and increments `iteration`.

    The graph's conditional edge (out of the grader/validator/checker) checks
    iteration against RETRIEVAL_LOOP_BUDGET and short-circuits to a low-confidence
    answer once we've run out of tries, so this node just does the rewrite + bump.
    """
    reason = _reason_from_state(state)
    new_query = rewrite_query(
        state["query"],
        reason=reason,
        invalid_citations=state.get("invalid_citations"),
        client=client,
    )
    iteration = state.get("iteration", 0) + 1
    notes = state.get("trace_notes", [])
    return {
        "sub_queries": [new_query],
        "iteration": iteration,
        "trace_notes": [*notes, f"rewriter[{reason}] iter={iteration}: {new_query[:60]!r}"],
    }
