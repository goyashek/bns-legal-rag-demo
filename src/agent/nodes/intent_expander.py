"""Intent expander / fact extractor. Fans one messy narrative into sub-queries.

A common legal failure: "someone broke into my house and stole my phone" involves
house-trespass and theft, but top-k on the single query won't surface both. This node
The easy tier pulls out the distinct offences and emits 3-5 parallel sub-queries, one
per offence. The retriever runs per sub-query and results get deduped.

This is my fix for the cross-sectional reasoning problem, and the part that took me the
longest to get right. Runs after the router classifies the query as `criminal`.

Client is injected (defaults to the shared easy client) so node logic is
unit-testable with a fake client at zero quota; live tests gate on the DeepSeek key.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.agent.llm import get_client, load_prompt
from src.agent.state import AgentState


class SubQueries(BaseModel):
    """Structured expander output. `instructor` forces the model to fill this."""

    sub_queries: list[str] = Field(
        description="1-5 focused, retrieval-friendly sub-queries, one per distinct "
        "legal issue in the user's query",
    )


def _dedupe(queries: list[str]) -> list[str]:
    """Drop blanks and case-insensitive duplicates, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        q = q.strip()
        key = q.lower()
        if q and key not in seen:
            seen.add(key)
            out.append(q)
    return out


def expand_intent(
    query: str, *, max_sub_queries: int = 5, client: object | None = None
) -> list[str]:
    """Decompose a narrative into distinct offence-focused sub-queries.

    Returns 1-5 sub-queries. A simple single-offence query just returns [query]
    unchanged, since over-expanding only adds noise and cost. Falls back to the
    original query if the model returns nothing usable.
    """
    client = client or get_client("easy")
    prompt = load_prompt("intent_expander").format(query=query, max_sub_queries=max_sub_queries)
    result: SubQueries = client.create(  # type: ignore[attr-defined]
        messages=[{"role": "user", "content": prompt}],
        response_model=SubQueries,
        temperature=0,
    )
    subs = _dedupe(result.sub_queries)[:max_sub_queries]
    return subs or [query]


def intent_expander_node(state: AgentState, *, client: object | None = None) -> AgentState:
    """LangGraph node. Sets sub_queries. Downstream retrieval fans out over these."""
    subs = expand_intent(state["query"], client=client)
    notes = state.get("trace_notes", [])
    return {
        "sub_queries": subs,
        "trace_notes": [*notes, f"intent_expander: {len(subs)} sub-queries"],
    }
