"""Query router. Easy-tier classifier that runs once the fast path misses.

Sorts the query into one of three routes:
  criminal            -> go on to intent expansion + retrieval
  out_of_scope        -> polite "not a criminal-law question" answer
  needs_clarification -> ask a follow-up

This is the semantic gate. The cheap fast path (fast_path.py) and the
distance-based OOD gate (ood_gate.py) wrap around it, so this only fires when
those two can't decide. I keep the prompt text in agent/prompts/ so it isn't
buried in code.

The LLM client is injected (defaults to the shared easy client) so node logic
is unit-testable with a fake client at zero quota; see tests/test_agent_nodes.py.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.agent.llm import get_client, load_prompt
from src.agent.state import AgentState, Route

# Canned replies for the two terminal routes. Deterministic (no second LLM call)
# so the graph can end immediately without spending tokens on a boilerplate reply.
OUT_OF_SCOPE_REPLY = (
    "I only cover Indian criminal law under the BNS, BNSS, and BSA (2023). "
    "That question looks outside criminal law, so I can't answer it reliably."
)
NEEDS_CLARIFICATION_REPLY = (
    "I can help with Indian criminal law (BNS/BNSS/BSA), but I need a bit more to "
    "go on. What happened, or which offence, section, or procedure do you mean?"
)


class RouteDecision(BaseModel):
    """Structured router output. `instructor` forces the model to fill this."""

    route: Route = Field(description="criminal | out_of_scope | needs_clarification")


def classify(query: str, *, client: object | None = None) -> Route:
    """Return the route for a query. Easy tier, temperature 0 so it is stable.

    `client` is any object with `.create(messages=..., response_model=...)`
    returning a RouteDecision — the real instructor client by default, a fake in
    tests.
    """
    client = client or get_client("easy")
    prompt = load_prompt("router").format(query=query)
    decision: RouteDecision = client.create(  # type: ignore[attr-defined]
        messages=[{"role": "user", "content": prompt}],
        response_model=RouteDecision,
        temperature=0,
    )
    return decision.route


def router_node(state: AgentState, *, client: object | None = None) -> AgentState:
    """LangGraph node. Sets `route`. The graph branches on the value.

    For the two terminal routes it also drops a canned `answer` into state so the
    graph can end without a generation call.
    """
    route = classify(state["query"], client=client)
    notes = state.get("trace_notes", [])
    update: AgentState = {
        "route": route,
        "trace_notes": [*notes, f"router: {route}"],
    }

    if route in ("out_of_scope", "needs_clarification"):
        from src.models.schemas import LegalAdvice

        reply = OUT_OF_SCOPE_REPLY if route == "out_of_scope" else NEEDS_CLARIFICATION_REPLY
        update["answer"] = LegalAdvice(
            query=state["query"],
            answer=reply,
            confidence="low",
            in_corpus=(route != "out_of_scope"),
        )
    return update
