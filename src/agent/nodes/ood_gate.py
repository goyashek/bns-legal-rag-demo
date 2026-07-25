"""Out-of-domain gate. Cheap distance-threshold rejection before spending LLM calls.

If the best retrieved chunk is beyond a cosine-distance threshold, the query just isn't
in BNS/BNSS/BSA, so short-circuit to "I can't find this in the criminal statutes" before
the grader or generator burn tokens. Picked 0.75 after eyeballing a few queries. This
complements the router's `out_of_scope`: the router catches the obvious non-criminal
stuff, this catches the criminal-sounding-but-not-in-corpus stuff.

Note on the number: the Qdrant collection is COSINE, so the retriever hands back a
similarity (higher = closer, ~[0,1] for normalized bge). The threshold here is a
DISTANCE (1 - similarity), so a query is out-of-domain when the best chunk's distance
exceeds 0.75, i.e. its similarity is below 0.25.
"""

from __future__ import annotations

from src.agent.state import AgentState
from src.retrieval.hybrid import RetrievedChunk


def is_out_of_domain(
    retrieved: list[RetrievedChunk],
    *,
    distance_threshold: float = 0.75,
) -> bool:
    """True if even the best chunk is too far to be in-corpus.

    Works off the top chunk's dense cosine similarity, converted to distance
    (1 - similarity). Nothing retrieved, or a top chunk with no dense score, counts
    as out-of-domain (there's nothing to stand on). The comparison is strict `>` so a
    chunk sitting exactly on the threshold is treated as in-corpus.
    """
    if not retrieved:
        return True

    # Best dense similarity across what came back (a chunk can rank via BM25 alone and
    # carry no dense score, so scan rather than trusting position 0).
    sims = [c.dense_score for c in retrieved if c.dense_score is not None]
    if not sims:
        return True

    best_distance = 1.0 - max(sims)
    return best_distance > distance_threshold


def ood_gate_node(state: AgentState) -> AgentState:
    """LangGraph node. Sets `ood`. On True, graph routes to a 'not in corpus' answer."""
    ood = is_out_of_domain(state.get("retrieved", []))
    notes = state.get("trace_notes", [])
    return {
        "ood": ood,
        "trace_notes": [*notes, f"ood_gate: {'out-of-domain' if ood else 'in-corpus'}"],
    }
