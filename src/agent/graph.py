"""Wires the production and experimental graph variants.

The live path is intentionally short:

    fast_path (hit) ......................................► END
        │ miss
        ▼
    router (out_of_scope / needs_clarification) ..► END (canned answer)
        │ criminal
        ▼
    dense retrieve → ood_gate (ood) .............► END ("not in corpus")
        │ in-corpus
        ▼
    generator → citation_validator (invalid) → citation_repair → validate once more
        │ valid                                      │ invalid
        ▼                                            ▼
       END                                   low-confidence response

The older full graph keeps the expander, grader, checker, and rewrite loop for
evaluation. It is not the API default because the fixed 20-case ablation and a
manual statute audit found that those nodes often turned citation-valid answers
into generic low-confidence replies.
"""

from __future__ import annotations

import re
from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.agent.state import AgentState
from src.ingest.chunk_chonkie import LegalChunk
from src.retrieval.hybrid import RetrievedChunk

RETRIEVAL_LOOP_BUDGET = 2
CITATION_REPAIR_BUDGET = 1
RetrievalMode = Literal["hybrid", "dense", "sparse"]
PipelineVariant = Literal["production", "baseline", "grader", "checker", "full"]

# Retrieval knobs: 20 candidates -> 12 base chunks, with bounded doctrine hints
# and missing siblings allowed to extend the generation context to 24.
RETRIEVE_K = 20
CONTEXT_K = 12
HINT_K = 8
MAX_CONTEXT_K = 24
# The real Qdrant collection is "legal" (see src/retrieval/index.py + the on-disk
# data/processed/qdrant/collection/legal). NOTE: .env.example still says
# QDRANT_COLLECTION=bns_sections — that default is stale; the index build names it
# "legal". Kept here so the graph doesn't silently query an empty collection.
QDRANT_COLLECTION = "legal"

# ponytail: these cheap hints cover the audited doctrine gaps; replace them with a
# learned router only if a broader evaluation shows the small table has hit its ceiling.
_LEGAL_HINT_PATTERNS = (
    (re.compile(r"\b(plan(?:ned|ning)?|agree(?:d|ment)?)\b", re.I), "criminal conspiracy"),
    (
        re.compile(r"\b(smash(?:ed|ing)?|damag(?:e|ed|ing)|destroy(?:ed|ing)?)\b", re.I),
        "mischief damage to property",
    ),
    (
        re.compile(
            r"(?=.*\b(vot(?:e|er|ers|ing)|election)\b)(?=.*\b(cash|pay|paid|brib\w*)\b)",
            re.I,
        ),
        "punishment for bribery",
    ),
    (
        re.compile(r"(?=.*\b(kill\w*|death|died)\b)(?=.*\b(sudden fight|provok\w*)\b)", re.I),
        "culpable homicide punishment",
    ),
    (
        re.compile(
            r"(?=.*\b(forg\w*|fake signature)\b)"
            r"(?=.*\b(deed|sale document|property document)\b)",
            re.I,
        ),
        "definition of valuable security",
    ),
)


def legal_search_hints(query: str) -> list[str]:
    """Return at most one deterministic doctrine phrase for a matched narrative."""
    return [hint for pattern, hint in _LEGAL_HINT_PATTERNS if pattern.search(query)][:1]


# --- routing decisions (pure; branch on state, no side effects) --------------


def route_after_fast_path(state: AgentState) -> str:
    """fast_path hit -> straight to END; miss -> the LLM router."""
    return END if state.get("fast_path_hit") else "router"


def route_after_router(state: AgentState) -> str:
    """criminal -> intent expansion + retrieval; terminal routes -> END (canned answer)."""
    if state.get("route") == "criminal":
        return "intent_expander"
    return END


def route_after_router_to_retrieve(state: AgentState) -> str:
    """Production route: keep scope control but skip intent expansion."""
    return "retrieve" if state.get("route") == "criminal" else END


def route_after_ood_gate(state: AgentState) -> str:
    """out-of-domain -> canned 'not in corpus' answer; in-corpus -> grade the chunks."""
    if state.get("ood"):
        return "not_in_corpus"
    return "grader"


def route_after_ood_gate_to_generator(state: AgentState) -> str:
    """Production route: generate after the in-corpus check without grading."""
    return "not_in_corpus" if state.get("ood") else "generator"


def route_after_grader(state: AgentState) -> str:
    """Any relevant chunk -> generation; else rewrite + re-retrieve until the budget runs out.

    iteration counts rewrites done so far (rewriter bumps it). While it's below the
    budget we loop back through the rewriter; once it's spent we stop with a
    low-confidence answer rather than spinning.
    """
    if state.get("grade_pass"):
        return "generator"
    if state.get("iteration", 0) < RETRIEVAL_LOOP_BUDGET:
        return "rewriter"
    return "low_confidence"


def route_after_citation_validator(state: AgentState) -> str:
    """valid citations -> faithfulness check; fabricated -> rewrite (within budget).

    A fabricated citation is the failure this whole project is built to catch, so
    on invalid we loop back through the rewriter (which sees citation_valid=False
    and rewrites in 'invalid_citation' mode). Budget spent -> low_confidence.
    """
    if state.get("citation_valid"):
        return "checker"
    if state.get("iteration", 0) < RETRIEVAL_LOOP_BUDGET:
        return "rewriter"
    return "low_confidence"


def route_after_checker(state: AgentState) -> str:
    """faithful -> done; unfaithful -> rewrite + regenerate (within budget)."""
    if state.get("faithful"):
        return END
    if state.get("iteration", 0) < RETRIEVAL_LOOP_BUDGET:
        return "rewriter"
    return "low_confidence"


def route_after_grader_once(state: AgentState) -> str:
    """Ablation route: a failed grader ends rather than changing the query."""
    return "generator" if state.get("grade_pass") else "low_confidence"


def route_after_citation_validator_once(state: AgentState) -> str:
    """Ablation route: an invalid citation ends rather than changing the query."""
    return END if state.get("citation_valid") else "low_confidence"


def route_after_production_citation_validator(state: AgentState) -> str:
    """Production gets one same-context repair before an invalid draft is refused."""
    if state.get("citation_valid"):
        return END
    if state.get("iteration", 0) < CITATION_REPAIR_BUDGET:
        return "citation_repair"
    return "low_confidence"


def route_after_citation_to_checker_once(state: AgentState) -> str:
    """Ablation route: only structurally valid answers reach the checker."""
    return "checker" if state.get("citation_valid") else "low_confidence"


def route_after_checker_once(state: AgentState) -> str:
    """Ablation route: measure the checker without the rewriter confound."""
    return END if state.get("faithful") else "low_confidence"


# --- orchestration node (lives here, not nodes/, since it drives the retrieval
#     layer rather than making an agent decision) -------------------------------

# The retriever + reranker load transformer models (~expensive), so build them
# once and reuse across queries. Cached like fast_path._resolver so importing the
# module stays cheap and tests can inject fakes instead.
_RETRIEVER = None
_RERANKER = None


def _retrieval_stack(*, with_reranker: bool = True):
    """Lazily build the retriever and, when needed, the reranker."""
    global _RETRIEVER, _RERANKER
    if _RETRIEVER is None:
        from src.retrieval.hybrid import HybridRetriever

        _RETRIEVER = HybridRetriever(
            collection=QDRANT_COLLECTION, bm25_path="data/processed/bm25.pkl"
        )
    if with_reranker and _RERANKER is None:
        from src.retrieval.rerank import Reranker

        _RERANKER = Reranker()
    return _RETRIEVER, _RERANKER if with_reranker else None


def reset_retrieval_stack() -> None:
    """Close the cached retriever's Qdrant client and drop the cache.

    In EMBEDDED mode Qdrant takes a file lock on data/processed/qdrant and allows
    only one client per path, so the long-lived cache would otherwise block any
    other client in the same process (e.g. the retrieval integration tests). Call
    this to release the lock — tests use it on teardown; a server never needs to.
    """
    global _RETRIEVER, _RERANKER
    if _RETRIEVER is not None:
        client = getattr(_RETRIEVER, "client", None)
        if client is not None:
            client.close()
    _RETRIEVER = None
    _RERANKER = None


def _dedupe_by_chunk_id(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Collapse duplicates across sub-queries, keeping the best RRF score per chunk."""
    best: dict[str, RetrievedChunk] = {}
    for c in chunks:
        cid = c.chunk.chunk_id
        if cid not in best or c.rrf_score > best[cid].rrf_score:
            best[cid] = c
    return sorted(best.values(), key=lambda c: c.rrf_score, reverse=True)


def _unique_in_order(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep the first occurrence of each chunk without changing ranking order."""
    seen: set[str] = set()
    unique: list[RetrievedChunk] = []
    for chunk in chunks:
        if chunk.chunk.chunk_id not in seen:
            seen.add(chunk.chunk.chunk_id)
            unique.append(chunk)
    return unique


def _complete_repeated_sections(
    chunks: list[RetrievedChunk], corpus: list[LegalChunk] | None = None
) -> list[RetrievedChunk]:
    """Add missing siblings when two chunks from one section already ranked."""
    counts: dict[tuple[str, str], int] = {}
    for result in chunks:
        key = (result.chunk.act, result.chunk.section_id)
        counts[key] = counts.get(key, 0) + 1
    repeated = {key for key, count in counts.items() if count > 1}
    if not repeated:
        return chunks

    if corpus is None:
        from src.agent.nodes.fast_path import _resolver

        corpus = _resolver()[0]
    present = {result.chunk.chunk_id for result in chunks}
    extras = [
        RetrievedChunk(chunk=chunk, rrf_score=0.0)
        for chunk in corpus
        if (chunk.act, chunk.section_id) in repeated and chunk.chunk_id not in present
    ]
    return [*chunks, *extras]


def retrieve_node(
    state: AgentState,
    *,
    retriever=None,
    reranker=None,
    mode: RetrievalMode = "hybrid",
    use_reranker: bool = True,
) -> AgentState:
    """Fan retrieval over sub-queries and optionally rerank the result.

    retriever/reranker are injectable so the fan+dedupe logic tests with fakes; by
    default the cached real stack is used. Reranks against the ORIGINAL query (the
    user's actual intent), not the sub-queries, so the final ordering reflects what
    was asked rather than one decomposed facet.
    """
    if retriever is None:
        retriever, default_reranker = _retrieval_stack(with_reranker=use_reranker)
        if reranker is None:
            reranker = default_reranker

    explicit_sub_queries = state.get("sub_queries") or []
    sub_queries = explicit_sub_queries or [state["query"]]
    pooled: list[RetrievedChunk] = []
    for sq in sub_queries:
        pooled.extend(retriever.retrieve(sq, top_k=RETRIEVE_K, mode=mode))

    deduped = _dedupe_by_chunk_id(pooled)
    ranked = (
        reranker.rerank(state["query"], deduped, top_k=CONTEXT_K)
        if use_reranker and reranker is not None
        else deduped[:CONTEXT_K]
    )
    hints = [] if explicit_sub_queries else legal_search_hints(state["query"])
    for hint in hints:
        ranked.extend(retriever.retrieve(hint, top_k=HINT_K, mode=mode))
    ranked = _complete_repeated_sections(_unique_in_order(ranked))[:MAX_CONTEXT_K]

    notes = state.get("trace_notes", [])
    return {
        "retrieved": ranked,
        "trace_notes": [
            *notes,
            f"retrieve: {len(sub_queries)} sub-queries -> {len(pooled)} hits "
            f"-> {len(deduped)} unique -> {len(ranked)} {mode}"
            + (f" + {len(hints)} hint" if hints else "")
            + (" + rerank" if use_reranker else ""),
        ],
    }


def not_in_corpus_node(state: AgentState) -> AgentState:
    """Terminal for the OOD case: a low-confidence 'not in the statutes' answer."""
    from src.models.schemas import LegalAdvice

    notes = state.get("trace_notes", [])
    answer = LegalAdvice(
        query=state["query"],
        answer=(
            "I couldn't find this in the BNS, BNSS, or BSA. It may fall outside the "
            "criminal statutes I cover, or be phrased in a way I can't match to a section."
        ),
        confidence="low",
        in_corpus=False,
    )
    return {"answer": answer, "trace_notes": [*notes, "not_in_corpus: OOD terminal"]}


def low_confidence_node(state: AgentState) -> AgentState:
    """Terminal for retrieval, citation, or grounding failures."""
    from src.models.schemas import LegalAdvice

    notes = state.get("trace_notes", [])
    if state.get("faithful") is False:
        message = (
            "I found related statutory sections but couldn't verify a fully grounded answer "
            "from them. Try naming the specific offence or act, or rephrasing."
        )
    elif state.get("citation_valid") is False:
        message = (
            "I couldn't verify the generated citations against the retrieved statutory "
            "sections. Try naming the specific offence or act, or rephrasing."
        )
    else:
        message = (
            "I couldn't retrieve enough clearly on-point sections to answer this "
            "confidently. Try naming the specific offence or act, or rephrasing."
        )
    answer = LegalAdvice(
        query=state["query"],
        answer=message,
        confidence="low",
        in_corpus=True,
    )
    return {"answer": answer, "trace_notes": [*notes, "low_confidence: terminal"]}


# --- graph assembly ----------------------------------------------------------


def build_graph(
    *,
    retrieval_mode: RetrievalMode = "dense",
    use_reranker: bool = False,
    pipeline: PipelineVariant = "production",
):
    """Construct the live graph, a fixed ablation, or the legacy full graph.

    The three ablations skip entry controls so each downstream stage has one job
    to earn in the RAGAS comparison. The full graph remains available to reproduce
    the earlier experiment.
    """
    if pipeline not in {"production", "baseline", "grader", "checker", "full"}:
        raise ValueError("pipeline must be production, baseline, grader, checker, or full")
    from src.agent.nodes.checker import checker_node
    from src.agent.nodes.citation_validator import citation_validator_node
    from src.agent.nodes.fast_path import fast_path_node
    from src.agent.nodes.generator import citation_repair_node, generator_node
    from src.agent.nodes.grader import grader_node
    from src.agent.nodes.intent_expander import intent_expander_node
    from src.agent.nodes.ood_gate import ood_gate_node
    from src.agent.nodes.rewriter import rewriter_node
    from src.agent.nodes.router import router_node

    builder = StateGraph(AgentState)
    builder.add_node(
        "retrieve",
        lambda state: retrieve_node(state, mode=retrieval_mode, use_reranker=use_reranker),
    )
    builder.add_node("generator", generator_node)
    builder.add_node("citation_validator", citation_validator_node)
    builder.add_node("low_confidence", low_confidence_node)

    if pipeline == "production":
        builder.add_node("fast_path", fast_path_node)
        builder.add_node("router", router_node)
        builder.add_node("ood_gate", ood_gate_node)
        builder.add_node("not_in_corpus", not_in_corpus_node)
        builder.add_node("citation_repair", citation_repair_node)

        builder.add_edge(START, "fast_path")
        builder.add_conditional_edges("fast_path", route_after_fast_path, ["router", END])
        builder.add_conditional_edges("router", route_after_router_to_retrieve, ["retrieve", END])
        builder.add_edge("retrieve", "ood_gate")
        builder.add_conditional_edges(
            "ood_gate", route_after_ood_gate_to_generator, ["not_in_corpus", "generator"]
        )
        builder.add_edge("generator", "citation_validator")
        builder.add_conditional_edges(
            "citation_validator",
            route_after_production_citation_validator,
            [END, "citation_repair", "low_confidence"],
        )
        builder.add_edge("citation_repair", "citation_validator")
        builder.add_edge("not_in_corpus", END)
        builder.add_edge("low_confidence", END)
        return builder.compile()

    if pipeline != "full":
        builder.add_edge(START, "retrieve")
        if pipeline == "baseline":
            builder.add_edge("retrieve", "generator")
        else:
            builder.add_node("grader", grader_node)
            builder.add_edge("retrieve", "grader")
            builder.add_conditional_edges(
                "grader", route_after_grader_once, ["generator", "low_confidence"]
            )
        builder.add_edge("generator", "citation_validator")
        if pipeline == "checker":
            builder.add_node("checker", checker_node)
            builder.add_conditional_edges(
                "citation_validator",
                route_after_citation_to_checker_once,
                ["checker", "low_confidence"],
            )
            builder.add_conditional_edges(
                "checker", route_after_checker_once, [END, "low_confidence"]
            )
        else:
            builder.add_conditional_edges(
                "citation_validator", route_after_citation_validator_once, [END, "low_confidence"]
            )
        builder.add_edge("low_confidence", END)
        return builder.compile()

    builder.add_node("fast_path", fast_path_node)
    builder.add_node("router", router_node)
    builder.add_node("intent_expander", intent_expander_node)
    builder.add_node("ood_gate", ood_gate_node)
    builder.add_node("grader", grader_node)
    builder.add_node("rewriter", rewriter_node)
    builder.add_node("checker", checker_node)
    builder.add_node("not_in_corpus", not_in_corpus_node)

    builder.add_edge(START, "fast_path")
    builder.add_conditional_edges("fast_path", route_after_fast_path, ["router", END])
    builder.add_conditional_edges("router", route_after_router, ["intent_expander", END])
    builder.add_edge("intent_expander", "retrieve")
    builder.add_edge("retrieve", "ood_gate")
    builder.add_conditional_edges("ood_gate", route_after_ood_gate, ["not_in_corpus", "grader"])
    builder.add_conditional_edges(
        "grader", route_after_grader, ["generator", "rewriter", "low_confidence"]
    )
    builder.add_edge("rewriter", "retrieve")  # loop back: re-retrieve on the rewritten query
    builder.add_edge("generator", "citation_validator")
    builder.add_conditional_edges(
        "citation_validator",
        route_after_citation_validator,
        ["checker", "rewriter", "low_confidence"],
    )
    builder.add_conditional_edges(
        "checker", route_after_checker, ["rewriter", "low_confidence", END]
    )
    builder.add_edge("not_in_corpus", END)
    builder.add_edge("low_confidence", END)

    return builder.compile()


# --- entry point -------------------------------------------------------------

_GRAPH = None


def answer_query(
    query: str,
    *,
    retrieval_mode: RetrievalMode = "dense",
    use_reranker: bool = False,
    pipeline: PipelineVariant = "production",
) -> AgentState:
    """Run the cached production graph or an explicitly selected experiment."""
    global _GRAPH
    if pipeline != "production" or retrieval_mode != "dense" or use_reranker:
        return build_graph(
            retrieval_mode=retrieval_mode,
            use_reranker=use_reranker,
            pipeline=pipeline,
        ).invoke({"query": query, "trace_notes": [], "iteration": 0})
    if _GRAPH is None:
        _GRAPH = build_graph(retrieval_mode="dense", use_reranker=False, pipeline="production")
    return _GRAPH.invoke({"query": query, "trace_notes": [], "iteration": 0})
