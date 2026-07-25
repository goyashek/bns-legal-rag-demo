"""Exact-section fast path. Deterministic, runs before the LLM router.

If the query names an exact section ("BNS Section 103", "Section 302 IPC") I skip
embedding and the LLM entirely and just do a direct metadata lookup. Aiming for
under 50ms, and it avoids semantic drift on exact lookups.

IPC references get resolved through the IPC->BNS mapping first, so "302 IPC"
correctly returns BNS 103.

Precision over recall here, on purpose: a false positive (firing on a narrative
query and returning a wrong section with high confidence) is far worse than a false
negative (missing an explicit reference and falling through to the normal pipeline,
which would still answer). So the pattern only fires when an ACT CODE sits right next
to the number. "section 103" with no act is ambiguous about which statute, so it does
NOT fast-path; it falls through to the router/retriever.
"""

from __future__ import annotations

import re

from src.agent.state import AgentState
from src.ingest.chunk_chonkie import LegalChunk
from src.models.schemas import Citation, LegalAdvice

# Old-code -> new-code statute normalization. IPC/CrPC/Evidence are repealed; a hit on
# one is resolved onto the corpus (BNS/BNSS/BSA). IPC->BNS is section-level via the
# mapping; CrPC/Evidence normalize the ACT here but still need their own section maps
# (not built yet), so those only fast-path when the number already lines up.
_OLD_TO_NEW_ACT = {"IPC": "BNS", "CRPC": "BNSS", "EVIDENCE": "BSA"}
_CORPUS_ACTS = {"BNS", "BNSS", "BSA"}

# An explicit section reference: an act code adjacent to a section number, in either
# order, optionally with "section"/"s." between. Requiring the act is what keeps this
# from firing on bare numbers in narrative text.
# Act alternatives are ordered LONGEST-FIRST (BNSS before BNS, CrPC before ...), because
# regex alternation is greedy-by-order: "BNS|BNSS" would match "BNS" inside "BNSS" and
# misread "BNSS 173" as BNS. Longest-first makes "BNSS" win.
_ACT_ALT = r"BNSS|BNS|BSA|IPC|CrPC|Evidence"
SECTION_PATTERN = re.compile(
    rf"""
    \b(?:
        (?P<act1>{_ACT_ALT})\s*
        (?:section\s+|sec\.?\s*|s\.?\s*)?
        (?P<num1>\d+[A-Z]?(?:\(\d+\))?)
      |
        (?:section\s+|sec\.?\s*|s\.?\s*)?
        (?P<num2>\d+[A-Z]?(?:\(\d+\))?)\s*
        (?:of\s+the\s+|of\s+|,\s*)?
        (?P<act2>{_ACT_ALT})
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def detect_exact_section(
    query: str,
    *,
    ipc_bns_mapping: dict[str, str] | None = None,
) -> tuple[str, str] | None:
    """Return (act, section_id) if the query is an exact-section lookup, else None.

    The returned act is always a corpus act (BNS/BNSS/BSA): an IPC reference is
    resolved to its BNS section via ipc_bns_mapping. Only fires on an explicit
    act+number reference, never on narrative queries. Returns None if an old-code
    reference can't be resolved (e.g. IPC section with no mapping entry).
    """
    m = SECTION_PATTERN.search(query)
    if m is None:
        return None

    act = (m.group("act1") or m.group("act2")).upper()
    num = m.group("num1") or m.group("num2")
    # normalize the printed subsection form: keep "103" and "63A", drop "(1)" — the
    # corpus is keyed at section level.
    section_id = re.match(r"\d+[A-Z]?", num).group(0)  # type: ignore[union-attr]

    if act in _CORPUS_ACTS:
        return (act, section_id)

    new_act = _OLD_TO_NEW_ACT.get(act)
    if new_act == "BNS":
        if ipc_bns_mapping and section_id in ipc_bns_mapping:
            return ("BNS", ipc_bns_mapping[section_id])
        return None  # can't resolve an IPC section we don't have a mapping for
    # CrPC/Evidence: no section-level map yet, so don't guess.
    return None


def lookup_section(act: str, section_id: str, chunks: list[LegalChunk]) -> LegalChunk | None:
    """Return the first chunk for (act, section_id), or None if not in the corpus."""
    return next(
        (c for c in chunks if c.act == act and c.section_id == section_id),
        None,
    )


def lookup_section_chunks(act: str, section_id: str, chunks: list[LegalChunk]) -> list[LegalChunk]:
    """Return every ordered chunk belonging to an exact statutory section."""
    return sorted(
        (c for c in chunks if c.act == act and c.section_id == section_id),
        key=lambda c: c.chunk_id,
    )


def _classification_line(chunk: LegalChunk) -> str:
    """One deterministic line from the enriched metadata already on the chunk.

    cognizable/bailable are True/False/None (None = the BNSS First Schedule was
    conditional or conflicting, so enrich_metadata refused to guess — we stay
    silent rather than assert a flag we don't know). offence_category is the BNS
    chapter title. Pure lookup, no LLM: the audit flagged these fields as enriched
    but never surfaced; the exact-section path already holds the right chunk.
    """
    meta = chunk.metadata
    parts: list[str] = []
    if meta.get("cognizable") is not None:
        parts.append("cognizable" if meta["cognizable"] else "non-cognizable")
    if meta.get("bailable") is not None:
        parts.append("bailable" if meta["bailable"] else "non-bailable")
    line = ""
    if parts:
        line = f"Classification: {', '.join(parts)}."
    category = meta.get("offence_category")
    if category:
        line = f"{line} Category: {category}.".strip()
    return line


def build_fast_path_answer(query: str, chunks: list[LegalChunk]) -> LegalAdvice:
    """Assemble a direct, complete-section answer without an LLM call."""
    first = chunks[0]
    bodies = [c.text.removeprefix(f"{c.summary}\n\n") if c.summary else c.text for c in chunks]
    answer = f"{first.act} Section {first.section_id} — {first.heading}.\n\n{' '.join(bodies)}"
    classification = _classification_line(first)
    if classification:
        answer = f"{answer}\n\n{classification}"
    return LegalAdvice(
        query=query,
        answer=answer,
        citations=[Citation(act=first.act, section_id=first.section_id, heading=first.heading)],
        offences_identified=[first.heading],
        confidence="high",
        in_corpus=True,
    )


def fast_path_node(state: AgentState) -> AgentState:
    """LangGraph node. On an exact-section hit, set fast_path_hit + fast_path_answer.

    On a hit the graph routes straight to output, otherwise falls through to the
    router. Corpus + IPC mapping are resolved lazily (see _resolver) so importing
    this module stays cheap and the graph wiring in Week 2 can inject a shared one.
    """
    corpus, mapping = _resolver()
    hit = detect_exact_section(state["query"], ipc_bns_mapping=mapping)
    notes = state.get("trace_notes", [])

    if hit is not None:
        section_chunks = lookup_section_chunks(*hit, corpus)
        if section_chunks:
            return {
                "fast_path_hit": True,
                "fast_path_answer": build_fast_path_answer(state["query"], section_chunks),
                "trace_notes": [*notes, f"fast_path: hit {hit[0]} {hit[1]}"],
            }

    return {"fast_path_hit": False, "trace_notes": [*notes, "fast_path: miss"]}


# --- lazy corpus/mapping resolution -----------------------------------------
# Cached so repeated queries don't re-read disk. Week 2 graph wiring can replace this.
_CACHE: tuple[list[LegalChunk], dict[str, str]] | None = None


def _resolver() -> tuple[list[LegalChunk], dict[str, str]]:
    global _CACHE
    if _CACHE is None:
        from src.ingest.enrich_metadata import load_ipc_bns_mapping
        from src.retrieval.index import load_chunks

        chunks = load_chunks("data/processed/sections.jsonl")
        mapping: dict[str, str] = {}
        for c in chunks:
            for ipc in c.metadata.get("ipc_equivalents", []):
                mapping.setdefault(ipc, c.section_id)
        # metadata already carries the reverse mapping; fall back to the PDF only if empty
        if not mapping:
            try:
                mapping = load_ipc_bns_mapping("data/raw/COMPARISON SUMMARY BNS to IPC .pdf")
            except Exception:
                mapping = {}
        _CACHE = (chunks, mapping)
    return _CACHE
