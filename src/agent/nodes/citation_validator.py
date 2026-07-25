"""Deterministic citation validator.

The LLM hallucination checker (checker.py) still runs, but this sits in front of it
and is pure code, basically free on cost and latency:

  parse the structured citations and every section named in the answer, then check
  them against each other and the retrieved chunk set. If the answer cites BNS 307
  but only 306 was retrieved, reject it and go back to the rewriter.

This catches a confident citation to a section that was never retrieved. I'd rather
do it deterministically than trust another LLM call.

Section-level granularity: the generator legitimately cites a subsection ("318(2)")
but the corpus + retrieval are keyed at section level ("318"), so both sides are
normalized to the section before the membership test — the same trick fast_path uses
on the query side ("103(2)" -> "103"). A citation whose normalized section isn't in
the retrieved set is the real fabrication to reject.
"""

from __future__ import annotations

import re

from src.agent.legal_status import is_uncommenced
from src.agent.state import AgentState
from src.models.schemas import LegalAdvice
from src.retrieval.hybrid import RetrievedChunk

# Keep "103" and "63A", drop the "(1)"/"(2)" subsection tail. Matches
# fast_path.detect_exact_section's normalization so the two agree.
_SECTION_RE = re.compile(r"\d+[A-Z]?")
_PROSE_SECTION_ID = r"\d+[A-Z]?(?:\(\d+\))?"
_PROSE_SECTION_LIST = rf"{_PROSE_SECTION_ID}(?:\s*(?:,|and|or|/|&)\s*{_PROSE_SECTION_ID})*"
_PROSE_SECTION_ID_RE = re.compile(_PROSE_SECTION_ID)
_EXPLICIT_PROSE_SECTION_RE = re.compile(
    rf"\b(BNS|BNSS|BSA)\s+(?:Sections?\s+)?({_PROSE_SECTION_LIST})",
    re.IGNORECASE,
)
_PROSE_SECTION_RE = re.compile(
    rf"\bSections?\s+({_PROSE_SECTION_LIST})",
    re.IGNORECASE,
)


def normalize_section(section_id: str) -> str:
    """Reduce a printed section id to its section-level key (drop subsection)."""
    m = _SECTION_RE.match(section_id.strip())
    return m.group(0) if m else section_id.strip()


def extract_cited_sections(answer: LegalAdvice) -> list[tuple[str, str]]:
    """Pull every structured (act, section_id) citation.

    Act is upper-cased so the membership check is case-insensitive on the act code.
    """
    return [(c.act.strip().upper(), c.section_id.strip()) for c in answer.citations]


def extract_prose_sections(answer: LegalAdvice) -> list[tuple[str | None, str]]:
    """Pull explicit and unqualified statutory section mentions from the answer."""
    explicit = [
        (act.upper(), section)
        for act, section_list in _EXPLICIT_PROSE_SECTION_RE.findall(answer.answer)
        for section in _PROSE_SECTION_ID_RE.findall(section_list)
    ]
    prose_without_explicit = _EXPLICIT_PROSE_SECTION_RE.sub("", answer.answer)
    unqualified = [
        (None, section)
        for section_list in _PROSE_SECTION_RE.findall(prose_without_explicit)
        for section in _PROSE_SECTION_ID_RE.findall(section_list)
    ]
    return list(dict.fromkeys([*explicit, *unqualified]))


def validate_citations(
    answer: LegalAdvice,
    retrieved: list[RetrievedChunk],
) -> tuple[bool, list[str]]:
    """Check structured citations and prose section mentions against retrieval.

    Returns (all_valid, invalid_citations), where invalid_citations lists the
    "ACT SECTION" strings that were cited but not retrieved. Both sides are
    normalized to section level first, so citing "318(2)" is valid when section
    "318" was retrieved. A section named only in prose is rejected because downstream
    checks use the structured citation list to select supporting text.
    """
    retrieved_keys = {
        (c.chunk.act.strip().upper(), normalize_section(c.chunk.section_id)) for c in retrieved
    }
    structured = extract_cited_sections(answer)
    structured_keys = {(act, normalize_section(section)) for act, section in structured}
    structured_sections = {section for _, section in structured_keys}
    invalid: list[str] = []
    for act, section_id in structured:
        if (act, normalize_section(section_id)) not in retrieved_keys:
            invalid.append(f"{act} {section_id}")
        if is_uncommenced(act, section_id):
            invalid.append(f"{act} {section_id} (not in force)")
    for act, section_id in extract_prose_sections(answer):
        section = normalize_section(section_id)
        if act is not None and (act, section) not in structured_keys:
            invalid.append(f"{act} {section_id}")
        elif act is None and section not in structured_sections:
            invalid.append(f"SECTION {section_id}")
        if act is not None and is_uncommenced(act, section_id):
            invalid.append(f"{act} {section_id} (not in force)")
        elif act is None:
            for structured_act, structured_section in structured:
                if normalize_section(structured_section) == section and is_uncommenced(
                    structured_act, section_id
                ):
                    invalid.append(f"{structured_act} {section_id} (not in force)")
    invalid = list(dict.fromkeys(invalid))
    return (not invalid, invalid)


def citation_validator_node(state: AgentState) -> AgentState:
    """LangGraph node. Sets citation_valid + invalid_citations.

    On invalid, the graph routes back to the rewriter (within loop budget). An
    answer with no citations at all is treated as invalid — a substantive legal
    answer must cite something, and a citation-free pass would skip the whole point.
    """
    answer = state.get("answer")
    notes = state.get("trace_notes", [])
    if answer is None:
        return {
            "citation_valid": False,
            "invalid_citations": [],
            "trace_notes": [*notes, "citation_validator: no answer"],
        }

    # The generator only sees graded chunks when any passed the filter. Validating
    # against the larger retrieval pool could approve a citation to text the model
    # never received.
    generation_context = state.get("relevant_chunks") or state.get("retrieved", [])
    valid, invalid = validate_citations(answer, generation_context)
    # No citations at all is not a valid substantive answer.
    if not answer.citations:
        valid = False
    return {
        "citation_valid": valid,
        "invalid_citations": invalid,
        "trace_notes": [
            *notes,
            f"citation_validator: {'valid' if valid else 'invalid'}"
            + (f" {invalid}" if invalid else ""),
        ],
    }
