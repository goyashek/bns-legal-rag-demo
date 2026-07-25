"""Answer generator. Pydantic-constrained cited advice via the hard tier.

Builds the LegalAdvice output from the graded chunks through `get_client("hard")`.

instructor returns citations as structured (act, section_id) pairs the
deterministic citation validator (Fri) checks exactly. The prompt forbids citing
anything outside the provided chunks; the validator enforces it, the prompt just
makes violations rare. Client is injected so tests generate with a fake at zero
quota.
"""

from __future__ import annotations

import re

from src.agent.legal_status import current_law_note
from src.agent.llm import get_client, load_prompt
from src.agent.state import AgentState
from src.models.schemas import LegalAdvice
from src.retrieval.hybrid import RetrievedChunk


def _omit_rejected_subsections(
    text: str,
    act: str,
    section_id: str,
    invalid_citations: list[str],
) -> str:
    """Hide a rejected subsection body from a repair prompt."""
    for rejected in invalid_citations:
        match = re.match(
            rf"(?i)^{re.escape(act)}\s+{re.escape(section_id)}(\(\d+\))",
            rejected,
        )
        if match is None:
            continue
        marker = re.escape(match.group(1))
        text = re.sub(
            rf"(?ms)^\s*{marker}(?=\s).*?(?=^\s*\(\d+\)(?=\s)|\Z)",
            "[Rejected subsection omitted from repair context.]\n",
            text,
            count=1,
        )
    return text


def _format_context(
    chunks: list[RetrievedChunk], *, invalid_citations: list[str] | None = None
) -> str:
    """Render the graded chunks as a citable list.

    One block per chunk, leading with the exact act + section_id the model must
    cite verbatim, so the returned Citation fields line up with what the validator
    checks against the retrieved set.
    """
    blocks: list[str] = []
    for c in chunks:
        ch = c.chunk
        note = current_law_note(ch.act, ch.section_id)
        text = ch.text[:4000]
        if invalid_citations:
            text = _omit_rejected_subsections(text, ch.act, ch.section_id, invalid_citations)
        block = f"[{ch.act} Section {ch.section_id}] {ch.heading}\n{text}"
        blocks.append(f"{block}\n{note}" if note else block)
    return "\n\n".join(blocks)


def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    previous_answer: LegalAdvice | None = None,
    invalid_citations: list[str] | None = None,
    client: object | None = None,
) -> LegalAdvice:
    """Generate structured LegalAdvice grounded in `chunks`.

    Two things I want to hold true: every citation points at a section that's
    actually in `chunks` (validator enforces it, prompt just makes it likely),
    and the output validates against LegalAdvice (instructor retries off-schema
    returns). `query` and `in_corpus` are set deterministically after generation
    so the model can't drift them.
    """
    client = client or get_client("hard")
    prompt = load_prompt("generator").format(
        query=query,
        context=_format_context(
            chunks,
            invalid_citations=invalid_citations if previous_answer is not None else None,
        ),
    )
    if previous_answer is not None:
        prompt += "\n\n" + load_prompt("citation_repair").format(
            invalid_citations=", ".join(invalid_citations or []) or "unspecified",
        )
    advice: LegalAdvice = client.create(  # type: ignore[attr-defined]
        messages=[{"role": "user", "content": prompt}],
        response_model=LegalAdvice,
        temperature=0,
        # The final schema includes citations and can exceed the control-node cap.
        max_tokens=1536,
    )
    # Pin the fields the pipeline owns, not the model.
    advice.query = query
    advice.in_corpus = True
    return advice


def generator_node(state: AgentState, *, client: object | None = None) -> AgentState:
    """LangGraph node. Sets `answer`. Flows straight into the citation validator.

    Generates over the graded-relevant chunks (the grader already filtered noise);
    falls back to the full retrieved set if grading left nothing recorded.
    """
    chunks = state.get("relevant_chunks") or state.get("retrieved", [])
    answer = generate_answer(state["query"], chunks, client=client)
    notes = state.get("trace_notes", [])
    return {
        "answer": answer,
        "trace_notes": [*notes, f"generator: {len(answer.citations)} citations"],
    }


def citation_repair_node(state: AgentState, *, client: object | None = None) -> AgentState:
    """Repair one citation-invalid draft without running retrieval again."""
    previous = state.get("answer")
    notes = state.get("trace_notes", [])
    iteration = state.get("iteration", 0) + 1
    if previous is None:
        return {
            "iteration": iteration,
            "trace_notes": [*notes, "citation_repair: skipped missing draft"],
        }

    chunks = state.get("relevant_chunks") or state.get("retrieved", [])
    repair_client = client or get_client("easy")
    answer = generate_answer(
        state["query"],
        chunks,
        previous_answer=previous,
        invalid_citations=state.get("invalid_citations", []),
        client=repair_client,
    )
    return {
        "answer": answer,
        "iteration": iteration,
        "trace_notes": [*notes, f"citation_repair: {len(answer.citations)} citations"],
    }
