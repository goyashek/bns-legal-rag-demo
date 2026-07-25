"""Hallucination checker. LLM-as-judge faithfulness pass, after the deterministic validator.

The deterministic citation validator (citation_validator.py) already guaranteed
every cited section was actually retrieved. This node catches the subtler failure:
claims that paraphrase or overstate what the cited sections really say. DeepSeek
Flash judges each claim against the text it cites.

If it comes back "unfaithful", route back to the rewriter/generator, as long as
we're still inside the loop budget.

Client is injected (defaults to shared Flash) so the node logic tests with a fake
at zero quota; live tests gate on the key.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.agent.llm import get_client, load_prompt
from src.agent.state import AgentState
from src.models.schemas import LegalAdvice
from src.retrieval.hybrid import RetrievedChunk


class FaithfulnessVerdict(BaseModel):
    """Structured faithfulness verdict. `instructor` forces the model to fill this."""

    faithful: bool = Field(description="True only if every claim is backed by the cited text")


def _cited_context(answer: LegalAdvice, chunks: list[RetrievedChunk]) -> str:
    """Render the text of the sections the answer actually cites.

    The checker only needs the cited sections, not the whole retrieved set — that's
    what "are these claims supported by their sources" means. Falls back to all
    chunks if nothing lines up (shouldn't happen post-validator, but stay robust).
    """
    cited = {(c.act.strip().upper(), c.section_id.strip()) for c in answer.citations}
    picked = [
        c for c in chunks if (c.chunk.act.strip().upper(), c.chunk.section_id.strip()) in cited
    ]
    use = picked or chunks
    return "\n\n".join(
        f"[{c.chunk.act} Section {c.chunk.section_id}] {c.chunk.heading}\n{c.chunk.text[:4000]}"
        for c in use
    )


def check_faithfulness(
    answer: LegalAdvice,
    chunks: list[RetrievedChunk],
    *,
    client: object | None = None,
) -> tuple[bool, list[str]]:
    """Judge whether every claim is actually backed by its cited source text.

    Returns the verdict plus an empty compatibility list. The graph routes only on
    the boolean, so asking for arbitrary claim excerpts wastes tokens and can make
    a short structured response exceed its output limit.
    """
    client = client or get_client("easy")
    prompt = load_prompt("checker").format(
        answer=answer.answer, context=_cited_context(answer, chunks)
    )
    verdict: FaithfulnessVerdict = client.create(  # type: ignore[attr-defined]
        messages=[{"role": "user", "content": prompt}],
        response_model=FaithfulnessVerdict,
        temperature=0,
        max_tokens=256,
    )
    return (verdict.faithful, [])


def checker_node(state: AgentState, *, client: object | None = None) -> AgentState:
    """LangGraph node. Sets `faithful`. Basically terminal: faithful -> output."""
    answer = state["answer"]
    # Judge only the chunks supplied to generation. The larger retrieval pool can
    # contain a section that supports a claim but was filtered out before prompting
    # the generator.
    generation_context = state.get("relevant_chunks") or state.get("retrieved", [])
    faithful, unsupported = check_faithfulness(answer, generation_context, client=client)
    notes = state.get("trace_notes", [])
    return {
        "faithful": faithful,
        "trace_notes": [
            *notes,
            f"checker: {'faithful' if faithful else 'unfaithful'}"
            + (f" {unsupported}" if unsupported else ""),
        ],
    }
