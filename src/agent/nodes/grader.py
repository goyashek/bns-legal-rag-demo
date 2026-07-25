"""Relevance grader. LLM-as-judge over the retrieved chunks, run in parallel.

Grades each reranked chunk yes/no on whether it's actually relevant to the query
(easy tier, fired concurrently). If any chunk passes I move on to generation;
if none do, the graph kicks over to the rewriter for another retrieval loop (budget 2).

The point is to stop generation from reasoning over context that's only
marginally on-topic. That's one of the quiet ways single-shot RAG gives you a
wrong-but-confident answer.

The fan-out is real (one call per chunk, concurrent via asyncio), matching the
"parallel over chunks" design — this is the biggest quota sink in the system, so
~8 calls per query against the free-tier cap. Client is injected (defaults to the
shared async easy client) so tests grade with a fake at zero quota.
"""

from __future__ import annotations

import asyncio
import os

from pydantic import BaseModel, Field

from src.agent.llm import get_client, load_prompt
from src.agent.state import AgentState
from src.retrieval.hybrid import RetrievedChunk

# The grader filters noise; it is not a completeness test. One complete statutory
# section can answer a narrow question such as theft, especially after chunk repair.
MIN_RELEVANT = 1

# Cap concurrent grade calls so a wide candidate set cannot burst past the easy-tier
# RPM limit. 8 reranked chunks sit under it, but the semaphore keeps it safe if the
# reranked set ever grows.
_MAX_CONCURRENCY = 8


class GradeVerdict(BaseModel):
    """Structured per-chunk verdict. `instructor` forces the model to fill this."""

    relevant: bool = Field(description="True if the chunk helps answer the query")


def _grade_prompt(query: str, chunk: RetrievedChunk) -> str:
    c = chunk.chunk
    return load_prompt("grader").format(
        query=query,
        act=c.act,
        section_id=c.section_id,
        heading=c.heading,
        text=c.text[:4000],  # cap: statutory bodies can be long; the head carries the offence
    )


async def _agrade_one(query: str, chunk: RetrievedChunk, client, sem: asyncio.Semaphore) -> bool:
    async with sem:
        verdict: GradeVerdict = await client.create(
            messages=[{"role": "user", "content": _grade_prompt(query, chunk)}],
            response_model=GradeVerdict,
            temperature=0,
        )
    return verdict.relevant


async def _agrade_chunks(
    query: str, chunks: list[RetrievedChunk], client, *, close_client: bool = False
) -> list[bool]:
    sem = asyncio.Semaphore(_MAX_CONCURRENCY)
    try:
        return await asyncio.gather(*(_agrade_one(query, c, client, sem) for c in chunks))
    finally:
        if close_client:
            await client.aclose()


def grade_chunks(
    query: str, chunks: list[RetrievedChunk], *, client: object | None = None
) -> list[RetrievedChunk]:
    """Grade all chunks in parallel and keep only the relevant ones (order preserved).

    Runs the async fan-out to completion. `client` is an async instructor client
    (its `.create` is awaited); defaults to the shared async easy client. Errors
    propagate — a quota/parse failure surfaces loudly rather than silently grading
    everything "not relevant" and firing a pointless rewrite loop.
    """
    if not chunks:
        return []
    # The experimental grader stays cheap by default but can use the hard tier.
    close_client = client is None
    if close_client:
        tier = os.getenv("GRADER_TIER", "easy").strip().lower()
        client = get_client("hard" if tier == "hard" else "easy", async_client=True)
    verdicts = asyncio.run(_agrade_chunks(query, chunks, client, close_client=close_client))
    return [c for c, keep in zip(chunks, verdicts, strict=True) if keep]


def grader_node(state: AgentState, *, client: object | None = None) -> AgentState:
    """LangGraph node. Sets relevant_chunks + grade_pass when any chunk is relevant."""
    relevant = grade_chunks(state["query"], state.get("retrieved", []), client=client)
    grade_pass = len(relevant) >= MIN_RELEVANT
    notes = state.get("trace_notes", [])
    return {
        "relevant_chunks": relevant,
        "grade_pass": grade_pass,
        "trace_notes": [
            *notes,
            f"grader: {len(relevant)} relevant -> {'pass' if grade_pass else 'rewrite'}",
        ],
    }
