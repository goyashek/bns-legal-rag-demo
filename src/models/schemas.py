"""Pydantic response models for the structured output.

`LegalAdvice` is the one output type for the whole system. I use `instructor` to
force the generator to fill it. The citation validator checks the structured list
against retrieval and rejects section numbers that appear only in the answer prose.
`Citation.act` / `Citation.section_id` have to stay aligned with `LegalChunk.act` /
`LegalChunk.section_id`, since the validator compares them directly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """One statutory citation. Has to be verifiable against a retrieved chunk."""

    act: str = Field(description="Statute code: BNS | BNSS | BSA")
    section_id: str = Field(description="Section as printed, e.g. '103' or '103(1)'")
    heading: str | None = Field(default=None, description="Section heading, if surfaced")
    quote: str | None = Field(
        default=None, description="Optional short verbatim snippet supporting the point"
    )


class LegalAdvice(BaseModel):
    """The system's answer. Every substantive claim should trace back to a Citation."""

    query: str
    answer: str = Field(description="Plain-language explanation grounded in the citations")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Every section named or relied on in the answer. The citation "
        "validator checks each one against the retrieved set.",
    )
    offences_identified: list[str] = Field(
        default_factory=list,
        description="Distinct offences the query maps to (from intent expansion)",
    )
    confidence: Literal["high", "low"] = Field(
        default="high",
        description="'low' when the self-correction loop ran out of budget, or OOD",
    )
    in_corpus: bool = Field(
        default=True,
        description="False when the OOD gate tripped (not found in BNS/BNSS/BSA)",
    )
    disclaimer: str = Field(
        default="This is statutory information, not legal advice. Consult a lawyer.",
    )
    trace_url: str | None = Field(
        default=None, description="LangSmith trace URL so you can audit the run"
    )


class QueryRequest(BaseModel):
    """POST /query body."""

    query: str = Field(min_length=1, max_length=2000)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    qdrant_connected: bool
    version: str
