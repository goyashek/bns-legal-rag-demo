"""Chonkie-based chunking. I ditched my earlier regex splitter, too brittle.

Stage 2 of ingestion. Takes `RawSection`, emits `LegalChunk` ready for metadata
enrichment and indexing.

How this works:
  1. Hierarchy is already set by parse_pdf (Act -> Chapter -> Section).
  2. Chonkie SemanticChunker (backbone potion-base-8M, cos 0.5, 512 tok) runs within
     each section as a safety net. For a statute this usually keeps one section = one
     chunk; only the few long sections (e.g. BNS 356 Defamation, ~8.7KB) split.
  3. Summary-augmented chunking: I prepend a one-sentence summary of the parent
     section to every chunk, which fixes the "right text, wrong parent" mismatch.

Two things I insulate against on purpose:
  - Chonkie's constructor arg is `threshold`; I expose `similarity_threshold` as my
    own public name and map it, so a Chonkie API rename doesn't ripple out here
    (Chonkie is young, so I keep the framework swappable).
  - The summariser is injectable. The default is a deterministic, key-free heading
    anchor, so index builds do not spend LLM tokens.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ingest.parse_pdf import RawSection

# A summariser maps (section_text, heading) -> one-line summary.
Summarizer = Callable[[str, str], str]


@dataclass
class LegalChunk:
    """A retrieval-ready chunk. Metadata gets filled across the chunk + enrich stages.

    The citation validator (agent/nodes/citation_validator.py) checks every
    generated citation against `act` + `section_id` on the retrieved chunks, so
    these fields have to stay exact.
    """

    chunk_id: str  # stable id, e.g. "BNS::103::0"
    act: str
    section_id: str
    heading: str
    text: str  # summary-augmented body (summary prepended)
    summary: str = ""  # the one-sentence parent summary
    chapter: str | None = None
    # enrichment fields (filled by enrich_metadata), see that module for semantics
    metadata: dict = field(default_factory=dict)


def _heading_summary(_text: str, heading: str) -> str:
    """Default key-free summary: the section heading is a decent parent anchor.

    It's what a lawyer skims to place a passage, and it needs no LLM call, so
    chunking stays reproducible in CI.
    """
    return heading.strip().rstrip(".")


def chunk_sections(
    sections: Iterable[RawSection],
    *,
    embedding_model: str = "minishlab/potion-base-8M",
    similarity_threshold: float = 0.5,
    max_tokens: int = 512,
    summary_augment: bool = True,
    summarizer: Summarizer | None = None,
) -> list[LegalChunk]:
    """Semantic-chunk each section, optionally prepending a parent summary.

    embedding_model is Chonkie's SemanticChunker backbone (its own default is
    minishlab/potion-base-32M; I pin 8M per the locked stack). Returns LegalChunks
    with chunk_id, act, section_id, heading and text set; metadata is left for
    enrich_metadata. chunk_id is stable across runs so index upserts stay idempotent,
    and every chunk traces back to exactly one (act, section_id).

    summarizer overrides how the prepended summary is produced; defaults to a
    deterministic heading anchor when summary_augment is on and none is passed.
    """
    from chonkie import SemanticChunker

    chunker = SemanticChunker(
        embedding_model=embedding_model,
        threshold=similarity_threshold,  # Chonkie's own arg name
        chunk_size=max_tokens,
    )
    if summary_augment and summarizer is None:
        summarizer = _heading_summary

    chunks: list[LegalChunk] = []
    for sec in sections:
        summary = summarizer(sec.text, sec.heading) if summary_augment and summarizer else ""

        pieces = chunker.chunk(sec.text)
        # A statutory section is a coherent legal unit. If the whole thing fits the
        # embedding budget, keep it as ONE chunk — splitting it hurts retrieval and
        # produces junk fragments (BNS s.1 "Short title", 356 tok, was splitting into
        # 5 pieces incl. 22- and 26-token scraps). Only fall back to Chonkie's semantic
        # split when the section genuinely overflows max_tokens (e.g. BNS s.356
        # Defamation, ~1940 tok). SemanticChunker returns [] on degenerate text, so
        # that also collapses to the whole section.
        if pieces and sum(p.token_count for p in pieces) > max_tokens:
            # SemanticChunker can return tiny adjacent fragments even when they fit
            # together. Repack those fragments up to the configured token budget so
            # a legal sentence never ends up split merely because an illustration
            # changed topic.
            units: list[tuple[str, int]] = []
            sentence: list[str] = []
            sentence_tokens = 0
            for piece in pieces:
                sentence.append(piece.text)
                sentence_tokens += piece.token_count
                if piece.text.rstrip().endswith((".", "?", "!")):
                    units.append((" ".join(sentence).strip(), sentence_tokens))
                    sentence = []
                    sentence_tokens = 0
            if sentence:
                units.append((" ".join(sentence).strip(), sentence_tokens))

            bodies = []
            current: list[str] = []
            current_tokens = 0
            for text, tokens in units:
                if current and current_tokens + tokens > max_tokens:
                    bodies.append(" ".join(current).strip())
                    current = []
                    current_tokens = 0
                current.append(text)
                current_tokens += tokens
            if current:
                bodies.append(" ".join(current).strip())
        else:
            bodies = [sec.text]

        for idx, body in enumerate(bodies):
            text = f"{summary}\n\n{body}" if summary else body
            chunks.append(
                LegalChunk(
                    chunk_id=f"{sec.act}::{sec.section_id}::{idx}",
                    act=sec.act,
                    section_id=sec.section_id,
                    heading=sec.heading,
                    text=text,
                    summary=summary,
                    chapter=sec.chapter,
                )
            )
    return chunks


def write_chunks_jsonl(chunks: Iterable[LegalChunk], path: str | Path) -> int:
    """Serialise chunks to newline-delimited JSON (data/processed/sections.jsonl).

    Returns the count written. This is the Week 1 Tue deliverable artifact that the
    indexer reads back in Thursday.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            n += 1
    return n
