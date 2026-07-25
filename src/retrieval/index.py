"""Build the retrieval indices: Qdrant (dense) + in-process BM25 (sparse).

Run once after ingestion. Reads data/processed/sections.jsonl and writes a Qdrant
collection plus a serialized BM25 index. Idempotent on chunk_id so re-running
doesn't duplicate anything.

Qdrant runs in embedded local mode by default (a path on disk, no server), so the
repo clones and runs without Docker for dev + CI. Pass qdrant_url to point at a
real server instead (that's what docker-compose wires up for deploy).
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import uuid
from pathlib import Path

from src.ingest.chunk_chonkie import LegalChunk

# bge-large-en-v1.5 is trained with an instruction prefix on the QUERY side only;
# passages are embedded bare. Keeping it here so index-time (bare) and query-time
# (prefixed) stay in sync with the retriever.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Stable namespace so a chunk_id always maps to the same Qdrant point id across runs
# (Qdrant needs int/uuid ids, not arbitrary strings), which is what makes upserts
# idempotent.
_POINT_NAMESPACE = uuid.UUID("1b6f0c4e-1e2a-4c3d-9f5a-2d7e6b8c9a01")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alnum tokenization shared by BM25 index build and query time.

    The stub warned about this: if index-time and query-time tokenization drift, the
    BM25 scores don't line up. So both sides import THIS function; don't reimplement.
    """
    return _TOKEN_RE.findall(text.lower())


def point_id_for(chunk_id: str) -> str:
    """Deterministic Qdrant point id (uuid5) for a chunk_id -> idempotent upserts."""
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def load_chunks(sections_jsonl: str | Path) -> list[LegalChunk]:
    """Load the enriched chunks from data/processed/sections.jsonl."""
    chunks: list[LegalChunk] = []
    with Path(sections_jsonl).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(LegalChunk(**json.loads(line)))
    return chunks


def _payload(chunk: LegalChunk) -> dict:
    """Qdrant point payload. Every field the retriever/validator reads must be here."""
    return {
        "chunk_id": chunk.chunk_id,
        "act": chunk.act,
        "section_id": chunk.section_id,
        "heading": chunk.heading,
        "text": chunk.text,
        "summary": chunk.summary,
        "chapter": chunk.chapter,
        "metadata": chunk.metadata,
    }


def build_qdrant_index(
    chunks: list[LegalChunk],
    *,
    collection: str,
    embed_model: str = "BAAI/bge-large-en-v1.5",
    qdrant_url: str | None = None,
    qdrant_path: str | Path = "data/processed/qdrant",
    recreate: bool = False,
    batch_size: int = 64,
) -> int:
    """Embed the chunks and upsert them into a Qdrant collection.

    Each point's payload includes chunk_id, act, section_id, heading, text, and the
    enrichment metadata. The retriever hands these back and the citation validator
    checks against them, so dropping a field here breaks downstream. Returns the
    number of points upserted.

    qdrant_url points at a server; when None, an embedded on-disk store at
    qdrant_path is used (no Docker needed).
    """
    from qdrant_client import QdrantClient, models
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embed_model)
    dim = model.get_sentence_embedding_dimension()

    if qdrant_url:
        client = QdrantClient(url=qdrant_url)
    else:
        Path(qdrant_path).mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(qdrant_path))

    exists = client.collection_exists(collection)
    if recreate and exists:
        client.delete_collection(collection)
        exists = False
    if not exists:
        client.create_collection(
            collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )

    # Passages embedded bare (no query instruction), normalized for cosine.
    texts = [c.text for c in chunks]
    vectors = model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )

    points = [
        models.PointStruct(id=point_id_for(c.chunk_id), vector=vec.tolist(), payload=_payload(c))
        for c, vec in zip(chunks, vectors, strict=True)
    ]
    for i in range(0, len(points), batch_size):
        client.upsert(collection, points=points[i : i + batch_size])
    return len(points)


def build_bm25_index(chunks: list[LegalChunk], *, out_path: str | Path) -> None:
    """Build and serialize an in-process rank_bm25 index over the chunk text.

    Kept in-process rather than standing up Elasticsearch, since it's only ~1k
    chunks. Tokenization uses the shared tokenize() so it matches query time exactly.
    Serialized as {bm25, chunk_ids} so the retriever can map a BM25 row back to its
    chunk_id.
    """
    from rank_bm25 import BM25Okapi

    corpus = [tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(corpus)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": [c.chunk_id for c in chunks]}, f)


def rebuild_local_index(
    *, raw_dir: str | Path = "data/raw", processed_dir: str | Path = "data/processed"
) -> tuple[int, int]:
    """Parse local source PDFs and replace the generated chunks and indices."""
    from src.ingest.chunk_chonkie import chunk_sections, write_chunks_jsonl
    from src.ingest.enrich_metadata import enrich
    from src.ingest.parse_pdf import PUBLISHED_SECTION_COUNTS, parse_statute, verify_section_counts

    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    statute_pdfs = {act: raw_dir / f"{act.lower()}.pdf" for act in ("BNS", "BNSS", "BSA")}
    comparison_pdf = raw_dir / "COMPARISON SUMMARY BNS to IPC .pdf"
    missing = [str(path) for path in (*statute_pdfs.values(), comparison_pdf) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source PDFs: " + ", ".join(missing))

    sections = [section for act, pdf in statute_pdfs.items() for section in parse_statute(pdf, act)]
    deltas = verify_section_counts(sections, PUBLISHED_SECTION_COUNTS)
    if any(deltas.values()):
        raise ValueError(f"Published section-count check failed: {deltas}")

    chunks = chunk_sections(sections)
    enrich(
        chunks,
        bnss_pdf=statute_pdfs["BNSS"],
        comparison_pdf=comparison_pdf,
        chapter_title_pdfs=statute_pdfs,
    )
    write_chunks_jsonl(chunks, processed_dir / "sections.jsonl")
    build_qdrant_index(
        chunks,
        collection="legal",
        qdrant_path=processed_dir / "qdrant",
        recreate=True,
    )
    build_bm25_index(chunks, out_path=processed_dir / "bm25.pkl")
    return len(sections), len(chunks)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description="Rebuild the local legal retrieval index")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args(argv)
    sections, chunks = rebuild_local_index(raw_dir=args.raw_dir, processed_dir=args.processed_dir)
    print(f"Built {chunks} chunks from {sections} sections.")


if __name__ == "__main__":
    main()
