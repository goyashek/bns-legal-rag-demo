"""Hybrid retrieval: dense (Qdrant) + sparse (BM25) fused with RRF.

Dense alone puts trespass near theft, which is wrong for legal text. BM25 alone
has no semantic flex. RRF (k=60) fuses the two. Output feeds the reranker, then
the agent graph.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.ingest.chunk_chonkie import LegalChunk
from src.retrieval.index import BGE_QUERY_INSTRUCTION, tokenize


@dataclass
class RetrievedChunk:
    """A chunk plus its retrieval scores, carried through the graph.

    I keep dense/sparse/rrf scores separate so the ablation notebook can attribute
    wins to each signal instead of one blended number.
    """

    chunk: LegalChunk
    rrf_score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None  # filled by rerank.Reranker; None before reranking


def reciprocal_rank_fusion(
    dense_ranking: list[str],
    sparse_ranking: list[str],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse two ranked chunk_id lists with RRF.

    Score per chunk = sum over the lists of 1 / (k + rank), rank 0-based. Pure
    function, no I/O, so it's easy to unit-test (tests/test_retrieval.py). Returns
    [(chunk_id, rrf_score)] sorted by score desc (ties broken by chunk_id for
    determinism).
    """
    scores: dict[str, float] = {}
    for ranking in (dense_ranking, sparse_ranking):
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class HybridRetriever:
    """Wraps the Qdrant + BM25 indices behind one `retrieve` call."""

    def __init__(
        self,
        *,
        collection: str,
        bm25_path: str,
        embed_model: str = "BAAI/bge-large-en-v1.5",
        qdrant_url: str | None = None,
        qdrant_path: str | Path = "data/processed/qdrant",
        rrf_k: int = 60,
    ) -> None:
        from qdrant_client import QdrantClient

        self.collection = collection
        self.rrf_k = rrf_k
        # EMBED_BACKEND=cloudflare serves the same bge weights over HTTP, so the demo
        # host does not need torch. Default stays the local sentence-transformers model.
        if os.getenv("EMBED_BACKEND", "local").lower() == "cloudflare":
            from src.retrieval.cf_embed import CloudflareEmbedder

            self.model = CloudflareEmbedder(embed_model)
        else:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(embed_model)
        self.client = (
            QdrantClient(url=qdrant_url) if qdrant_url else QdrantClient(path=str(qdrant_path))
        )

        with Path(bm25_path).open("rb") as f:
            saved = pickle.load(f)
        self.bm25 = saved["bm25"]
        self.bm25_chunk_ids: list[str] = saved["chunk_ids"]

    def _dense_search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Return [(chunk_id, cosine_score)] from Qdrant, best first."""
        vec = self.model.encode(
            BGE_QUERY_INSTRUCTION + query, normalize_embeddings=True, show_progress_bar=False
        )
        hits = self.client.query_points(
            self.collection, query=vec.tolist(), limit=limit, with_payload=True
        ).points
        return [(h.payload["chunk_id"], h.score) for h in hits]

    def _sparse_search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Return [(chunk_id, bm25_score)] from the in-process index, best first."""
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.bm25_chunk_ids, scores, strict=True), key=lambda kv: -kv[1])
        return ranked[:limit]

    def retrieve(
        self, query: str, *, top_k: int = 20, mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    ) -> list[RetrievedChunk]:
        """Return a hybrid, dense-only, or sparse-only ranking before reranking.

        The default hybrid mode over-fetches each arm before RRF so a chunk strong
        in one signal is not lost. Dense and sparse modes exist for the retrieval
        ablation and keep the same chunk/payload path as the production retriever.
        """
        if mode not in {"hybrid", "dense", "sparse"}:
            raise ValueError("mode must be 'hybrid', 'dense', or 'sparse'")

        pool = max(top_k * 2, 40)
        limit = pool if mode == "hybrid" else top_k
        dense = [] if mode == "sparse" else self._dense_search(query, limit)
        sparse = [] if mode == "dense" else self._sparse_search(query, limit)

        dense_ids = [cid for cid, _ in dense]
        sparse_ids = [cid for cid, _ in sparse]
        dense_score = {cid: s for cid, s in dense}
        sparse_score = {cid: s for cid, s in sparse}
        dense_rank = {cid: r for r, cid in enumerate(dense_ids)}
        sparse_rank = {cid: r for r, cid in enumerate(sparse_ids)}

        if mode == "hybrid":
            ranking = reciprocal_rank_fusion(dense_ids, sparse_ids, k=self.rrf_k)[:top_k]
        elif mode == "dense":
            ranking = dense
        else:
            ranking = sparse

        # Pull payloads for the winners in one Qdrant call.
        from qdrant_client import models

        winner_ids = [cid for cid, _ in ranking]
        payloads = self._payloads_for(winner_ids, models)

        results: list[RetrievedChunk] = []
        for cid, score in ranking:
            payload = payloads.get(cid)
            if payload is None:
                continue
            results.append(
                RetrievedChunk(
                    chunk=_chunk_from_payload(payload),
                    rrf_score=score,
                    dense_score=dense_score.get(cid),
                    sparse_score=sparse_score.get(cid),
                    dense_rank=dense_rank.get(cid),
                    sparse_rank=sparse_rank.get(cid),
                )
            )
        return results

    def _payloads_for(self, chunk_ids: list[str], models) -> dict[str, dict]:
        """Fetch payloads for a set of chunk_ids via a payload filter."""
        if not chunk_ids:
            return {}
        hits, _ = self.client.scroll(
            self.collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="chunk_id", match=models.MatchAny(any=chunk_ids))]
            ),
            limit=len(chunk_ids),
            with_payload=True,
        )
        return {h.payload["chunk_id"]: h.payload for h in hits}


def _chunk_from_payload(payload: dict) -> LegalChunk:
    return LegalChunk(
        chunk_id=payload["chunk_id"],
        act=payload["act"],
        section_id=payload["section_id"],
        heading=payload["heading"],
        text=payload["text"],
        summary=payload.get("summary", ""),
        chapter=payload.get("chapter"),
        metadata=payload.get("metadata", {}),
    )
