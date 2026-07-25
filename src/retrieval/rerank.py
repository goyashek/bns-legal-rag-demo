"""Cross-encoder reranker, on by default.

Reranking is pretty standard now, so bge-reranker-base is in from the start. I
ablate it in notebooks/02_retrieval_ablation.ipynb to see how many points it
actually buys. It cuts the ~20 RRF-fused candidates down to the ~8 the agent
reasons over.
"""

from __future__ import annotations

from dataclasses import replace

from src.retrieval.hybrid import RetrievedChunk


class Reranker:
    """bge-reranker-base cross-encoder scoring (query, chunk) pairs."""

    def __init__(self, model: str = "BAAI/bge-reranker-base") -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        *,
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        """Re-score the candidates by cross-encoder relevance, return top_k.

        I write the score back onto each chunk (rerank_score) so the ablation can
        compare ordering before vs after reranking. Aiming for ~200ms on ~20
        candidates. Scores against the bare section text (heading + body), not the
        summary-augmented copy, since the summary is a retrieval aid, not the law.
        """
        if not candidates:
            return []

        pairs = [(query, f"{c.chunk.heading}. {c.chunk.text}") for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        rescored = [
            replace(c, rerank_score=float(s)) for c, s in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=lambda c: c.rerank_score, reverse=True)
        return rescored[:top_k]
