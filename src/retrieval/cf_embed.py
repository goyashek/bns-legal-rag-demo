"""Cloudflare Workers AI query embedder (`@cf/baai/bge-large-en-v1.5`).

Same weights as the local `sentence-transformers` model that built the index, served
over HTTP so the demo host does not need torch (~2GB RAM -> ~250MB). Only query
embeddings go through here; the corpus vectors were built locally and ship prebuilt.

Drop-in for the retriever: exposes `.encode(text, normalize_embeddings=..., ...)`
returning a numpy array, so `HybridRetriever._dense_search` is unchanged.

Env:
  CF_ACCOUNT_ID    Cloudflare account id
  CF_API_TOKEN     API token with the Workers AI Read/Run permission
"""

from __future__ import annotations

import os

import numpy as np

_CF_MODELS = {
    "BAAI/bge-large-en-v1.5": "@cf/baai/bge-large-en-v1.5",
    "BAAI/bge-base-en-v1.5": "@cf/baai/bge-base-en-v1.5",
    "BAAI/bge-small-en-v1.5": "@cf/baai/bge-small-en-v1.5",
}
_TIMEOUT = 30.0


class CloudflareEmbedder:
    """Minimal query-time embedder backed by the Workers AI REST API."""

    def __init__(self, embed_model: str = "BAAI/bge-large-en-v1.5") -> None:
        account = os.getenv("CF_ACCOUNT_ID")
        token = os.getenv("CF_API_TOKEN")
        if not account or not token:
            raise RuntimeError("CF_ACCOUNT_ID and CF_API_TOKEN must be set for the cloudflare backend")
        try:
            cf_model = _CF_MODELS[embed_model]
        except KeyError:
            raise ValueError(f"no Workers AI equivalent for {embed_model!r}") from None

        import httpx

        self.model_name = cf_model
        self._url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{cf_model}"
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {token}"}, timeout=_TIMEOUT
        )

    def encode(
        self,
        text: str | list[str],
        *,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,  # noqa: ARG002 - signature parity with SentenceTransformer
        **_: object,
    ) -> np.ndarray:
        """Embed one string (or a list) and return float32 vectors."""
        single = isinstance(text, str)
        payload = {"text": [text] if single else list(text)}
        resp = self._client.post(self._url, json=payload)
        resp.raise_for_status()
        body = resp.json()
        if not body.get("success", False):
            raise RuntimeError(f"Workers AI error: {body.get('errors')}")

        vectors = np.asarray(body["result"]["data"], dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.clip(norms, 1e-12, None)
        return vectors[0] if single else vectors
