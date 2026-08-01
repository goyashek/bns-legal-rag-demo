# BNS Legal RAG — public demo (BNS / BNSS / BSA)

Deployable demo of a retrieval-augmented QA system for India's 2023 criminal codes.
Dense retrieval over the BNS/BNSS/BSA corpus, a cited answer, and a **citation check
written in code** — every cited section must appear in the retrieved text or the answer
is rejected. Exact-section queries (`BNS 103`) use a deterministic fast path with no LLM.

> Statutory information, not legal advice. Not a substitute for a lawyer.

Main project and evaluation record: https://github.com/goyashek/bns-legal-rag

## What's different from the main repo

This copy is packaged to run on a 1 GB host, so it makes two substitutions:

- **Query embeddings via Cloudflare Workers AI** (`@cf/baai/bge-large-en-v1.5`) instead
  of loading `sentence-transformers` locally. Same weights that built the index, so the
  app needs ~250 MB instead of ~2 GB.
- **Prebuilt index committed** under `data/processed/` (16 MB), since the host has no
  source PDFs and no ingest step. Ingest, evaluation, and the API are not included.

The hosted embeddings are not bit-identical to the local model. Query vectors agree at
0.963 to 0.975 cosine, which reorders near-ties. Scored on the same 50 BNS development
scenarios (dense, no reranker) with the project's own harness:

| query embedder | P@5 | Recall@5 | MRR |
|---|---|---|---|
| local `bge-large-en-v1.5` | 0.200 | 0.750 | 0.706 |
| Cloudflare `bge-large-en-v1.5` | 0.208 | 0.777 | 0.673 |

On this development set, recall is marginally higher and MRR is marginally lower. The
headline evaluation numbers in the main repo were produced with the local
embedder and are not relabelled here.

The IPC→BNS bridge is inactive in this copy: it parses a government comparison PDF that
is not redistributed. Natural-language queries and exact BNS lookups work fully.

## Deploy (Streamlit Community Cloud)

1. Push this folder to a public GitHub repo.
2. On [share.streamlit.io](https://share.streamlit.io), click **New app**, pick the repo
   and `app.py`.
3. In **App settings → Secrets**, paste:

```toml
CF_ACCOUNT_ID = "..."          # Cloudflare account id
CF_API_TOKEN = "..."           # token with Workers AI Read/Run
LLM_API_KEY = "..."            # DeepSeek API key
LLM_BASE_URL = "https://api.deepseek.com"
LLM_EASY_MODEL = "deepseek-v4-flash"
LLM_HARD_MODEL = "deepseek-v4-flash"
LLM_DISABLE_THINKING = "true"
```

DeepSeek generation is metered, so `app.py` enforces a per-session rate limit and a
global daily cap. You should also set a hard spending limit on the DeepSeek account.

## Run locally

```bash
pip install -r requirements.txt
export CF_ACCOUNT_ID=... CF_API_TOKEN=... LLM_API_KEY=...
streamlit run app.py
```

Set `EMBED_BACKEND=local` (and install `sentence-transformers`) to embed on-device instead.

## Data & licensing

Corpus text is derived from the enacted acts published on
[India Code](https://indiacode.nic.in) (Govt of India). Code is MIT.
