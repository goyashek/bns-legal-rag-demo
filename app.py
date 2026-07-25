"""Agentic Legal RAG — public demo (Streamlit Community Cloud).

Open + rate-limited. Query embeddings come from Cloudflare Workers AI
(`@cf/baai/bge-large-en-v1.5`, the same weights that built the shipped index), so the
host needs no torch. Answer generation calls DeepSeek, so this app enforces a
per-session rate limit AND a global daily cap to bound cost. Exact-section lookups
(e.g. "BNS 103") use the deterministic fast path and cost nothing.

Secrets to set in Streamlit Cloud (App settings -> Secrets), NOT committed:
  CF_ACCOUNT_ID  = "<cloudflare account id>"
  CF_API_TOKEN   = "<token with Workers AI Read/Run>"
  LLM_API_KEY    = "<your DeepSeek API key>"
  LLM_BASE_URL   = "https://api.deepseek.com"
  LLM_EASY_MODEL = "deepseek-v4-flash"
  LLM_HARD_MODEL = "deepseek-v4-flash"
  LLM_DISABLE_THINKING = "true"
Also set a hard spending limit on the DeepSeek account as the ultimate backstop.
"""

from __future__ import annotations

import datetime
import os
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Streamlit Cloud stores config in st.secrets; the retrieval/LLM modules read os.environ,
# so mirror them across before anything imports the graph.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:  # noqa: BLE001 - no secrets file locally, which is fine
    pass

os.environ.setdefault("EMBED_BACKEND", "cloudflare")

# --- demo guardrails (tune these to your budget) ---
MAX_PER_SESSION = 15  # natural-language queries per browser session
MIN_INTERVAL_SEC = 8  # min seconds between queries in one session
GLOBAL_DAILY_CAP = 100  # total LLM-backed queries/day across ALL users (cost ceiling)


@st.cache_resource
def _global_counter() -> dict:
    """One counter per running instance; resets when the date rolls over."""
    return {"date": datetime.date.today(), "count": 0}


def _under_global_cap() -> bool:
    c = _global_counter()
    today = datetime.date.today()
    if c["date"] != today:
        c["date"], c["count"] = today, 0
    return c["count"] < GLOBAL_DAILY_CAP


@st.cache_resource(show_spinner="Loading retrieval index + embedding model (first load ~1 min)…")
def _load_answer_fn():
    from src.agent.graph import answer_query

    return answer_query


def _rate_limit_message() -> str | None:
    now = time.time()
    ss = st.session_state
    ss.setdefault("q_times", [])
    if ss["q_times"] and now - ss["q_times"][-1] < MIN_INTERVAL_SEC:
        wait = int(MIN_INTERVAL_SEC - (now - ss["q_times"][-1])) + 1
        return f"Please wait ~{wait}s between questions."
    if len(ss["q_times"]) >= MAX_PER_SESSION:
        return "Per-session demo limit reached. Refresh later, or run it locally for unlimited use."
    if not _under_global_cap():
        return "The shared daily demo limit has been reached. Please try again tomorrow."
    return None


st.set_page_config(page_title="Agentic Legal RAG — BNS/BNSS/BSA", page_icon="⚖️")
st.title("⚖️ Agentic Legal RAG — Indian Criminal Law")
st.caption(
    "Answers from the 2023 codes (BNS / BNSS / BSA) with citations checked in code. "
    "**Statutory information, not legal advice.** Public demo — rate-limited."
)

with st.expander("What this is / how it works"):
    st.markdown(
        "- Dense retrieval over the BNS/BNSS/BSA corpus, then a cited answer whose "
        "sections are verified against the retrieved text (deterministic citation check).\n"
        "- Exact-section queries like `BNS 103` use a fast path (no LLM).\n"
        "- Rate-limited and capped; not a legal service."
    )

answer_query = _load_answer_fn()

examples = ["punishment for theft", "someone attacked me with acid", "BNS 103", "is theft bailable?"]
st.write("Try: " + " · ".join(f"`{e}`" for e in examples))

query = st.text_input("Ask about the new criminal codes:", placeholder="e.g. what is the punishment for theft?")

if st.button("Ask", type="primary") and query.strip():
    msg = _rate_limit_message()
    if msg:
        st.warning(msg)
    else:
        st.session_state["q_times"].append(time.time())
        _global_counter()["count"] += 1
        try:
            with st.spinner("Retrieving + reasoning…"):
                state = answer_query(
                    query.strip(),
                    retrieval_mode="dense",
                    use_reranker=False,
                    pipeline="production",
                )
        except Exception as exc:  # noqa: BLE001 - demo surface, show a friendly message
            st.error(f"The model backend is unavailable right now ({type(exc).__name__}).")
        else:
            answer = state.get("answer") or state.get("fast_path_answer")
            if answer is None:
                st.info("No answer was produced for that query.")
            else:
                st.markdown(answer.answer)
                cites = getattr(answer, "citations", None) or []
                if cites:
                    st.markdown("**Cited sections:** " + ", ".join(f"{c.act} {c.section_id}" for c in cites))
                st.caption(f"confidence: {getattr(answer, 'confidence', '?')}")
