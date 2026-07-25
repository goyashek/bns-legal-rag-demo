"""Stage 3 of ingestion: attach only the metadata the agent actually routes on.

I kept scope tight here. The agent filters on offence category and the
cognizable/bailable flags, and the IPC->BNS bridge feeds both the fast path and
the MCQ eval, so those fields earn their place. Anything else is just noise.

Reality-vs-plan note: the scaffold stub assumed a metadata
CSV and a mapping CSV. Neither exists as a clean table. What I actually have in
data/raw/ are PDFs, so the sources are:
  - offence_category: the chapter TITLE the section sits under (my parser already
    fixes the chapter hierarchy; here I resolve roman -> title). E.g. BNS VI ->
    "Of Offences Affecting The Human Body".
  - cognizable / bailable: the BNSS FIRST SCHEDULE "Classification of Offences"
    table (pages ~172-217), keyed by BNS section number. It's a BORDERLESS 6-column
    table that defeats line-based table detection, so I cluster words by x-position.
    Verified against known offences (BNS 103 murder = cognizable + non-bailable) and
    checked for the one dangerous failure mode, a "Non-" prefix drifting out of its
    column and flipping non-cognizable -> cognizable (zero occurrences).
  - ipc_equivalents: reversed from the "COMPARISON SUMMARY BNS to IPC" table.

Discipline that matters for legal data: a WRONG cognizable/bailable flag is worse
than a missing one. So conditional entries ("According as offence abetted is
bailable or non-bailable"), and sub-sections that disagree once aggregated to the
section level, resolve to None, never a guess. Coverage is deliberately partial.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

from src.ingest.chunk_chonkie import LegalChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# offence_category: chapter roman numeral -> chapter title, per act
# ---------------------------------------------------------------------------

_CHAPTER_TITLE_RE = re.compile(r"CHAPTER\s+([IVXLC]+)\s*\n\s*([A-Z][A-Z ,'\-&]+?)\s*\n", re.M)


def load_chapter_titles(pdf_path: str | Path) -> dict[str, str]:
    """Map chapter roman numeral -> Title-Cased chapter title for one statute.

    The chapter title is the section's offence category ("Of Offences Affecting
    The Human Body"). First occurrence wins (the TOC and the body both print the
    heading; either is fine, they're identical).
    """
    import fitz

    with fitz.open(pdf_path) as doc:
        full = "\n".join(page.get_text() for page in doc)

    titles: dict[str, str] = {}
    for m in _CHAPTER_TITLE_RE.finditer(full):
        roman, raw = m.group(1), " ".join(m.group(2).split())
        if roman in titles or len(raw) <= 4 or "SECTION" in raw:
            continue
        titles[roman] = raw.title()
    return titles


# ---------------------------------------------------------------------------
# cognizable / bailable: BNSS First Schedule "Classification of Offences"
# ---------------------------------------------------------------------------

# Column x-position bands (points), calibrated against the schedule pages. The table
# is borderless so these are geometric, not ruled: section | offence | punishment |
# cognizable | bailable | triable-by.
_BAND_SEC = 100
_BAND_MID = 300
_BAND_COGN = 365
_BAND_BAIL = 450

# A section anchor in column 1: "103", "59(b)", "61(2)(a)". We keep only the leading
# integer, aggregating sub-sections up to the section the chunks are keyed on.
_SEC_ANCHOR_RE = re.compile(r"^(\d+)(?:\([0-9a-z]+\))*$")


def _band(x: float) -> str:
    if x < _BAND_SEC:
        return "sec"
    if x < _BAND_MID:
        return "mid"
    if x < _BAND_COGN:
        return "cogn"
    if x < _BAND_BAIL:
        return "bail"
    return "court"


def _classify_cell(text: str, kind: str) -> bool | None:
    """Read one cognizable/bailable cell. None for conditional/blank/unreadable.

    Order-independent: checks the negative ("non-...") first regardless of where the
    "Non-" token sorted, so a drifted prefix can't be misread as the positive.
    """
    t = text.lower()
    if "according as" in t or "same as" in t or not t.strip():
        return None
    stem = "cognizable" if kind == "cogn" else "bailable"
    if stem not in t:
        return None
    # negative wins if any "non" appears alongside the stem
    if "non" in t:
        return False
    return True


def _schedule_span(doc) -> tuple[int, int]:
    """Page range (start, end) of the BNS section of the First Schedule."""
    start = None
    for p in range(doc.page_count):
        if "OFFENCES UNDER THE BHARATIYA NYAYA SANHITA" in doc[p].get_text():
            start = p
            break
    if start is None:
        return (0, 0)
    end = doc.page_count
    for p in range(start + 1, doc.page_count):
        if "AGAINST OTHER LAWS" in doc[p].get_text():
            end = p
            break
    return (start, end)


def _aggregate(vals: list[bool | None]) -> bool | None:
    """Section-level consensus: the single agreed value, else None on conflict/empty."""
    distinct = {v for v in vals if v is not None}
    return next(iter(distinct)) if len(distinct) == 1 else None


def load_offence_classification(bnss_pdf: str | Path) -> dict[str, dict[str, bool | None]]:
    """Parse the BNSS First Schedule -> {bns_section: {cognizable, bailable}}.

    Values are True/False/None; None means the schedule was conditional, blank, or
    the section's sub-rows disagreed. Only sections that actually appear as offences
    in the schedule are returned (definitions and procedural sections won't).
    """
    import fitz

    with fitz.open(bnss_pdf) as doc:
        start, end = _schedule_span(doc)
        if start == end:
            logger.warning("BNSS First Schedule not located; no cognizable/bailable data")
            return {}

        rows_by_section: dict[str, list[tuple[bool | None, bool | None]]] = defaultdict(list)
        for p in range(start, end):
            words = doc[p].get_text("words")  # (x0,y0,x1,y1, text, block,line,word)
            visual_rows: dict[int, dict[str, list[tuple[float, str]]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for x0, y0, _x1, _y1, text, *_ in words:
                visual_rows[round(y0 / 4) * 4][_band(x0)].append((x0, text))

            for y in sorted(visual_rows):
                cells = visual_rows[y]
                anchor = next(
                    (
                        _SEC_ANCHOR_RE.match(t).group(1)  # type: ignore[union-attr]
                        for _, t in sorted(cells["sec"])
                        if _SEC_ANCHOR_RE.match(t)
                    ),
                    None,
                )
                if anchor is None:
                    continue
                cogn = " ".join(t for _, t in sorted(cells["cogn"]))
                bail = " ".join(t for _, t in sorted(cells["bail"]))
                rows_by_section[anchor].append(
                    (_classify_cell(cogn, "cogn"), _classify_cell(bail, "bail"))
                )

    return {
        sec: {
            "cognizable": _aggregate([c for c, _ in rows]),
            "bailable": _aggregate([b for _, b in rows]),
        }
        for sec, rows in rows_by_section.items()
    }


# ---------------------------------------------------------------------------
# ipc_equivalents: reversed from the BNS -> IPC comparison table
# ---------------------------------------------------------------------------

_IPC_CELL_RE = re.compile(r"\d+[A-Z]?")

# Column x-bands (points) in the "COMPARISON SUMMARY BNS to IPC" table. BNS section is
# leftmost (x~51), IPC section is the second numeric column (x~302); the subject and
# summary prose sit outside these bands and are ignored. pymupdf's find_tables drifts
# its column boundaries page-to-page here, so, like the schedule, we cluster by x.
_IPC_MAP_BNS_MAX = 150
_IPC_MAP_IPC_MIN = 250
_IPC_MAP_IPC_MAX = 360


def load_ipc_bns_mapping(comparison_pdf: str | Path) -> dict[str, str]:
    """Load the IPC->BNS section mapping, e.g. {"302": "103", "379": "303"}.

    Parsed from the "COMPARISON SUMMARY BNS to IPC" table (BNS section in the left
    column, IPC section in the second numeric column). The fast path uses it to
    resolve "302 IPC" -> BNS 103, and mcq_eval uses it because a lot of the
    BhashaBench criminal slice still cites repealed IPC. We collapse BNS sub-sections (2(8)) to the
    section number and skip "New" (no IPC equivalent). First write wins, so an IPC
    section that appears once maps to the first BNS section it lines up with.
    """
    import fitz

    mapping: dict[str, str] = {}
    with fitz.open(comparison_pdf) as doc:
        for page in doc:
            rows: dict[int, dict[str, list[tuple[float, str]]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for x0, y0, _x1, _y1, text, *_ in page.get_text("words"):
                if x0 < _IPC_MAP_BNS_MAX:
                    col = "bns"
                elif _IPC_MAP_IPC_MIN < x0 < _IPC_MAP_IPC_MAX:
                    col = "ipc"
                else:
                    continue
                rows[round(y0 / 3) * 3][col].append((x0, text))

            for y in sorted(rows):
                bns_tokens = [t for _, t in sorted(rows[y]["bns"]) if re.match(r"^\d+", t)]
                if not bns_tokens:
                    continue
                bns_sec = re.match(r"^(\d+)", bns_tokens[0]).group(1)  # type: ignore[union-attr]
                ipc_raw = " ".join(t for _, t in sorted(rows[y]["ipc"]))
                if "new" in ipc_raw.lower():
                    continue
                for ipc in _IPC_CELL_RE.findall(ipc_raw):
                    mapping.setdefault(ipc, bns_sec)
    return mapping


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def enrich(
    chunks: list[LegalChunk],
    *,
    bnss_pdf: str | Path | None = None,
    comparison_pdf: str | Path | None = None,
    chapter_title_pdfs: dict[str, str | Path] | None = None,
    ipc_bns_mapping: dict[str, str] | None = None,
) -> list[LegalChunk]:
    """Fill chunk.metadata with the routing/filtering fields.

    Writes offence_category (str | None), cognizable/bailable (bool | None), and
    ipc_equivalents (list[str], the IPC sections that map to this BNS section).
    Sources are resolved from the PDFs when the corresponding arg is given; anything
    unavailable stays absent/None rather than guessed.

    - bnss_pdf: source for cognizable/bailable (BNSS First Schedule).
    - comparison_pdf / ipc_bns_mapping: IPC<->BNS bridge. Pass a pre-loaded mapping to
      skip re-parsing (e.g. the fast path already loaded it).
    - chapter_title_pdfs: {act: pdf_path} to resolve offence_category from chapter
      titles. Only acts present here get a category.
    """
    classification = load_offence_classification(bnss_pdf) if bnss_pdf else {}

    if ipc_bns_mapping is None and comparison_pdf is not None:
        ipc_bns_mapping = load_ipc_bns_mapping(comparison_pdf)
    # reverse IPC->BNS into BNS-section -> [IPC sections]
    bns_to_ipc: dict[str, list[str]] = defaultdict(list)
    for ipc, bns in (ipc_bns_mapping or {}).items():
        bns_to_ipc[bns].append(ipc)

    titles_by_act: dict[str, dict[str, str]] = {}
    for act, pdf in (chapter_title_pdfs or {}).items():
        titles_by_act[act] = load_chapter_titles(pdf)

    for chunk in chunks:
        category = None
        if chunk.chapter and chunk.act in titles_by_act:
            category = titles_by_act[chunk.act].get(chunk.chapter)
        chunk.metadata["offence_category"] = category

        flags = classification.get(chunk.section_id) if chunk.act == "BNS" else None
        chunk.metadata["cognizable"] = flags["cognizable"] if flags else None
        chunk.metadata["bailable"] = flags["bailable"] if flags else None

        chunk.metadata["ipc_equivalents"] = (
            sorted(bns_to_ipc.get(chunk.section_id, []), key=lambda s: (len(s), s))
            if chunk.act == "BNS"
            else []
        )

    return chunks
