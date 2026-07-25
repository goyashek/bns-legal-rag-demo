"""Stage 1 of ingestion: parse BNS/BNSS/BSA gazette PDFs into raw sections.

Output feeds chunk_chonkie.chunk_sections. I wrote the parsing myself instead of
using a framework because the legal structure matters too much to hand off. I do
the Act -> Chapter -> Section hierarchy parse here to fix metadata boundaries
before Chonkie runs semantic chunking, so a bad split can't silently glue two
offences together.

What the real India Code PDFs actually look like (checked all three before writing
this):
  - Front matter is an "ARRANGEMENT OF SECTIONS" table of contents. Its entries are
    "<num>. <heading>." with NO dash-body, so they're separable from the real body.
  - Body sections are "<num>[A-Z]?. <heading><DASH> <body...>" where <DASH> is a long
    dash: em (U+2014 "—"), en (U+2013 "–"), or a double hyphen "--". A single hyphen
    does NOT start a body (that over-matched BNSS by one).
  - Edge cases I hit: BNS s.2 uses an en-dash where s.3+ use em-dashes; BNS s.217 has
    no period before the dash; running "SECTIONS"/"CHAPTER"/"PART" headers and bare
    page numbers repeat mid-body and have to be swept out of section text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Published section counts (from the India Code sourcing table). Used to seed the count gate.
PUBLISHED_SECTION_COUNTS: dict[str, int] = {"BNS": 358, "BNSS": 531, "BSA": 170}

# A long dash that opens a section body. Single hyphen deliberately excluded.
_DASH = r"(?:—|–|--)"

# Section start at beginning of a line: number, optional letter suffix (e.g. "111A"),
# a dot, the heading, then a long dash opening the body. Heading may wrap one line, so
# we allow a single embedded newline but stop before the next section number so a TOC
# line can't swallow forward into a later real body.
_SECTION_RE = re.compile(
    # number + optional letter suffix, a dot, then a separator that is EITHER whitespace
    # OR a dash pressed right against the dot (BNS s.255 prints as "255.—Heading.—Body").
    r"(?m)^[ \t]*(?P<num>\d+[A-Z]?)\.(?:[ \t]+|[ \t]*" + _DASH + r"[ \t]*)"
    r"(?P<heading>(?:(?!^[ \t]*\d+[A-Z]?\.).){0,240}?)"
    r"\.?[ \t]*" + _DASH,
    re.S,
)

# Running headers / noise to strip from a section body before it's stored.
_CHAPTER_RE = re.compile(r"(?m)^[ \t]*CHAPTER\s+([IVXLC]+)\s*$")
_PART_RE = re.compile(r"(?m)^[ \t]*PART\s+([IVXLC]+)\s*$")
_NOISE_LINE_RE = re.compile(r"(?m)^[ \t]*(?:SECTIONS?|ARRANGEMENT OF SECTIONS|\d{1,4})[ \t]*$")

# Post-body appendix. BNSS/BSA end with "THE (FIRST) SCHEDULE" of forms/tables that the
# last section would otherwise swallow (BNSS s.531 absorbed 100+ pages of warrant forms).
# BNS has no schedule. We only honour a schedule marker that falls after the last section.
_APPENDIX_RE = re.compile(r"THE\s+(?:FIRST\s+|SECOND\s+)?SCHEDULE\b")


@dataclass
class RawSection:
    """One statutory section lifted from the PDF, before chunking.

    section_id has to be unique within an act, and text needs to hold the full
    section body (sub-clauses included) without truncating mid-clause or leaking
    into the next section.
    """

    act: str  # "BNS" | "BNSS" | "BSA"
    section_id: str  # "103", "103(1)", "63A" as printed
    heading: str  # marginal/section heading
    text: str  # full section body
    chapter: str | None = None  # chapter number/title if present
    page_start: int | None = None
    page_end: int | None = None
    sub_clauses: list[str] = field(default_factory=list)


def _extract_pages(pdf_path: str | Path) -> list[str]:
    """Per-page text. Isolated so the pymupdf dependency stays in one place."""
    import fitz  # pymupdf; imported lazily so tests can skip cleanly when absent

    with fitz.open(pdf_path) as doc:
        return [page.get_text() for page in doc]


def _page_index(pages: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate pages and return (full_text, [(start_char, page_number)]).

    Page numbers are 1-based to match how a human reads the PDF. The offset table
    lets us map any char position in full_text back to the page it fell on.
    """
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for i, text in enumerate(pages):
        offsets.append((cursor, i + 1))
        parts.append(text)
        cursor += len(text) + 1  # +1 for the "\n" join below
    return "\n".join(parts), offsets


def _page_for(offset: int, offsets: list[tuple[int, int]]) -> int:
    """Map a char offset to its 1-based page number."""
    page = 1
    for start, pno in offsets:
        if offset >= start:
            page = pno
        else:
            break
    return page


def _find_body_start(full: str) -> int:
    """Char offset where the statute body begins (past the arrangement-of-sections TOC).

    All three bare acts open their body at section 1 with a dash-body
    ("1. Short title...—"). The arrangement-of-sections TOC lists section 1 too, but
    as a plain "1. Short title..." entry with no dash. I originally keyed off "first
    dash-match anywhere", but BNSS's TOC turned out to contain dash-bearing lines that
    fooled that, so I anchor on the first section-1 dash-match specifically.
    """
    for m in _SECTION_RE.finditer(full):
        if m.group("num") == "1":
            # Back up to the CHAPTER/PART heading introducing section 1, so the opening
            # chapter context isn't lost.
            head = max(full.rfind("CHAPTER", 0, m.start()), full.rfind("PART", 0, m.start()))
            return head if head != -1 else m.start()
    return 0


def _clean_body(text: str) -> str:
    """Strip running headers, bare page numbers, and collapse whitespace."""
    text = _NOISE_LINE_RE.sub("", text)
    # collapse 3+ newlines and trailing spaces, but keep clause structure readable
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_statute(pdf_path: str | Path, act: str) -> list[RawSection]:
    """Extract all sections from one statute PDF, one RawSection each, in order.

    act is the statute code ("BNS", "BNSS", or "BSA"). The parsed count should
    match the published section count and section_ids stay unique, so I log a
    warning on mismatch instead of passing silently.
    """
    pages = _extract_pages(pdf_path)
    full, offsets = _page_index(pages)

    body_start = _find_body_start(full)
    body = full[body_start:]

    # Collect chapter boundaries (offset -> roman numeral) within the body so each
    # section can be tagged with the chapter it falls under.
    chapters = [(m.start(), m.group(1)) for m in _CHAPTER_RE.finditer(body)]

    def chapter_at(pos: int) -> str | None:
        current = None
        for off, roman in chapters:
            if off <= pos:
                current = roman
            else:
                break
        return current

    # PDF footnotes can look like section starts, but they repeat a section number
    # already seen above. Drop duplicates before calculating body boundaries so a
    # skipped footnote cannot truncate the preceding real section.
    matches = []
    matched_ids: set[str] = set()
    for match in _SECTION_RE.finditer(body):
        if match.group("num") not in matched_ids:
            matches.append(match)
            matched_ids.add(match.group("num"))
    sections: list[RawSection] = []

    # Where the post-body appendix begins, so the final section stops there instead of
    # swallowing the schedule of forms. Only honour a marker that falls after the last
    # section start; the TOC also lists "THE FIRST SCHEDULE".
    body_end = len(body)
    if matches:
        appx = _APPENDIX_RE.search(body, matches[-1].start())
        if appx is not None:
            body_end = appx.start()

    for i, m in enumerate(matches):
        section_id = m.group("num")
        heading = re.sub(r"\s+", " ", m.group("heading")).strip().rstrip(".")
        body_from = m.end()
        body_to = matches[i + 1].start() if i + 1 < len(matches) else body_end
        raw_text = body[body_from:body_to]

        abs_start = body_start + m.start()
        abs_end = body_start + body_to
        sections.append(
            RawSection(
                act=act,
                section_id=section_id,
                heading=heading,
                text=_clean_body(raw_text),
                chapter=chapter_at(m.start()),
                page_start=_page_for(abs_start, offsets),
                page_end=_page_for(abs_end, offsets),
            )
        )

    expected = PUBLISHED_SECTION_COUNTS.get(act)
    if expected is not None and len(sections) != expected:
        logger.warning(
            "%s: parsed %d sections, published total is %d (delta %+d). "
            "Small deltas are usually dash-variant or lettered-section artifacts; "
            "inspect before trusting downstream.",
            act,
            len(sections),
            expected,
            len(sections) - expected,
        )
    return sections


def verify_section_counts(sections: list[RawSection], expected: dict[str, int]) -> dict[str, int]:
    """Check parsed section counts per act against published totals.

    expected is like {"BNS": 358, "BNSS": ..., "BSA": ...}. Returns {act: delta}
    where delta = parsed_count - expected_count, so all-zero means clean.
    """
    counts: dict[str, int] = {}
    for s in sections:
        counts[s.act] = counts.get(s.act, 0) + 1

    deltas: dict[str, int] = {}
    for act, exp in expected.items():
        delta = counts.get(act, 0) - exp
        deltas[act] = delta
        if delta != 0:
            logger.warning(
                "%s count off by %+d (parsed %d, expected %d)", act, delta, counts.get(act, 0), exp
            )
    return deltas
