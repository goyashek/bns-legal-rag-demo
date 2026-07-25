"""Small current-law exceptions that the bare-act text cannot express.

India Code still records BNS 106(2) as excluded from the 1 July 2024
commencement notification. I checked the official record on 2026-07-21:
https://www.indiacode.nic.in/show-data?actid=AC_CEN_5_23_00048_2023-45_1719292564123&orderno=1
"""

from __future__ import annotations

UNCOMMENCED_PROVISIONS = {("BNS", "106(2)")}


def is_uncommenced(act: str, section_id: str) -> bool:
    """Return whether an exact provision is present in the Act but not in force."""
    return (act.strip().upper(), section_id.strip().upper()) in UNCOMMENCED_PROVISIONS


def current_law_note(act: str, section_id: str) -> str | None:
    """Return a generation note for sections with a known commencement exception."""
    if (act.strip().upper(), section_id.split("(", 1)[0].strip().upper()) == ("BNS", "106"):
        return "CURRENT-LAW NOTE: BNS 106(2) has not commenced. Do not present it as law in force."
    return None
