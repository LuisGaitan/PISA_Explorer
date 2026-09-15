"""OECD link errors for cross-cycle comparisons.

PISA scales are re-linked every cycle; the uncertainty of that linking is a
published constant per pair of cycles and domain (the "link error"). The
OECD adds it to the sampling variance of any change over time:
    SE(change) = sqrt(SE_first^2 + SE_last^2 + link_error^2)
It cannot be computed from the public-use microdata, so it has to be typed
in from the published table. Fill LINK_ERRORS from:
    PISA 2025 Results (Volume I), Annex A5 "Link errors" (comparisons of
    PISA 2025 with PISA 2022 and PISA 2018), and PISA 2022 Results (Volume
    I), Annex A7 for 2018 -> 2022.
Keys: (first cycle, last cycle, domain) with domain in MATH / READ / SCIE;
values in PISA score points. While a pair is missing, trend SEs omit the
link error and the provenance note says so.
"""

import re

LINK_ERRORS: dict[tuple[str, str, str], float] = {
    # ("2022", "2025", "MATH"): ...,
    # ("2022", "2025", "READ"): ...,
    # ("2022", "2025", "SCIE"): ...,
    # ("2018", "2025", "MATH"): ...,
    # ("2018", "2025", "READ"): ...,
    # ("2018", "2025", "SCIE"): ...,
    # ("2018", "2022", "MATH"): ...,
    # ("2018", "2022", "READ"): ...,
    # ("2018", "2022", "SCIE"): ...,
}

SOURCE = ("OECD, PISA 2025 Results (Volume I), Annex A5 (link errors); "
          "PISA 2022 Results (Volume I), Annex A7")

DOMAIN_RE = re.compile(r"\bPV(?:\{pv\}|\d{1,2})(MATH|READ|SCIE)\b")
DOMAIN_NAMES = {"MATH": "mathematics", "READ": "reading", "SCIE": "science"}


def domain_of(measure: str | None) -> str | None:
    """MATH / READ / SCIE when the measure is a plausible-value score."""
    m = DOMAIN_RE.search(str(measure or ""))
    return m.group(1) if m else None


def link_error(first: str, last: str, measure: str | None) -> float | None:
    """The published link error for a change from `first` to `last` in the
    measure's domain, or None when unknown / not an achievement score."""
    domain = domain_of(measure)
    if domain is None:
        return None
    return LINK_ERRORS.get((str(first), str(last), domain))


def loaded() -> bool:
    return bool(LINK_ERRORS)
