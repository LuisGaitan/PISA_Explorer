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
    # OECD link errors (score points), one per cycle pair and domain; common
    # to every economy. As published in PISA 2025 Results (Volume I) and PISA
    # 2022 Results (Volume I), reproduced in the NCES PISA 2025 Technical
    # Notes (Table 4) and PISA 2022 Technical Notes (Table 3).
    ("2022", "2025", "MATH"): 1.220,
    ("2022", "2025", "READ"): 1.094,
    ("2022", "2025", "SCIE"): 3.116,
    ("2018", "2025", "MATH"): 2.551,
    ("2018", "2025", "READ"): 1.832,
    ("2018", "2025", "SCIE"): 3.507,
    ("2018", "2022", "MATH"): 2.24,
    ("2018", "2022", "READ"): 1.47,
    ("2018", "2022", "SCIE"): 1.61,
}

SOURCE = ("OECD link errors as published in PISA 2025 Results (Volume I) and "
          "PISA 2022 Results (Volume I); values reproduced in the NCES PISA 2025 "
          "Technical Notes, Table 4, and PISA 2022 Technical Notes, Table 3")

# Proficiency-level lower bounds (score points), as in the OECD's public
# Stata code (PISA 2025 STU_CommonFiles.do): mathematics 1c/1b/1a/2/3/4/5/6,
# reading 1c/1b/1a/2/3/4/5/6, science 1b/1a/2/3/4/5/6 (no 1c).
LEVELS: dict[str, list[tuple[str, float]]] = {
    "MATH": [("1c", 233.17), ("1b", 295.47), ("1a", 357.77), ("2", 420.07), ("3", 482.38),
             ("4", 544.68), ("5", 606.99), ("6", 669.30)],
    "READ": [("1c", 189.33), ("1b", 262.04), ("1a", 334.75), ("2", 407.47), ("3", 480.18),
             ("4", 552.89), ("5", 625.61), ("6", 698.32)],
    "SCIE": [("1b", 260.54), ("1a", 334.94), ("2", 409.54), ("3", 484.14), ("4", 558.73),
             ("5", 633.33), ("6", 707.93)],
}


def level_name(domain: str, cutoff: str) -> str | None:
    """'Level 2' for a lower bound written as in a plan ('420.07')."""
    try:
        value = float(cutoff)
    except ValueError:
        return None
    for name, bound in LEVELS.get(domain, []):
        if abs(bound - value) < 0.005:
            return f"Level {name}"
    return None


def levels_prompt() -> str:
    lines = []
    for dom, names in (("MATH", "mathematics"), ("READ", "reading"), ("SCIE", "science")):
        lines.append(f"  {names}: " + ", ".join(f"Level {n} >= {b}" for n, b in LEVELS[dom]))
    return "\n".join(lines)


DOMAIN_RE = re.compile(r"\bPV(?:\{pv\}|\d{1,2})(MATH|READ|SCIE)\b")
DOMAIN_NAMES = {"MATH": "mathematics", "READ": "reading", "SCIE": "science"}


def domain_of(measure: str | None) -> str | None:
    """MATH / READ / SCIE when the measure is a plausible-value score."""
    m = DOMAIN_RE.search(str(measure or ""))
    return m.group(1) if m else None


MEAN_SCORE_RE = re.compile(r"^\s*PV(?:\{pv\}|\d{1,2})(MATH|READ|SCIE)\s*$")


def is_mean_score(measure: str | None) -> bool:
    """True only for a plain plausible-value score. The published link errors
    apply to changes in MEAN scores; proficiency-level shares ("% below
    Level 2" = CASE WHEN PV… < 420.07 …) have their own link errors, which
    are not loaded — those changes keep the sampling-only SE."""
    return bool(MEAN_SCORE_RE.match(str(measure or "")))


def link_error(first: str, last: str, measure: str | None) -> float | None:
    """The published link error for a change in the mean of `measure` from
    `first` to `last`, or None when unknown / not a mean achievement score."""
    if not is_mean_score(measure):
        return None
    return LINK_ERRORS.get((str(first), str(last), domain_of(measure)))


def loaded() -> bool:
    return bool(LINK_ERRORS)
