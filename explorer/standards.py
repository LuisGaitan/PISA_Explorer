"""Construct -> standard variable, per cycle.

Two users asking "the gender gap in France" must get the same variable, and
the one the OECD's own reports use. Left to retrieval, the choice depended on
which search terms the router happened to emit that run (PRIVATESCH one time,
SC013Q01TA the next; HOMEPOS for "socio-economic status"). Here the choice is
a data table: when a question matches a construct, the standard variable's
cards are put in front of the planner with the reason, and — for constructs
marked with `swap_from` — a plan that picked a listed look-alike is switched
to the standard deterministically and the switch is stated in the provenance.

Only variables verified to exist in the listed cycles belong here
(tests/test_offline.py checks every entry against the catalog when the data
are present). Add a construct when production shows the planner choosing
inconsistently; do not add speculative ones.
"""

import re
from dataclasses import dataclass, field

CYCLES = ("2018", "2022", "2025")


@dataclass(frozen=True)
class Standard:
    construct: str
    pattern: re.Pattern
    variables: dict                      # cycle -> variable code
    instrument: str = "stu_qqq"
    reason: str = ""
    # plan variables that name the SAME construct and are replaced by the
    # standard (only when the question does not name them itself)
    swap_from: tuple = ()
    # other variables whose cards are shown alongside (related, not standard)
    companions: tuple = ()
    # a question pattern that switches the rule off
    unless: re.Pattern | None = None
    fields: tuple = field(default=("group_col", "variable", "quart_variable", "measure",
                                   "x", "y", "row_var", "col_var"))

    def variable_for(self, cycle: str) -> str | None:
        return self.variables.get(str(cycle))

    def codes(self) -> list[str]:
        return list(dict.fromkeys(list(self.variables.values()) + list(self.companions)))


def _all(code: str) -> dict:
    return {c: code for c in CYCLES}


STANDARDS: list[Standard] = [
    Standard(
        "creative thinking (2022 innovative domain)",
        re.compile(r"\bcreative[- ]thinking\b|\bcreativity (score|scale|test|assessment)\b", re.I),
        {"2022": "PV{pv}CRTH_NC"}, instrument="stu_crt",
        reason=("the creative-thinking plausible values (0-60 'number correct' scale) are in the "
                "2022 creative-thinking cognitive file, analysed through stu_crt (students joined "
                "to it, with their weights); assessed in 2022 only, 64 economies"),
    ),
    Standard(
        "global competence (2018 innovative domain)",
        re.compile(r"\bglobal competenc[ey]\b|\bglobal[- ]mindedness (score|test)\b", re.I),
        {"2018": "PV{pv}GLCM"},
        reason=("the global-competence cognitive test's plausible values PV1-10GLCM are in the "
                "2018 student file for the 27 economies that took it (Scotland as GBR; Moscow "
                "City, Moscow region and Tatarstan as QMC, QMR, QRT); assessed in 2018 only"),
    ),
    Standard(
        "financial literacy",
        re.compile(r"\bfinancial literacy\b|\bfinancial (knowledge|skills) (score|test|assessment)\b", re.I),
        {"2018": "PV{pv}FLIT", "2022": "PV{pv}FLIT"}, instrument="flt_qqq",
        reason=("the financial-literacy plausible values live in the financial-literacy files "
                "(flt_qqq), which carry their own final and replicate weights for the "
                "financial-literacy subsample; assessed in 2018 and 2022, not in 2025"),
    ),
    Standard(
        "gender",
        re.compile(r"\b(gender|girls?|boys?|female|male|sex)\b", re.I),
        {"2018": "ST004D01T", "2022": "ST004D01T", "2025": "MALE"},
        reason=("ST004D01T (1 = Female, 2 = Male) in 2018 and 2022; in 2025 the derived "
                "flag MALE (1 = Male, 0 = Female/Other) is complete for all 90 economies "
                "while ST004D01T is withheld for 14 — MALE is the 2025 convention for "
                "every economy"),
        companions=("ST004D01T",),
    ),
    Standard(
        "public vs private school",
        re.compile(r"\b(public|private|state|government)\b.{0,40}\bschool|"
                   r"\bschool.{0,40}\b(public|private|type)\b", re.I),
        _all("SC013Q01TA"), instrument="sch_qqq",
        reason=("the principal-reported item (1 = public, 2 = private; 82 of 90 economies "
                "in 2025) is the OECD's public/private classification; PRIVATESCH is "
                "derived from sampling frames (57 economies in 2025) and SCHLTYPE splits "
                "private into government-dependent / independent"),
        swap_from=("PRIVATESCH",),
        unless=re.compile(r"government[- ]dependent|independent private", re.I),
    ),
    Standard(
        "school location (rural / urban)",
        re.compile(r"\b(rural|urban|village|city|cities|town|metropolitan)\b", re.I),
        _all("SC001Q01TA"), instrument="sch_qqq",
        reason=("community size as reported by the principal: 1 village (<3,000), 2 small "
                "town, 3 town, 4 city (100,000-1M), 5 large city, 6 megacity (2022/2025); "
                "rural = codes 1-2, city = codes 4 and above"),
    ),
    Standard(
        "socio-economic status",
        re.compile(r"socio-?economic|\bescs\b|\bses\b|economic status|social background|"
                   r"\b(disadvantaged|advantaged) students\b", re.I),
        _all("ESCS"),
        reason=("the OECD index of economic, social and cultural status; HOMEPOS, HISEI "
                "and parental education are its components, not the index"),
        swap_from=("HOMEPOS", "HISEI", "PARED", "PAREDINT"),
    ),
    Standard(
        "immigrant background",
        re.compile(r"immigra|migrant|native[- ]born|foreign[- ]born", re.I),
        _all("IMMIG"),
        reason=("1 = native, 2 = second-generation, 3 = first-generation; immigrant "
                "= 2 or 3 in the OECD's reports"),
    ),
    Standard(
        "grade repetition",
        re.compile(r"repeat(ed|ing)? (a |the )?(grade|year|class)|grade repetition|repeaters?", re.I),
        _all("REPEAT"),
        reason="1 = repeated a grade at least once, 0 = never (derived from ST127)",
    ),
    Standard(
        "sense of belonging",
        re.compile(r"belong", re.I),
        _all("BELONG"),
        reason="the OECD sense-of-belonging index (WLE), available in every cycle",
    ),
    Standard(
        "life satisfaction",
        re.compile(r"life satisfaction|satisfied with (their|your|his|her )?li(fe|ves)|"
                   r"satisfaction with life", re.I),
        _all("ST016Q01NA"),
        reason=("the 0-10 life-satisfaction item is the OECD's headline indicator in every "
                "cycle (the 2022 LIFESAT index covers several domains and exists only in 2022)"),
        companions=("LIFESAT",),
    ),
    Standard(
        "bullying",
        re.compile(r"bull(y|ied|ying|ies)", re.I),
        {"2018": "BEINGBULLIED", "2022": "BULLIED", "2025": "BULLIED"},
        reason=("the exposure-to-bullying index (WLE); renamed between 2018 and 2022 and "
                "standardized within each cycle"),
    ),
    Standard(
        "mathematics anxiety",
        re.compile(r"math\w*[- ]anxiety|anxious about math|anxiety (in|about|towards) math", re.I),
        {"2022": "ANXMAT"},
        reason="the mathematics-anxiety index (WLE), collected in 2022 only",
    ),
    Standard(
        "curiosity",
        re.compile(r"curio(us|sity)", re.I),
        {"2022": "CURIOAGR", "2025": "CURIO"},
        reason="different curiosity scales in 2022 and 2025 — levels only, no cross-cycle change",
    ),
    Standard(
        "AI use",
        re.compile(r"\b(ai|a\.i\.|artificial intelligence|chatgpt|chatbots?|generative ai)\b", re.I),
        {"2025": "ST438Q01DA"},
        reason=("ST438Q01DA-Q04DA: how often students use AI chatbots for schoolwork "
                "(1 never ... 5 every day or almost; 84 of 90 economies, 2025 only); "
                "AIUSESCH is the AI-use-at-school index (WLE)"),
        companions=("ST438Q02DA", "ST438Q03DA", "ST438Q04DA", "AIUSESCH"),
    ),
    Standard(
        "language spoken at home",
        re.compile(r"language (spoken )?at home|home language|speak\w* .{0,20}at home", re.I),
        _all("ST022Q01TA"),
        reason="1 = language of the test, 2 = another language",
    ),
    Standard(
        "skipping school / truancy",
        re.compile(r"skip\w* (school|class)|truan|absentee", re.I),
        {"2022": "SKIPPING", "2025": "SKIPPING"},
        reason="derived index of skipped days and classes (2022, 2025); 2018 has the items ST062",
        companions=("ST062Q01TA",),
    ),
]


def matching(question: str) -> list[Standard]:
    text = question or ""
    return [s for s in STANDARDS
            if s.pattern.search(text) and not (s.unless and s.unless.search(text))]


def prompt_block(matches: list[Standard], cycles_present: dict | None = None) -> str:
    """Lines for the planner prompt naming the standard variable per cycle."""
    if not matches:
        return ""
    lines = ["STANDARD VARIABLES FOR THIS QUESTION (use these unless the user names "
             "another variable explicitly):"]
    for s in matches:
        per = ", ".join(f"{c}: {v}" for c, v in sorted(s.variables.items()))
        lines.append(f"- {s.construct} => {per} ({s.instrument}) — {s.reason}.")
    return "\n".join(lines) + "\n"


def named_in(question: str, codes) -> set[str]:
    """Variable codes the question spells out itself (in capitals)."""
    caps = " " + re.sub(r"[^A-Za-z0-9_ ]", " ", question or "") + " "
    return {c for c in codes if re.search(rf"\b{re.escape(c)}\b", caps)}
