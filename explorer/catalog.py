"""Keyword retrieval over the variable catalog.

This is the layer that keeps PISA's ~23,000 variables OUT of any LLM prompt:
a question is turned into a handful of search terms, and only the top-scoring
variables (name, label, table, value labels) are ever surfaced.

    from explorer.catalog import search
    search("gender", cycle="2022")
    search("mathematics anxiety", instrument="stu_qqq")
    search("immigrant background", limit=10)

Scoring is deliberately simple and transparent: exact variable-name match
beats name substring, which beats whole-word label matches, which beat label
substrings. Semantic/embedding search can replace this later behind the same
function signature.
"""

import functools
import re

import pandas as pd

from .db import CATALOG_DIR


# Query words that carry no topical signal (they match thousands of labels).
STOPWORDS = {
    "the", "and", "for", "with", "how", "what", "which", "who", "are", "is",
    "of", "in", "on", "to", "by", "about", "data", "dataset", "variable",
    "variables", "well", "very", "much", "many", "any", "all", "from", "this",
    "that", "these", "those", "you", "your", "their", "own", "do", "does",
    "did", "have", "has", "at", "as", "or", "an", "a", "be", "being",
}


# Words that name the respondent type rather than a construct.
ENTITY_WORDS = {"teacher", "teachers", "student", "students", "school", "schools",
                "parent", "parents", "country", "countries"}


@functools.lru_cache(maxsize=1)
def _load() -> pd.DataFrame:
    return _prepare(pd.read_parquet(CATALOG_DIR / "variables.parquet"))


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Derived lower-case / compact columns the scorer matches against."""
    df = df.copy()
    df["_name_lower"] = df.variable.str.lower()
    df["_label_lower"] = df.label.str.lower()
    df["_label_compact"] = df._label_lower.str.replace(r"[^a-z0-9]", "", regex=True)
    return df


def search(
    query: str,
    cycle: str | None = None,
    instrument: str | None = None,
    limit: int = 15,
) -> pd.DataFrame:
    """Return the catalog rows of the top `limit` VARIABLES matching `query`
    (all tables/cycles in which each selected variable exists)."""
    df = _load()
    if cycle:
        df = df[df.cycle == str(cycle)]
    if instrument:
        df = df[df.instrument == instrument.lower()]

    tokens = [t for t in re.split(r"\W+", query.lower())
              if len(t) >= 2 and t not in STOPWORDS]
    # Whole-phrase match, insensitive to hyphens/spaces ("well-being" ==
    # "wellbeing" == "well being"), outranks any pile of single-token hits —
    # otherwise "well-being" is won by every label containing "well". It also
    # rescues phrases made only of stopwords (well-being is exactly that).
    compact = re.sub(r"[^a-z0-9]", "", query.lower())
    phrase = compact if len(compact) >= 5 else ""
    if not tokens and not phrase:
        return df.head(0)[["variable", "table_name", "label", "n_value_labels"]]

    score = pd.Series(0.0, index=df.index)
    if phrase:
        score += df._label_compact.str.contains(phrase, regex=False) * 25.0
    for token in tokens:
        score += (df._name_lower == token) * 100.0
        score += df._name_lower.str.contains(token, regex=False) * 30.0
        score += df._label_lower.str.contains(rf"\b{re.escape(token)}\b", regex=True) * 8.0
        score += df._label_lower.str.contains(token, regex=False) * 3.0
        if len(token) >= 6:   # crude stem: satisfaction ~ satisfied, bullying ~ bullied
            score += df._label_lower.str.contains(rf"\b{re.escape(token[:4])}", regex=True) * 4.0
            # derived indices abbreviate the concept in their NAME (BULLIED,
            # BELONG, ANXMAT): a name starting with the stem is a strong hit —
            # except for PISA's entity words, which prefix hundreds of names
            if token not in ENTITY_WORDS:
                score += df._name_lower.str.startswith(token[:4]) * 30.0

    hits = df.assign(score=score)[score > 0]
    # Rank VARIABLES, not rows: a variable that exists in several cycles and
    # instruments (e.g. REPEAT in five tables) must not crowd out the others,
    # and every cycle's row of a selected variable must come back together —
    # the planner needs to see that a variable exists in 2025 as well as 2022.
    best = hits.groupby("variable")["score"].max()
    top = best.sort_values(ascending=False).head(limit)
    hits = hits[hits.variable.isin(top.index)]
    hits = hits.assign(score=hits.variable.map(top))
    hits = hits.sort_values(["score", "variable", "table_name"],
                            ascending=[False, True, True])
    return hits[["variable", "table_name", "cycle", "label", "n_value_labels", "score"]]


def describe(variable: str, cycle: str | None = None) -> pd.DataFrame:
    """Full catalog rows (including value labels) for one variable name."""
    df = _load()
    hits = df[df.variable.str.upper() == variable.upper()]
    if cycle:
        hits = hits[hits.cycle == str(cycle)]
    return hits[["variable", "table_name", "cycle", "label", "var_type", "value_labels"]]


def comparability(variable: str) -> pd.DataFrame:
    """Cross-cycle availability of a variable across shared instruments."""
    df = pd.read_parquet(CATALOG_DIR / "comparability.parquet")
    return df[df.variable.str.upper() == variable.upper()]
