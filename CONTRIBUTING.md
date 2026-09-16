# Contributing

Thanks for your interest. PISA Explorer has two audiences: people who ask
questions on the hosted site, and researchers who want to inspect or extend
the survey-statistics machinery. Contributions to either are welcome.

## Ground rules

- **Methodology first.** Every estimate must use the final student weight,
  average over the plausible values, and carry a Fay-BRR standard error.
  A new analysis template belongs in `explorer/analysis.py` and must be
  checked against a published OECD figure before it is merged — say which
  table you reproduced in the pull request.
- **The LLM plans, it never computes.** Keep statistics out of prompts and
  inside the validated templates. If the planner needs a new capability,
  add a template and expose its parameters in `PLAN_SYSTEM`.
- **No raw data in the repository.** The OECD public-use files stay on your
  machine; only code, docs, and small fixtures are committed. `data/` is
  ignored.
- **Provenance is part of the feature.** A result the user cannot trace to
  tables, variables, filter and method is not finished.

## Setting up

```
pip install -r requirements.txt
```

Download the OECD public-use files (see README "Raw data") into one folder
and point `PISA_RAW_ROOT` at it (environment variable or a line in `.env`).
Then `python pipeline/convert.py`, `build_db.py`, `build_catalog.py`,
`validate.py`. The statistics layer needs no API key; the chat layer needs
`GEMINI_API_KEY`.

## Tests

`pytest tests/` runs the offline checks (no data, no API key). With the data
converted, `pytest tests/test_golden_plans.py` replays the golden plan set
(exact numbers the engine must reproduce), and `python pipeline/validate.py`
plus `python pipeline/check_2025_published.py` validate the data itself.
With an API key, `python scripts/golden_live.py` runs the golden questions
through the model and `python scripts/replay_events.py` re-asks recorded
production questions.

Before a deploy, run all of them. If a change alters a computed number on
purpose (a new estimator rule, a new comparability rule), bump
`METHOD_VERSION` in `explorer/version.py`, re-record the golden plans with
`python scripts/golden_record.py --update`, and say why in the commit.

## Pull requests

- One change per pull request, with a sentence on what it does and how you
  verified it.
- Match the existing style: plain Python, standard library where possible,
  short functions with docstrings that say *why*.
- Do not change the methodology constants (Fay k, replicate count, PV count,
  proficiency cutoffs) without citing the OECD Technical Report section.
