"""Chat with the PISA database from the terminal.

  python -m explorer.chat                      # interactive session
  python -m explorer.chat "your question"      # one-shot

Every answer shows its full provenance: the tables and variables it came
from (with codebook labels), the filter, the student counts behind the
numbers, and the exact statistical method. Commands in the session:

  /export <file.csv>   save the last result table
  /plan                show the last analysis plan (what the LLM filled in)
  /prov                show full provenance again
  /quit                exit
"""

import sys

from .agent import Agent, AgentResult


def print_result(result: AgentResult) -> None:
    print(f"\n{result.answer}\n")
    if result.table is not None and not result.table.empty:
        print(result.table.round(2).to_string(index=False))
        print()
    if result.provenance:
        print_provenance(result.provenance)


def print_provenance(prov: dict) -> None:
    print("--- where this comes from " + "-" * 34)
    print(f"  source:  {prov['source']}")
    print(f"  tables:  {', '.join(prov['tables'])}")
    for v in prov["variables"]:
        print(f"    {v['variable']} ({v['table']}): {v['label']}")
    print(f"  filter:  {prov['filter']}")
    for s in prov["sample"]:
        if "students" in s:
            weighted = (f", representing ~{s['weighted_students']:,} in the population"
                        if s.get("weighted_students") else "")
            print(f"  sample:  {s['table']}: {s['students']:,} students{weighted}")
    print(f"  method:  {prov['method']}")
    if prov.get("sql"):
        print(f"  sql:     {prov['sql']}")
    for note in prov["notes"]:
        print(f"  NOTE:    {note}")
    print("-" * 60)


def repl() -> None:
    agent = Agent()
    last: AgentResult | None = None
    history: list[dict] = []
    print("PISA Explorer — ask about PISA 2018/2022/2025. /quit to exit.")
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.lower() in ("/quit", "/exit", "quit", "exit"):
            break
        if line.startswith("/export"):
            parts = line.split(maxsplit=1)
            if last is None or last.table is None:
                print("nothing to export yet")
            elif len(parts) < 2:
                print("usage: /export <file.csv>")
            else:
                last.table.to_csv(parts[1], index=False)
                print(f"saved {len(last.table)} rows to {parts[1]}")
            continue
        if line == "/plan":
            print(last.plan if last and last.plan else "no plan yet")
            continue
        if line == "/prov":
            if last and last.provenance:
                print_provenance(last.provenance)
            else:
                print("no provenance yet")
            continue
        try:
            last = agent.ask(line, history=history)
        except Exception as e:
            print(f"error: {e}")
            continue
        history.append({"question": line, "answer": last.answer,
                        "explanation": (last.plan or {}).get("explanation")})
        del history[:-6]
        print_result(last)


def main() -> None:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = Agent().ask(question)
        print_result(result)
    else:
        repl()


if __name__ == "__main__":
    main()
