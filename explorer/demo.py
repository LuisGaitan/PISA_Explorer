"""End-to-end demo of the foundation: catalog retrieval + survey-correct
statistics answering a real trend question.

  Question: "Did the gender gap in math change between 2018, 2022 and 2025
             in Korea, the US, Finland, and Brazil?"

Run:  python -m explorer.demo
"""

import time

from . import catalog
from .analysis import gap, trend, weighted_mean
from .db import connect

COUNTRIES = ("KOR", "USA", "FIN", "BRA")
WHERE = "CNT IN ('KOR','USA','FIN','BRA')"
CYCLES = ("2018", "2022", "2025")


def main() -> None:
    t0 = time.time()

    print("=" * 70)
    print("1. CATALOG RETRIEVAL — what the AI layer would fetch, not prompt-stuff")
    print("=" * 70)
    print("\nsearch('gender', instrument='stu_qqq'):")
    print(catalog.search("gender", instrument="stu_qqq", limit=4).to_string(index=False))
    print("\ndescribe('ST004D01T'):")
    print(catalog.describe("ST004D01T").to_string(index=False))
    print("\nsearch('mathematics anxiety') — a 2022-only construct:")
    print(catalog.search("mathematics anxiety", instrument="stu_qqq", limit=4)
          .to_string(index=False))

    con = connect(read_only=True)

    print()
    print("=" * 70)
    print("2. WEIGHTED MEAN MATH (10 PVs averaged, Fay-BRR SE over 80 replicates)")
    print("=" * 70)
    for cycle in CYCLES:
        res = weighted_mean(
            con, f"stu_qqq_{cycle}", "PV{pv}MATH", by=("CNT",), where=WHERE
        ).sort_values("CNT")
        print(f"\n  PISA {cycle}:")
        for _, r in res.iterrows():
            print(f"    {r.CNT}: {r.estimate:6.1f}  (SE {r.se:.2f})")

    print()
    print("=" * 70)
    print("3. GENDER GAP IN MATH (boys - girls), replicate-wise SE")
    print("=" * 70)
    gaps = {}
    for cycle in CYCLES:
        # ST004D01T (1=female, 2=male) is populated for these four economies in
        # every cycle; 14 economies release only the derived MALE flag in 2025.
        gaps[cycle] = gap(
            con, f"stu_qqq_{cycle}", "PV{pv}MATH",
            group_col="ST004D01T", minuend=2, subtrahend=1,   # 2=Male, 1=Female
            by=("CNT",), where=WHERE,
        ).sort_values("CNT")
        print(f"\n  PISA {cycle} (positive = boys ahead):")
        for _, r in gaps[cycle].iterrows():
            print(f"    {r.CNT}: {r.estimate:+6.1f}  (SE {r.se:.2f})")

    print()
    print("=" * 70)
    print("4. TREND: did the gap change 2018 -> 2022 -> 2025?")
    print("=" * 70)
    tr = trend({c: gaps[c] for c in CYCLES}, by=("CNT",))
    print("\n  (change = gap_2025 - gap_2018; |change| > 2*SE ~= significant;")
    print("   link error not yet included — see analysis.trend docstring)")
    for _, r in tr.iterrows():
        sig = "significant" if abs(r.change) > 2 * r.se_change else "not significant"
        print(f"    {r.CNT}: gap {r.estimate_2018:+5.1f} -> {r.estimate_2022:+5.1f} "
              f"-> {r.estimate_2025:+5.1f}  "
              f"change {r.change:+5.1f} (SE {r.se_change:.2f})  [{sig}]")

    con.close()
    print(f"\ntotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
