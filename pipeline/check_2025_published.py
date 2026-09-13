"""Reproduce PISA 2025 Technical Report Annex 14 (Tables 14.A.11-13) from the
converted data: per-economy sample size, weighted population estimate, and
the standard error of the mean science / reading / mathematics score.

This is the strongest available check of the whole chain (conversion ->
DuckDB views -> PV x Fay-BRR estimator) for the new cycle, because the SEs
depend on all 80 replicate weights and all 10 plausible values being intact.
(The annex does not print the means themselves; those are validated against
the published country tables separately.)

Usage: python pipeline/check_2025_published.py [--tolerance 0.02]
Exit code 0 when every economy matches within tolerance.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from explorer.analysis import weighted_mean          # noqa: E402
from explorer.db import connect                      # noqa: E402
from pipeline.sources import DATA_2025               # noqa: E402

ANNEX = DATA_2025 / "Excel Files" / "14-PISA-2025-Technical Report-AnnexTables.xlsx"
SHEETS = {"SCIE": "T14.A.11", "READ": "T14.A.12", "MATH": "T14.A.13"}

# Annex country names -> CNT codes used in the PUF (only the non-obvious).
NAME_TO_CNT = {
    "Albania": "ALB", "Argentina": "ARG", "Armenia": "ARM", "Australia": "AUS",
    "Austria": "AUT", "Azerbaijan": "AZE", "Belgium": "BEL", "Brazil": "BRA",
    "Brunei Darussalam": "BRN", "Bulgaria": "BGR", "Cambodia": "KHM",
    "Canada": "CAN", "Chile": "CHL", "Chinese Taipei": "TAP", "Colombia": "COL",
    "Costa Rica": "CRI", "Croatia": "HRV", "Czechia": "CZE", "Denmark": "DNK",
    "Dominican Republic": "DOM", "Dushanbe (Tajikistan)": "QTJ", "Ecuador": "ECU",
    "El Salvador": "SLV", "Estonia": "EST", "Finland": "FIN", "France": "FRA",
    "Georgia": "GEO", "Germany": "DEU", "Greece": "GRC", "Guatemala": "GTM",
    "Hong Kong (China)": "HKG", "Hungary": "HUN", "Iceland": "ISL",
    "Indonesia": "IDN", "Ireland": "IRL", "Israel": "ISR", "Italy": "ITA",
    "Japan": "JPN", "Jordan": "JOR", "Kazakhstan": "KAZ", "Kenya": "KEN",
    "Korea": "KOR", "Kosovo": "KSV", "Kurdistan Region (Iraq)": "QKI",
    "Kyrgyzstan": "KGZ", "Latvia": "LVA", "Lebanon": "LBN", "Lithuania": "LTU",
    "Luxembourg": "LUX", "Macao (China)": "MAC", "Malaysia": "MYS", "Malta": "MLT",
    "Mauritius": "MUS", "Mexico": "MEX", "Moldova": "MDA", "Mongolia": "MNG",
    "Montenegro": "MNE", "Morocco": "MAR", "Netherlands": "NLD",
    "New Zealand": "NZL", "North Macedonia": "MKD", "Norway": "NOR",
    "Palestinian Authority": "PSE", "Paraguay": "PRY", "Peru": "PER",
    "Philippines": "PHL", "Poland": "POL", "Portugal": "PRT", "Qatar": "QAT",
    "Romania": "ROU", "Rwanda": "RWA", "Saudi Arabia": "SAU", "Serbia": "SRB",
    "Singapore": "SGP", "Slovak Republic": "SVK", "Slovenia": "SVN", "Spain": "ESP",
    "Sweden": "SWE", "Switzerland": "CHE", "Thailand": "THA", "Türkiye": "TUR",
    "Ukrainian regions (17 of 27)": "QUA", "United Arab Emirates": "ARE",
    "United Kingdom": "GBR", "United States": "USA", "Uruguay": "URY",
    "Uzbekistan": "UZB", "Viet Nam": "VNM", "Zambia": "ZMB",
    "B-S-J-Z (China)": "QCI", "Dubai (UAE)": "QAZ",
}


def _annex_rows(sheet: str, columns: dict[int, str]) -> pd.DataFrame:
    raw = pd.read_excel(ANNEX, sheet_name=sheet, header=None)
    header_row = raw.index[raw[0].astype(str).str.strip().str.lower()
                           == "country/economy"][0]
    df = raw.iloc[header_row + 1:, list(columns)].copy()
    df.columns = list(columns.values())
    first_numeric = list(columns.values())[1]
    df = df[pd.to_numeric(df[first_numeric], errors="coerce").notna()]
    # footnote digits glued to names ("Albania1") and stray asterisks
    df["name"] = df["name"].astype(str).str.replace(r"[\d\*]+$", "", regex=True).str.strip()
    df["CNT"] = df["name"].map(NAME_TO_CNT)
    return df


def load_annex(sheet: str) -> pd.DataFrame:
    """Table 14.A.11-13: sample size and SE of the mean. (Their 'Population
    Estimate' column is the estimated number of 15-year-olds, not the weighted
    participants — the latter comes from Table 14.A.1.)"""
    return _annex_rows(sheet, {0: "name", 1: "n", 3: "se"})


def load_weighted_participants() -> pd.DataFrame:
    """Table 14.A.1 column 'Weighted number of participating students'."""
    df = _annex_rows("T14.A.1", {0: "name", 8: "n_a1", 9: "population"})
    return df[["CNT", "population"]].dropna(subset=["CNT"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="max |SE_ours - SE_published| in score points")
    args = parser.parse_args()

    con = connect(read_only=True)
    counts = con.sql("SELECT CNT, count(*) AS n_ours, sum(W_FSTUWT) AS population_ours "
                     "FROM stu_qqq_2025 GROUP BY CNT").df()
    weighted = load_weighted_participants().rename(columns={"population": "population_pub"})
    failures = 0
    for domain, sheet in SHEETS.items():
        annex = load_annex(sheet)
        # Cyprus appears in the annex but is not part of the OECD PUF.
        unmapped = sorted(set(annex[annex.CNT.isna()].name) - {"Cyprus"})
        if unmapped:
            print(f"{domain}: unmapped annex names (add to NAME_TO_CNT): {unmapped}")
        ours = (weighted_mean(con, "stu_qqq_2025", f"PV{{pv}}{domain}", by=("CNT",))
                .rename(columns={"se": "se_ours"}))
        merged = (annex.dropna(subset=["CNT"])
                  .rename(columns={"n": "n_pub", "se": "se_pub"})
                  .merge(ours, on="CNT", how="left")
                  .merge(counts, on="CNT", how="left")
                  .merge(weighted, on="CNT", how="left"))
        merged["se_pub"] = merged.se_pub.astype(float)
        merged["population_pub"] = merged.population_pub.astype(float)
        merged["se_diff"] = (merged.se_ours - merged.se_pub).abs()
        # The annex tables were produced from an earlier database release
        # (IDB2.1, see Annex 25.A); the PUF differs by at most one student
        # (Belgium, Argentina) and, for Israel — whose data were amended
        # after that release — by 0.13% of the weighted count.
        merged["n_ok"] = (merged.n_pub.astype(int) - merged.n_ours).abs() <= 1
        merged["pop_ok"] = ((merged.population_ours - merged.population_pub).abs()
                            <= 5e-3 * merged.population_pub)
        merged["se_ok"] = merged.se_diff <= args.tolerance
        # A domain absent from the PUF for an economy (no PVs at all) is a
        # documented data gap, not a pipeline failure — reported separately.
        merged["no_pvs"] = merged.estimate.isna()
        ok = merged.n_ok & merged.pop_ok & (merged.se_ok | merged.no_pvs)
        bad = merged[~ok]
        print(f"{domain}: {int(ok.sum())}/{len(merged)} economies match "
              f"(sample size ±1, weighted participants ±0.5%, SE within "
              f"{args.tolerance}); max SE diff {merged.se_diff.max():.4f}")
        for _, r in merged[merged.no_pvs].iterrows():
            print(f"   NOTE {r.CNT} ({r['name']}): no {domain} plausible values in the "
                  f"PUF (published SE {r.se_pub:.3f})")
        for _, r in bad.iterrows():
            print(f"   MISMATCH {r.CNT} ({r['name']}): n {r.n_pub} vs {r.n_ours}, "
                  f"weighted {r.population_pub:,.1f} vs {r.population_ours:,.1f}, "
                  f"SE {r.se_pub:.3f} vs {r.se_ours:.3f}, mean {r.estimate:.1f}")
        failures += len(bad)
    con.close()
    print("-" * 50)
    print("ALL PUBLISHED FIGURES REPRODUCED" if failures == 0
          else f"{failures} ECONOMY/DOMAIN MISMATCHES")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
