"""Registry of PISA source files: where each SAS file lives, what it is, and
the official row counts used for validation.

The raw OECD databases are read-only sources and never modified:
  2018 (CY07MSU): C:\\Users\\Luis\\Desktop\\DataMining\\PISA_Data2018
  2022 (CY08MSP): C:\\Users\\Luis\\Desktop\\DataMining\\PISA_Data2022

`expected_rows=None` means no official count is on record; the pipeline then
validates against the row count stored in the SAS file header itself
(meta.number_rows), which it also cross-checks when an official count exists.
"""

from dataclasses import dataclass
from pathlib import Path

DATA_2018 = Path(r"C:\Users\Luis\Desktop\DataMining\PISA_Data2018")
DATA_2022 = Path(r"C:\Users\Luis\Desktop\DataMining\PISA_Data2022")

# Repo-local output area (gitignored, fully rebuildable from sources)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PARQUET_DIR = DATA_DIR / "parquet"
METADATA_DIR = DATA_DIR / "metadata"
DB_PATH = DATA_DIR / "pisa.duckdb"

ESCS_TREND_CSV = DATA_2022 / "ESCS_Trend" / "escs_trend.csv"

# Columns that must survive conversion in the student questionnaire files
STU_QQQ_KEY_COLUMNS = (
    ["CNT", "CNTSCHID", "CNTSTUID", "W_FSTUWT", "ESCS", "STRATUM"]
    + [f"PV{i}{d}" for i in range(1, 11) for d in ("MATH", "READ", "SCIE")]
    + [f"W_FSTURWT{i}" for i in range(1, 81)]
)

EXPECTED_ECONOMIES_STU_QQQ = 80


@dataclass(frozen=True)
class Source:
    cycle: str        # "2018" | "2022"
    instrument: str   # e.g. "stu_qqq"
    sas_path: Path
    expected_rows: int | None = None  # official public-use count, if known

    @property
    def name(self) -> str:
        return f"{self.instrument}_{self.cycle}"

    @property
    def parquet_path(self) -> Path:
        return PARQUET_DIR / f"pisa{self.cycle}" / f"{self.instrument}.parquet"

    @property
    def metadata_path(self) -> Path:
        return METADATA_DIR / f"{self.name}.json"


SOURCES: list[Source] = [
    # ---- PISA 2018 (CY07MSU) ----
    Source("2018", "stu_qqq", DATA_2018 / "STU" / "cy07_msu_stu_qqq.sas7bdat", 612_004),
    Source("2018", "sch_qqq", DATA_2018 / "SCH" / "cy07_msu_sch_qqq.sas7bdat", 21_903),
    Source("2018", "tch_qqq", DATA_2018 / "TCH" / "cy07_msu_tch_qqq.sas7bdat", 107_367),
    Source("2018", "stu_cog", DATA_2018 / "COG" / "cy07_msu_stu_cog.sas7bdat"),
    Source("2018", "stu_tim", DATA_2018 / "TIM" / "cy07_msu_stu_tim.sas7bdat"),
    Source("2018", "stu_ttm", DATA_2018 / "TTM" / "CY07_MSU_STU_TTM.SAS7BDAT"),
    Source("2018", "flt_qqq", DATA_2018 / "FLT" / "cy07_msu_flt_qqq.sas7bdat"),
    Source("2018", "flt_cog", DATA_2018 / "FLT" / "cy07_msu_flt_cog.sas7bdat"),
    Source("2018", "flt_tim", DATA_2018 / "FLT" / "cy07_msu_flt_tim.sas7bdat"),
    Source("2018", "flt_ttm", DATA_2018 / "FLT" / "CY07_MSU_FLT_TTM.SAS7BDAT"),
    # ---- PISA 2022 (CY08MSP) ----
    Source("2022", "stu_qqq", DATA_2022 / "STU_QQQ_SAS" / "CY08MSP_STU_QQQ.SAS7BDAT", 613_744),
    Source("2022", "sch_qqq", DATA_2022 / "SCH_QQQ_SAS" / "CY08MSP_SCH_QQQ.SAS7BDAT", 21_629),
    Source("2022", "tch_qqq", DATA_2022 / "TCH_QQQ_SAS" / "CY08MSP_TCH_QQQ.SAS7BDAT", 68_054),
    Source("2022", "stu_cog", DATA_2022 / "STU_COG_SAS" / "CY08MSP_STU_COG.SAS7BDAT", 613_744),
    Source("2022", "stu_tim", DATA_2022 / "STU_TIM_SAS" / "CY08MSP_STU_TIM.SAS7BDAT", 613_744),
    Source("2022", "flt_qqq", DATA_2022 / "FLT_SAS" / "CY08MSP_FLT_QQQ.SAS7BDAT", 97_983),
    Source("2022", "flt_cog", DATA_2022 / "FLT_SAS" / "CY08MSP_FLT_COG.SAS7BDAT", 97_983),
    Source("2022", "flt_tim", DATA_2022 / "FLT_SAS" / "CY08MSP_FLT_TIM.SAS7BDAT", 97_983),
    Source("2022", "crt_cog", DATA_2022 / "CRT_SAS" / "CY08MSP_CRT_COG.SAS7BDAT", 499_843),
]


def get_sources(only: list[str] | None = None) -> list[Source]:
    """Filter sources by name (e.g. 'stu_qqq_2018'), instrument, or cycle."""
    if not only:
        return list(SOURCES)
    keys = {k.lower() for k in only}
    return [
        s for s in SOURCES
        if s.name in keys or s.instrument in keys or s.cycle in keys
    ]
