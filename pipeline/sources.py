"""Registry of PISA source files: where each raw file lives, what it is, and
the official row counts used for validation.

The raw OECD databases are read-only sources and never modified. They live
outside the repository, under one folder named by PISA_RAW_ROOT (environment
variable, or a line in the gitignored .env at the repo root; default: ./raw):
  <PISA_RAW_ROOT>/PISA_Data2018   2018 (CY07MSU), the SAS release
  <PISA_RAW_ROOT>/PISA_Data2022   2022 (CY08MSP), the SAS release
  <PISA_RAW_ROOT>/PISA_Data2025   2025 (CY09MS),  the SPSS release
Per-cycle overrides: PISA_RAW_2018 / PISA_RAW_2022 / PISA_RAW_2025.
Download the public-use files from the OECD PISA data pages and keep the
OECD folder layout used below.

PISA 2025 is distributed in SPSS format only (one .sav per file), so each
source carries its file format and, where the file's declared encoding is
wrong, an explicit encoding override. The 2025 file names map onto the
instrument names used since 2018 by CONTENT, not by OECD file name:

  CY09_MS_STU_PUF          -> stu_qqq   student questionnaire + PVs + weights
                                        (also carries the ICT and parent items)
  CY09_MS_SCH_PUF          -> sch_qqq   school questionnaire
  CY09_MS_TCH_PUF          -> tch_qqq   teacher questionnaire
  CY09_MS_COG_*            -> stu_cog   cognitive item responses (scored)
  CY09_MS_STU_TT_PUF       -> stu_tim   questionnaire timing (ST001_TT ...),
                                        same content as stu_tim_2018/2022
  CY09_MS_COG_PROCESS_*    -> stu_ttm   cognitive item process data (total
                                        time, actions, time to first action),
                                        the successor of stu_ttm_2018
  CY09_MS_LDW_*            -> ldw_cog   Learning in the Digital World item
                                        responses + process data (2025's
                                        innovative domain; like crt_cog_2022)

Not yet released by the OECD (expected 2027): the Foreign Language Assessment
files. Not downloaded: the LDW self-regulated-learning file.

`expected_rows=None` means no official count is on record; the pipeline then
validates against the row count stored in the file header itself
(meta.number_rows), which it also cross-checks when an official count exists.
"""

import os
from dataclasses import dataclass
from pathlib import Path

CYCLES = ("2018", "2022", "2025")

# Repo-local output area (gitignored, fully rebuildable from sources)
REPO_ROOT = Path(__file__).resolve().parent.parent


def _dotenv(name: str) -> str | None:
    """A single KEY=value from the gitignored .env (no dependency on
    python-dotenv); the process environment wins when both are set."""
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _setting(name: str, default: str) -> str:
    return os.environ.get(name) or _dotenv(name) or default


RAW_ROOT = Path(_setting("PISA_RAW_ROOT", str(REPO_ROOT / "raw")))
DATA_2018 = Path(_setting("PISA_RAW_2018", str(RAW_ROOT / "PISA_Data2018")))
DATA_2022 = Path(_setting("PISA_RAW_2022", str(RAW_ROOT / "PISA_Data2022")))
DATA_2025 = Path(_setting("PISA_RAW_2025", str(RAW_ROOT / "PISA_Data2025")))
DATA_DIR = REPO_ROOT / "data"
PARQUET_DIR = DATA_DIR / "parquet"
METADATA_DIR = DATA_DIR / "metadata"
DB_PATH = DATA_DIR / "pisa.duckdb"

# Prefer the repo-local copy (present in deployment containers, where the
# raw Desktop folders don't exist); fall back to the raw source.
_ESCS_LOCAL = DATA_DIR / "escs_trend.csv"
ESCS_TREND_CSV = _ESCS_LOCAL if _ESCS_LOCAL.exists() else (
    DATA_2022 / "ESCS_Trend" / "escs_trend.csv")

# Columns that must survive conversion in the student questionnaire files
# (identical names in all three cycles).
STU_QQQ_KEY_COLUMNS = (
    ["CNT", "CNTSCHID", "CNTSTUID", "W_FSTUWT", "ESCS", "STRATUM"]
    + [f"PV{i}{d}" for i in range(1, 11) for d in ("MATH", "READ", "SCIE")]
    + [f"W_FSTURWT{i}" for i in range(1, 81)]
)

# Distinct CNT codes in each cycle's student questionnaire file.
EXPECTED_ECONOMIES_STU_QQQ = {"2018": 80, "2022": 80, "2025": 90}


@dataclass(frozen=True)
class Source:
    cycle: str        # "2018" | "2022" | "2025"
    instrument: str   # e.g. "stu_qqq"
    sas_path: Path    # the raw file (SAS7BDAT or SAV — see `fmt`)
    expected_rows: int | None = None  # official public-use count, if known
    fmt: str = "sas"                  # "sas" | "sav"
    encoding: str | None = None       # override when the file's own is wrong

    @property
    def name(self) -> str:
        return f"{self.instrument}_{self.cycle}"

    @property
    def path(self) -> Path:
        return self.sas_path

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
    # ---- PISA 2025 (CY09MS, SPSS public-use files, database of 2026-08-06) ----
    # Row counts are the PUF header counts; the student file's per-country
    # sizes match Table 14.A.1/14.A.11 of the 2025 Technical Report.
    Source("2025", "stu_qqq", DATA_2025 / "CY09_MS_STU_PUF.sav", 755_721, fmt="sav"),
    Source("2025", "sch_qqq", DATA_2025 / "CY09_MS_SCH_PUF.sav", 25_133, fmt="sav"),
    Source("2025", "tch_qqq", DATA_2025 / "CY09_MS_TCH_PUF.sav", 25_776, fmt="sav"),
    Source("2025", "stu_cog", DATA_2025 / "CY09_MS_COG_20260806.sav", 755_737, fmt="sav"),
    # The questionnaire-timing file declares UTF-8 but contains Latin-1
    # bytes; readstat refuses it without the override.
    Source("2025", "stu_tim", DATA_2025 / "CY09_MS_STU_TT_PUF.sav", 721_037,
           fmt="sav", encoding="LATIN1"),
    Source("2025", "stu_ttm", DATA_2025 / "CY09_MS_COG_PROCESS_20260806.sav", 721_037,
           fmt="sav"),
    Source("2025", "ldw_cog", DATA_2025 / "CY09_MS_LDW_20260806.sav", 721_037,
           fmt="sav"),
]


def read_metadata(source: Source):
    """Header-only read (row count, labels, types) with the right reader."""
    import pyreadstat

    kwargs = {"metadataonly": True}
    if source.encoding:
        kwargs["encoding"] = source.encoding
    reader = pyreadstat.read_sav if source.fmt == "sav" else pyreadstat.read_sas7bdat
    _, meta = reader(str(source.path), **kwargs)
    return meta


def get_sources(only: list[str] | None = None) -> list[Source]:
    """Filter sources by name (e.g. 'stu_qqq_2018'), instrument, or cycle."""
    if not only:
        return list(SOURCES)
    keys = {k.lower() for k in only}
    return [
        s for s in SOURCES
        if s.name in keys or s.instrument in keys or s.cycle in keys
    ]
