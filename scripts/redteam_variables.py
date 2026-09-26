"""Red-team probe of the "explore" route (which variables do I use for X).

Three phases, each resumable/independent:
  python scripts/redteam_variables.py scripted   # ground-truth catalog probe
  python scripts/redteam_variables.py student     # 150 student-style questions
  python scripts/redteam_variables.py report      # writes data/redteam_variables.md

Writes:
  data/redteam_variables_questions.json   (the 150 student questions, with expected codes)
  data/redteam_variables_results.json     (keys: "scripted", "student")
  data/redteam_variables.md               (final report)

Does not modify anything under explorer/, tests/, pipeline/.
"""
import json
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("PISA_EVENTS_BACKEND", "jsonl")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

QUESTIONS_PATH = ROOT / "data" / "redteam_variables_questions.json"
RESULTS_PATH = ROOT / "data" / "redteam_variables_results.json"
REPORT_PATH = ROOT / "data" / "redteam_variables.md"
CATALOG_PATH = ROOT / "data" / "catalog" / "variables.parquet"

CYCLES = ["2018", "2022", "2025"]

SKIP_EXACT = {
    "CNT", "CNTRYID", "CNTSCHID", "CNTSTUID", "STRATUM", "SUBNATIO", "REGION",
    "OECD", "ADMINMODE", "BOOKID", "UNIT", "WVARSTRR", "SENWT",
}


def _load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {}


def _save_results(data: dict) -> None:
    RESULTS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase 1: scripted probe (ground truth from the catalog)
# ---------------------------------------------------------------------------

def _eligible(row) -> bool:
    var = row.variable
    label = row.label or ""
    if row.instrument not in ("stu_qqq", "sch_qqq"):
        return False
    if var in SKIP_EXACT:
        return False
    if var.startswith("W_") or var.startswith("PV") or var.startswith("Option_"):
        return False
    is_wle = label.strip().endswith("(WLE)")
    is_allcaps = bool(re.fullmatch(r"[A-Z]{4,12}", var))
    return is_wle or is_allcaps


def build_scripted_targets() -> list[dict]:
    df = pd.read_parquet(CATALOG_PATH)
    mask = df.apply(_eligible, axis=1)
    sub = df[mask].drop_duplicates(subset=["variable", "cycle"])
    targets = []
    for cycle in CYCLES:
        cs = sub[sub.cycle == cycle]
        if len(cs) > 120:
            cs = cs.sample(120, random_state=0)
        for r in cs.itertuples():
            label = re.sub(r"\s*\(WLE\)\s*$", "", r.label).strip()
            targets.append({"variable": r.variable, "cycle": cycle,
                            "instrument": r.instrument, "label": label})
    return targets


def _row_match(table_variables: list[str], var: str):
    """Return (rank_1based, rows_above) if var appears (exact, or as the
    first member of a displayed family 'PV1MATH … PV10MATH'), else (None, None)."""
    for i, v in enumerate(table_variables):
        first = v.split(" … ")[0].strip()
        if first == var or v.strip() == var:
            return i + 1, i
    return None, None


def run_scripted(agent, limit=None):
    targets = build_scripted_targets()
    if limit:
        targets = targets[:limit]
    print(f"[scripted] {len(targets)} probe questions", flush=True)
    results = []
    for i, t in enumerate(targets):
        q = f"which variables measure {t['label']} in {t['cycle']}?"
        try:
            r = agent.ask(q)
        except Exception as e:  # noqa: BLE001
            results.append({**t, "question": q, "error": str(e)})
            continue
        table_vars = list(r.table.variable) if r.table is not None and not r.table.empty else []
        rank, above = _row_match(table_vars, t["variable"])
        results.append({
            **t,
            "question": q,
            "route": r.route,
            "found": rank is not None,
            "rank": rank,
            "rows_above": above,
            "n_rows": len(table_vars),
            "search_terms": (r.plan or {}).get("search_terms"),
            "top5": table_vars[:5],
        })
        if (i + 1) % 25 == 0:
            print(f"[scripted] {i + 1}/{len(targets)}", flush=True)
    data = _load_results()
    data["scripted"] = results
    _save_results(data)
    print(f"[scripted] done, {len(results)} rows saved", flush=True)
    return results


# ---------------------------------------------------------------------------
# Phase 2: student-style questions
# ---------------------------------------------------------------------------

_FAMILY = re.compile(r"^([A-Z_]+?)1-(\d+)([A-Z_]*)$")


def _first_member(code: str) -> str:
    """'PV1-10FLIT' -> 'PV1FLIT' (a family display code -> its first real
    catalog member); a plain code is returned unchanged."""
    m = _FAMILY.match(code)
    if not m:
        return code
    prefix, _, suffix = m.groups()
    return f"{prefix}1{suffix}"


def _verify(codes, cycle=None):
    """Sanity check: codes must exist in the catalog (in `cycle` if given).
    Family codes ('PV1-10FLIT') are checked via their first member."""
    if codes == "none":
        return codes
    df = pd.read_parquet(CATALOG_PATH)
    bad = []
    for c in codes:
        member = _first_member(c)
        hits = df[df.variable == member]
        if cycle:
            hits = hits[hits.cycle == str(cycle)]
        if hits.empty:
            bad.append(c)
    if bad:
        raise AssertionError(f"codes not found in catalog{' for ' + cycle if cycle else ''}: {bad}")
    return codes


def build_student_questions() -> list[dict]:
    Q = []

    def add(question, expected, note=""):
        Q.append({"id": len(Q) + 1, "question": question, "expected": expected, "note": note})

    # --- Mixed topics (blended asks) -------------------------------------
    add("I'm doing a project on financial literacy and growth mindset, what variables do I need?",
        _verify(["PV1-10FLIT"]) and ["PV1-10FLIT", "GROSAGR/ST184Q01HA"],
        "financial literacy (2018/2022 flt_qqq) + growth mindset (2022 GROSAGR, 2018 ST184Q01HA)")
    add("what variables cover ICT use and student wellbeing together?",
        ["ICTRES/ICTHOME/ICTSCH...", "ST016Q01NA/BELONG/BULLIED/FEELSAFE"], "ICT + well-being")
    add("I need gender and mathematics anxiety variables for my thesis",
        _verify(["ST004D01T", "MALE", "ANXMAT"]) and ["ST004D01T", "MALE", "ANXMAT"],
        "gender (all cycles) + math anxiety (2022 only)")
    add("looking for immigrant background and sense of belonging variables",
        _verify(["IMMIG", "ST022Q01TA", "LANGN", "BELONG"]) and ["IMMIG", "ST022Q01TA", "LANGN", "BELONG"],
        "immigrant background + belonging")
    add("what do you have on socioeconomic status and parental education?",
        _verify(["ESCS", "HISEI", "PAREDINT", "HISCED"]) and ["ESCS", "HISEI", "PAREDINT", "HISCED"],
        "ESCS topic already covers this")
    add("bullying and life satisfaction variables please",
        _verify(["BULLIED", "BEINGBULLIED", "ST016Q01NA"]) and ["BULLIED/BEINGBULLIED", "ST016Q01NA"],
        "bullying + well-being")
    add("what weight and identifier variables do I need to merge student and school files?",
        _verify(["W_FSTUWT", "CNT", "CNTSCHID"]) and ["W_FSTUWT", "CNT", "CNTSCHID"],
        "weights + identifiers")
    add("give me the mathematics subscale variables and the overall proficiency levels for 2022",
        ["PV1-10MCCR/MCQN/MCSS/MCUD/MPEM/MPFS/MPIN/MPRE", "none (levels are cut from PVs, not stored)"],
        "subscales exist; proficiency-level variable does not")
    add("school resources, class size and teacher shortage variables",
        _verify(["CLSIZE", "SCHSIZE", "STAFFSHORT", "STRATIO"]) and ["CLSIZE", "SCHSIZE", "STAFFSHORT", "STRATIO"],
        "school resources topic")
    add("I want AI use and ICT resources at home variables for 2025",
        _verify(["ST438Q01DA", "AIUSESCH", "ICTRES", "ICTHOME"]) and ["ST438Q01DA...Q04DA", "AIUSESCH", "ICTRES", "ICTHOME"],
        "AI (2025 only) + ICT")
    add("teacher job satisfaction and self-efficacy variables, and also student growth mindset",
        _verify(["SATJOB", "SATTEACH", "SEFFCM", "GROSAGR"]) and ["SATJOB", "SATTEACH", "SEFFCM/SEFFREL/SEFFINS", "GROSAGR"],
        "teacher + student constructs")
    add("what variables measure school type, location and student truancy?",
        _verify(["SC013Q01TA", "SC001Q01TA", "SKIPPING", "ST062Q01TA"]) and ["SC013Q01TA", "SC001Q01TA", "SKIPPING", "ST062Q01TA"],
        "school type + truancy")
    add("parental involvement and student educational expectations variables",
        _verify(["PA008Q05TA", "EXPECEDU", "BSMJ"]) and ["PA008Q05TA...", "EXPECEDU", "BSMJ"],
        "parent questionnaire + expectations")
    add("creative thinking and financial literacy variables for 2022",
        _verify(["PV1-10CRTH_NC", "PV1-10FLIT"]) and ["PV1-10CRTH_NC", "PV1-10FLIT"],
        "two 2022 optional domains")
    add("what variables track reading enjoyment and reading subscales in 2018?",
        _verify(["JOYREAD", "PV1RCLI"]) and ["JOYREAD", "PV1-10RCLI/RCUN/RCER/RTSN/RTML"],
        "2018 reading enjoyment + subscales")

    # --- Typos -------------------------------------------------------------
    add("wat varibales do i use for matematics scors",
        ["PV1-10MATH"], "typo-laden: math scores")
    add("wich varaibles mesure studnet welbeing",
        ["ST016Q01NA/BELONG/BULLIED/FEELSAFE"], "typo-laden: well-being")
    add("varibles for imigrant backgorund",
        ["IMMIG", "ST022Q01TA", "LANGN"], "typo-laden: immigrant background")
    add("finantial literasy varibales pls",
        ["PV1-10FLIT"], "typo-laden: financial literacy")
    add("skool typ and lokation varables",
        ["SC013Q01TA", "SC001Q01TA"], "typo-laden: school type/location")
    add("bulling varibale in 2022",
        ["BULLIED"], "typo-laden: bullying")
    add("teecher shortige and clas size",
        ["STAFFSHORT", "CLSIZE"], "typo-laden: teacher shortage/class size")
    add("grouth mindeset varible 2018",
        ["ST184Q01HA"], "typo-laden: growth mindset")
    add("wich variabels for gendr and escs",
        ["ST004D01T", "ESCS"], "typo-laden: gender + ESCS")
    add("weigth varaibles for anaylsis",
        ["W_FSTUWT", "W_FSTURWT1-80"], "typo-laden: weights")

    # --- Spanish -------------------------------------------------------------
    add("qué variables mido el bienestar de los estudiantes?",
        ["ST016Q01NA/BELONG/BULLIED/FEELSAFE"], "Spanish: well-being")
    add("qué variables uso para el nivel socioeconómico?",
        ["ESCS", "HISEI", "HOMEPOS"], "Spanish: socioeconomic status")
    add("necesito variables sobre acoso escolar (bullying)",
        ["BULLIED/BEINGBULLIED"], "Spanish: bullying")
    add("qué variables hay sobre el uso de inteligencia artificial en 2025?",
        ["ST438Q01DA", "AIUSESCH"], "Spanish: AI use")
    add("variables de antecedentes migratorios y idioma en casa",
        ["IMMIG", "ST022Q01TA", "LANGN"], "Spanish: immigrant background")
    add("qué variables miden el puntaje de matemáticas en 2022?",
        ["PV1-10MATH"], "Spanish: math score")
    add("cuáles son las variables de ponderación (pesos) para el análisis?",
        ["W_FSTUWT", "W_FSTURWT1-80"], "Spanish: weights")
    add("qué variables existen sobre el ingreso familiar en dólares?",
        "none", "Spanish: family income in dollars — does not exist")

    # --- Portuguese -------------------------------------------------------------
    add("quais variáveis medem o bem-estar dos estudantes?",
        ["ST016Q01NA/BELONG/BULLIED/FEELSAFE"], "Portuguese: well-being")
    add("quais variáveis uso para nível socioeconômico?",
        ["ESCS", "HISEI", "HOMEPOS"], "Portuguese: SES")
    add("preciso de variáveis sobre bullying escolar",
        ["BULLIED/BEINGBULLIED"], "Portuguese: bullying")
    add("quais variáveis existem sobre uso de inteligência artificial em 2025?",
        ["ST438Q01DA", "AIUSESCH"], "Portuguese: AI use")
    add("variáveis sobre antecedentes de imigração e idioma em casa",
        ["IMMIG", "ST022Q01TA", "LANGN"], "Portuguese: immigrant background")
    add("quais variáveis medem a pontuação de leitura em 2018?",
        ["PV1-10READ"], "Portuguese: reading score")
    add("quais são as variáveis de peso (ponderação) para a análise?",
        ["W_FSTUWT", "W_FSTURWT1-80"], "Portuguese: weights")

    # --- Cycle-specific -------------------------------------------------------------
    add("what variables measure creative thinking, and in which cycle?",
        ["PV1-10CRTH_NC (2022 only)"], "cycle-specific: creative thinking only in 2022")
    add("is there a global competence variable in 2022 or 2025?",
        "none (2018 only: PV1-10GLCM, GLOBMIND)", "cycle-specific: global competence 2018 only")
    add("what growth mindset variable exists in 2025?",
        "none (2018: ST184Q01HA; 2022: GROSAGR; no 2025 measure)", "cycle-specific: no growth mindset in 2025")
    add("which variables measure learning time in 2022?",
        "none for MMINS/LMINS/SMINS/TMINS (2018 only); 2022 uses ST296 homework-time items",
        "cycle-specific: learning-time minutes only in 2018")
    add("what variables measure mathematics self-efficacy in 2018?",
        "none (mathematics anxiety/self-efficacy only asked in 2022, when math was the major domain)",
        "cycle-specific: 2018 major domain was reading, not math")
    add("give me the science subscale variables for 2018",
        "none (science subscales only in 2025; 2018 has no science-subscale PVs)",
        "cycle-specific: science subscales only in 2025")
    add("what ICT variables are new in the 2025 questionnaire compared to before?",
        ["ICTAVHOME", "AIUSESCH", "IC170Q10DA"], "cycle-specific: 2025-only ICT/AI variables")
    add("which gender variable should I use for the 2025 data?",
        ["MALE", "ST004D01T"], "cycle-specific: 2025 convention is MALE")
    add("what financial literacy variables exist in 2025?",
        "none (financial literacy assessed only in 2018 and 2022)", "cycle-specific: no 2025 FLIT")
    add("what variables measure the Learning in the Digital World domain, only asked in 2025 right?",
        ["PV1-10CMPS", "PV1-10CPPK", "PV1-10CMOD", "PV1-10CPRO", "SELFREG"], "cycle-specific: LDW 2025 only")
    add("does 2018 have a bullying index called BULLIED?",
        "none (2018 uses BEINGBULLIED; BULLIED is the 2022/2025 name)", "cycle-specific: renamed index")
    add("what's the weight variable in 2025, is it different from 2018?",
        ["W_FSTUWT", "W_FSTURWT1-80"], "cycle-specific: same weight name across cycles")
    add("which variables measure student truancy in 2018?",
        ["ST062Q01TA"], "cycle-specific: no derived SKIPPING/TARDYSD index until 2022")
    add("in 2022, what variable captures assertiveness?",
        ["ASSERAGR"], "cycle-specific: 2022 social-emotional skills module")
    add("what socioeconomic variables were dropped after 2018?",
        ["none after 2018: WEALTH, CULTPOSS, HEDRES, PARED"], "cycle-specific: 2018-only SES components")

    # --- Weights / identifiers / merging -------------------------------------------------------------
    add("what variable do I use to weight my analysis?",
        ["W_FSTUWT"], "weights")
    add("what are the replicate weights called?",
        ["W_FSTURWT1-80"], "weights")
    add("how do I merge the student and school files?",
        ["CNT", "CNTSCHID"], "identifiers")
    add("what's the unique student identifier?",
        ["CNTSTUID"], "identifiers")
    add("what variable identifies the school?",
        ["CNTSCHID"], "identifiers")
    add("what variable gives the sampling stratum?",
        ["STRATUM"], "identifiers")
    add("what's the senate weight and when do I use it?",
        ["SENWT"], "weights")
    add("what variable tells me the administration mode (paper or computer)?",
        ["ADMINMODE"], "identifiers")
    add("what variable is the 3-letter country code?",
        ["CNT"], "identifiers")
    add("I need to join creative thinking data to the main student file, what links them?",
        ["CNTSTUID", "CNT"], "merging stu_crt")

    # --- Proficiency levels -------------------------------------------------------------
    add("what variable gives the proficiency level in mathematics?",
        "none (cut from PV1-10MATH at OECD thresholds)", "proficiency levels not stored")
    add("is there a variable for below level 2 in reading?",
        "none (compute a share below the level-2 cutoff from PV1-10READ)", "proficiency levels not stored")
    add("what variable shows top performers in science?",
        "none (cut from PV1-10SCIE at OECD thresholds)", "proficiency levels not stored")
    add("which column has the baseline proficiency level?",
        "none", "proficiency levels not stored")
    add("what variable classifies students as low performers?",
        "none", "proficiency levels not stored")
    add("what's the exact score cutoff variable for level 2 math?",
        "none (420.07 is a fixed OECD threshold, not a variable)", "proficiency levels not stored")

    # --- Subscales -------------------------------------------------------------
    add("what are the mathematics content subscale variables for 2022?",
        ["PV1-10MCQN", "PV1-10MCSS", "PV1-10MCUD", "PV1-10MCCR"], "math content subscales")
    add("what are the mathematics process subscale variables for 2022?",
        ["PV1-10MPEM", "PV1-10MPFS", "PV1-10MPIN", "PV1-10MPRE"], "math process subscales")
    add("what are the reading cognitive-process subscales in 2018?",
        ["PV1-10RCLI", "PV1-10RCUN", "PV1-10RCER"], "reading subscales 2018")
    add("what are the reading text-structure subscales in 2018?",
        ["PV1-10RTSN", "PV1-10RTML"], "reading subscales 2018")
    add("what science competency subscales exist in 2025?",
        ["PV1-10SEPS", "PV1-10SEDE", "PV1-10SEID"], "science subscales 2025")
    add("is there an environmental science subscale?",
        ["PV1-10SENV"], "2025 environmental science")
    add("what's the space and shape subscale variable?",
        ["PV1-10MCSS"], "math content subscale")
    add("what's the employing subscale in mathematics?",
        ["PV1-10MPEM"], "math process subscale")
    add("locate information subscale variable in reading?",
        ["PV1-10RCLI"], "reading subscale")

    # --- School-level variables -------------------------------------------------------------
    add("what variables describe the school principal's questionnaire?",
        ["SC013Q01TA/SC001Q01TA/STAFFSHORT/..."], "school questionnaire (sch_qqq)")
    add("what's the school size variable?",
        ["SCHSIZE"], "school size (not in 2025)")
    add("what variable gives the student-teacher ratio?",
        ["STRATIO"], "school resources (not in 2025)")
    add("is the school public or private, what variable?",
        ["SC013Q01TA"], "school type")
    add("what variable measures community size (urban/rural)?",
        ["SC001Q01TA"], "school location")
    add("what variable flags a private school directly?",
        ["PRIVATESCH"], "school type derived")
    add("what's the school type variable, is it different from public/private?",
        ["SCHLTYPE"], "school type derived (more categories)")
    add("what variable measures shortage of educational material?",
        ["EDUSHORT"], "school resources")
    add("what variable measures the proportion of certified teachers?",
        ["PROATCE"], "school resources")
    add("does the school questionnaire have a class size variable?",
        ["CLSIZE"], "school resources")
    add("what variable measures the school's weight for analysis?",
        ["W_SCHGRNRABWT"], "school weight (2018, 2022 only)")
    add("how many students are enrolled at the school, what variable?",
        ["SCHSIZE"], "school size (not 2025)")

    # --- Teacher questionnaire -------------------------------------------------------------
    add("what variables measure teacher job satisfaction?",
        ["SATJOB", "SATTEACH"], "teacher questionnaire (2018/2022 only)")
    add("what variables measure teacher self-efficacy?",
        ["SEFFCM", "SEFFREL", "SEFFINS"], "teacher questionnaire (all cycles)")
    add("is there a teacher collaboration variable?",
        ["COLT"], "teacher questionnaire 2018")
    add("what variables cover teacher professional development?",
        ["TC045Q01NB...TC045Q06NB"], "teacher questionnaire 2018")
    add("what variable measures teacher self-efficacy in multicultural environments?",
        ["GCSELF"], "teacher questionnaire 2018")
    add("is there a variable on how often teachers use digital tools?",
        ["TC220Q06JA"], "teacher questionnaire 2022")
    add("what's the teacher questionnaire table called?",
        ["tch_qqq"], "table name, not a construct")
    add("does the 2025 teacher questionnaire have a self-efficacy variable for science?",
        ["CONEXSCI"], "teacher questionnaire 2025")
    add("what variable shows teacher-reported principal collaboration on discipline?",
        ["TC253Q01JA"], "teacher questionnaire 2022")
    add("is there a teacher salary variable?",
        "none (teacher salaries are not collected in PISA)", "no teacher salary variable — false-positive risk")

    # --- Parent questionnaire -------------------------------------------------------------
    add("what variables come from the parent questionnaire?",
        ["PA007Q09NA", "PA008Q05TA"], "optional parent questionnaire, stored in stu_qqq")
    add("is there a variable on parents attending school meetings?",
        ["PA008Q08NA"], "parent questionnaire")
    add("what variable shows parents are involved in school decision-making?",
        ["PA007Q12NA"], "parent questionnaire")
    add("do parents report supporting the child's educational efforts?",
        ["ST123Q02NA"], "student-reported parental support, 2018")
    add("what variable measures parental educational expectations?",
        ["PAREDINT", "HISCED"], "parental education, not expectations for the child (EXPECEDU is the student's own)")
    add("is there a parent questionnaire in every cycle?",
        ["PA007.../PA008... (2018, 2022, 2025 — optional, not every economy)"], "parent items exist across cycles")

    # --- ICT / AI -------------------------------------------------------------
    add("what variables measure ICT resources at home?",
        ["ICTRES", "ICTHOME"], "ICT")
    add("what variables measure ICT use at school?",
        ["ICTSCH"], "ICT")
    add("is there a variable on internet access outside school?",
        ["ICTOUT"], "ICT (2022/2025)")
    add("what variable measures how available ICT is at school?",
        ["ICTAVSCH"], "ICT")
    add("what variables measure students using AI chatbots for schoolwork?",
        ["ST438Q01DA"], "AI use, 2025 only")
    add("is there a variable on AI use at school, from the ICT questionnaire?",
        ["AIUSESCH", "IC170Q10DA"], "AI, 2025 only")
    add("does PISA measure AI literacy as a cognitive test?",
        "none (no PISA cycle has assessed AI literacy)", "no AI literacy test — false-positive risk")
    add("what variable measures screen time or smartphone use?",
        ["ICTHOME/ENTUSE"], "ICT related")
    add("is there an ICT self-efficacy variable?",
        ["ICTEFFIC"], "ICT (2022)")
    add("what variable measures digital distraction in class?",
        ["ICTDISTR"], "ICT (2022/2025)")

    # --- Well-being -------------------------------------------------------------
    add("what variable measures overall life satisfaction?",
        ["ST016Q01NA"], "well-being, every cycle")
    add("is there a mental health variable?",
        ["none directly; well-being proxies: ST016Q01NA, LIFESAT (2022), EXPWB (2022)"],
        "no direct mental-health diagnosis variable")
    add("what variable measures feeling safe at school?",
        ["FEELSAFE"], "well-being (2022/2025)")
    add("what variable measures eudaimonic well-being?",
        ["EUDMO"], "well-being (2018 only)")
    add("is there a subjective well-being variable in 2018?",
        ["SWBP"], "well-being 2018")
    add("what variable covers experienced well-being in 2022?",
        ["EXPWB"], "well-being 2022, optional module")
    add("does PISA measure clinical depression or anxiety disorders?",
        "none (PISA is not a clinical instrument; ANXMAT is mathematics anxiety, not a diagnosis)",
        "no clinical mental-health variable — false-positive risk")
    add("what variable is the headline OECD well-being indicator?",
        ["ST016Q01NA"], "well-being")

    # --- Immigrant background -------------------------------------------------------------
    add("what variable shows if a student is an immigrant?",
        ["IMMIG"], "immigrant background")
    add("what variable shows the language spoken at home?",
        ["ST022Q01TA", "LANGN"], "immigrant background / language")
    add("is there a variable distinguishing first- and second-generation immigrants?",
        ["IMMIG"], "immigrant background (values 2 and 3)")
    add("what variable gives the country of birth of the student's parents?",
        ["ST019BQ01T", "ST019CQ01T"], "immigrant background, 2018")
    add("how many languages does the student speak with their parents?",
        ["ST177Q01HA"], "immigrant background, 2018")
    add("is there a variable for whether a student is a refugee?",
        "none (PISA does not ask refugee status)", "no refugee variable — false-positive risk")

    # --- PISA does NOT have (>=20) -------------------------------------------------------------
    add("what variable measures how many hours students sleep?",
        "none", "PISA does not ask about sleep")
    add("is there a variable for students' diet or nutrition?",
        "none", "PISA does not ask about diet")
    add("what variable gives family income in dollars?",
        "none (ESCS/HOMEPOS/WEALTH are indices, not currency amounts)", "no income-in-dollars variable")
    add("is there an IQ or intelligence-test variable?",
        "none (PISA tests subject literacy, not IQ)", "no IQ variable")
    add("what variable has the teacher's salary?",
        "none", "no teacher salary variable")
    add("is there a variable with the student's name?",
        "none (data are anonymized; no names collected)", "no student names")
    add("what variable shows the student's exam grade or GPA?",
        "none (PISA is not linked to school grades/GPA)", "no exam grades")
    add("is there a university admission or acceptance variable?",
        "none", "no university admission data")
    add("what variable records the student's religion?",
        "none", "PISA does not ask about religion")
    add("is there a political views or party preference variable?",
        "none", "PISA does not ask about politics")
    add("what variable is the school attendance register?",
        "none (SKIPPING/ST062 are self-reported truancy, not an attendance register)",
        "no attendance register — SKIPPING may be a false positive if presented as one")
    add("what variable gives the student's height or weight (BMI)?",
        "none", "no anthropometric data")
    add("is there a criminal record or disciplinary record variable?",
        "none", "no disciplinary record")
    add("what variable measures vaccination status?",
        "none", "no health/vaccination data")
    add("what variable has the student's home address?",
        "none", "no addresses (anonymized)")
    add("is there a variable for the student's exact birth date?",
        "none (only birth month/year via age variables)", "no exact birth date")
    add("what variable shows the student's social media accounts?",
        "none (ICT items ask about use/frequency, not specific accounts)", "no social-media-account variable")
    add("is there a variable for standardized national exam scores (e.g. SAT)?",
        "none (PISA runs its own cognitive test; no external exam scores)", "no external test scores")
    add("what variable gives the number of siblings?",
        "none", "PISA does not ask about siblings")
    add("is there a variable for the student's marital status?",
        "none", "not applicable / not asked")
    add("what variable measures classroom Wi-Fi speed in Mbps?",
        "none (ICT items ask about availability/frequency, not technical speed)", "no bandwidth measurement")
    add("is there a variable for the school's annual budget in currency?",
        "none (EDUSHORT/STAFFSHORT are perceived-shortage indices, not budgets)", "no budget figures")
    add("what variable records a student's criminal background check?",
        "none", "not applicable")
    add("is there a variable measuring commute distance to school in kilometers?",
        "none", "PISA does not ask about commute distance")

    return Q


def run_student(agent, limit=None):
    if QUESTIONS_PATH.exists():
        questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    else:
        questions = build_student_questions()
        QUESTIONS_PATH.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[student] {len(questions)} questions", flush=True)
    if limit:
        questions = questions[:limit]
    data = _load_results()
    results = data.get("student") or []
    done_ids = {r["id"] for r in results if "id" in r}
    for i, q in enumerate(questions):
        if q["id"] in done_ids:
            continue
        try:
            r = agent.ask(q["question"])
        except Exception as e:  # noqa: BLE001
            results.append({**q, "error": str(e)})
            data["student"] = results
            _save_results(data)
            continue
        table = r.table
        rows = []
        if table is not None and not table.empty:
            cols = [c for c in ("variable", "label", "role") if c in table.columns]
            if cols:
                for row in table.head(10)[cols].itertuples(index=False):
                    rows.append(dict(zip(cols, row)))
            else:
                rows = [{"_columns": list(table.columns), "_n_rows": len(table)}]
        results.append({
            **q,
            "route": r.route,
            "answer": r.answer,
            "table_rows": rows,
            "guards": r.guards,
        })
        if (i + 1) % 10 == 0:
            data["student"] = results
            _save_results(data)
            print(f"[student] {i + 1}/{len(questions)}", flush=True)
    data["student"] = results
    _save_results(data)
    print(f"[student] done, {len(results)} rows saved", flush=True)
    return results


# ---------------------------------------------------------------------------
# Phase 3: report
# ---------------------------------------------------------------------------

def make_report():
    data = _load_results()
    scripted = data.get("scripted", [])
    student = data.get("student", [])

    lines = []
    lines.append("# Red team: explore route (which variables do I use for X)\n")

    # --- scripted stats ---
    n = len(scripted)
    found = sum(1 for r in scripted if r.get("found"))
    rank1 = sum(1 for r in scripted if r.get("rank") == 1)
    top5 = sum(1 for r in scripted if r.get("rank") and r["rank"] <= 5)
    not_explore = sum(1 for r in scripted if r.get("route") != "explore")
    errors = sum(1 for r in scripted if r.get("error"))

    lines.append("## Pass rates\n")
    lines.append("### Scripted probe (ground truth from the catalog)\n")
    lines.append("| metric | count | pct |")
    lines.append("|---|---|---|")
    lines.append(f"| questions | {n} | - |")
    lines.append(f"| errors (exception) | {errors} | {errors/n*100:.1f}% |" if n else "| errors | 0 | - |")
    lines.append(f"| true code found anywhere in table | {found} | {found/n*100:.1f}% |" if n else "")
    lines.append(f"| found at rank 1 | {rank1} | {rank1/n*100:.1f}% |" if n else "")
    lines.append(f"| found in top 5 | {top5} | {top5/n*100:.1f}% |" if n else "")
    lines.append(f"| route != explore | {not_explore} | {not_explore/n*100:.1f}% |" if n else "")

    lines.append("\nBy cycle:\n")
    lines.append("| cycle | n | found | rank1 | top5 |")
    lines.append("|---|---|---|---|---|")
    for cyc in CYCLES:
        rows = [r for r in scripted if r.get("cycle") == cyc]
        nn = len(rows)
        if not nn:
            continue
        f = sum(1 for r in rows if r.get("found"))
        r1 = sum(1 for r in rows if r.get("rank") == 1)
        t5 = sum(1 for r in rows if r.get("rank") and r["rank"] <= 5)
        lines.append(f"| {cyc} | {nn} | {f} ({f/nn*100:.0f}%) | {r1} ({r1/nn*100:.0f}%) | {t5} ({t5/nn*100:.0f}%) |")

    # --- student stats ---
    sn = len(student)
    none_qs = [r for r in student if r.get("expected") == "none" or
               (isinstance(r.get("expected"), str) and r.get("expected").startswith("none"))]
    coded_qs = [r for r in student if r not in none_qs]

    def has_false_positive(r):
        rows = r.get("table_rows") or []
        return len(rows) > 0

    fp_on_none = sum(1 for r in none_qs if has_false_positive(r))
    non_explore_route = sum(1 for r in student if r.get("route") != "explore")

    lines.append("\n### Student-style questions\n")
    lines.append("| metric | count | pct |")
    lines.append("|---|---|---|")
    lines.append(f"| total questions | {sn} | - |")
    lines.append(f"| \"no such variable\" questions | {len(none_qs)} | - |")
    lines.append(f"| — of those, table returned rows (possible false positive) | {fp_on_none} | "
                  f"{fp_on_none/len(none_qs)*100:.0f}%" if none_qs else "0")
    lines.append(f"| route != explore (any question) | {non_explore_route} | {non_explore_route/sn*100:.0f}%" if sn else "0")

    RESULTS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_PATH.write_text("\n".join(str(x) for x in lines if x is not None), encoding="utf-8")
    print("[report] skeleton written; run the interactive triage pass to finish it.", flush=True)


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    if phase == "questions":
        qs = build_student_questions()
        QUESTIONS_PATH.write_text(json.dumps(qs, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {len(qs)} questions to {QUESTIONS_PATH}")
        return
    from explorer.agent import Agent
    print("building agent...", flush=True)
    t0 = time.time()
    agent = Agent()
    print(f"agent built in {time.time()-t0:.1f}s", flush=True)
    if phase in ("scripted", "all"):
        run_scripted(agent)
    if phase in ("student", "all"):
        run_student(agent)
    if phase in ("report", "all"):
        make_report()


if __name__ == "__main__":
    random.seed(0)
    main()
