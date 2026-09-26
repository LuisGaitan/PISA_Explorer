"""Topic -> the variables a student should use, per cycle, verified.

The tool was built so that a class can find the RIGHT variables for its own
analysis ("what variables do I use for math scores and social-emotional
learning in 2022?"). Keyword retrieval alone answered that badly: it never
named the plausible values (their codebook label says "Plausible Value 1 in
Mathematics", not "score"), ranked lexical MATH* look-alikes above the 2022
social and emotional skills scales, and listed one arbitrary PV number
(PV6MATH) out of ten. This table answers first, deterministically; keyword
hits follow as "related".

Every code here is checked against the catalog by tests/test_round4.py when
the data are present: a topic may only list variables that exist in the
named cycle and table. Families ("PV1-10MATH", "W_FSTURWT1-80") expand to
their members for that check and are shown as one row.
"""

import re
from dataclasses import dataclass

CYCLES = ("2018", "2022", "2025")

FAMILY = re.compile(r"^([A-Z_]+?)1-(\d+)([A-Z_]*)$")


def expand(code: str) -> list[str]:
    """'PV1-10MATH' -> ['PV1MATH', ..., 'PV10MATH']; a plain code -> [code]."""
    m = FAMILY.match(code)
    if not m:
        return [code]
    prefix, last, suffix = m.group(1), int(m.group(2)), m.group(3)
    return [f"{prefix}{i}{suffix}" for i in range(1, last + 1)]


def display(code: str) -> str:
    """'PV1-10MATH' -> 'PV1MATH … PV10MATH'."""
    members = expand(code)
    return code if len(members) == 1 else f"{members[0]} … {members[-1]}"


SCORE_WORDS = re.compile(
    r"\b(scores?|performance|achievement|results?|outcomes?|proficien\w*|plausible|"
    r"pvs?|test|literacy|competenc\w*|dependent variable|attainment)\b", re.I)


@dataclass(frozen=True)
class Topic:
    construct: str
    pattern: re.Pattern
    variables: dict                      # cycle -> tuple of codes / families
    instrument: str = "stu_qqq"
    note: str = ""
    requires: re.Pattern | None = None   # a second pattern that must also match

    def matches(self, text: str) -> bool:
        return bool(self.pattern.search(text)) and \
            (self.requires is None or bool(self.requires.search(text)))


def _all(*codes: str) -> dict:
    return {c: tuple(codes) for c in CYCLES}


PV_NOTE = ("ten plausible values per student: every statistic is computed ten times "
           "and combined with Rubin's rules, with the final student weight W_FSTUWT "
           "and the 80 replicate weights W_FSTURWT1-80 for the standard error "
           "(never a single PV, never the average of the ten)")

TOPICS: list[Topic] = [
    Topic("mathematics score",
          re.compile(r"\bmath(s|ematics|ematical|ematic)?\b", re.I), _all("PV1-10MATH"),
          note=PV_NOTE, requires=SCORE_WORDS),
    Topic("reading score",
          re.compile(r"\bread(ing)?\b", re.I), _all("PV1-10READ"),
          note=PV_NOTE, requires=SCORE_WORDS),
    Topic("science score",
          re.compile(r"\bscien(ce|ces|tific)\b", re.I), _all("PV1-10SCIE"),
          note=PV_NOTE, requires=SCORE_WORDS),
    Topic("test scores in the three core domains",
          re.compile(r"\b(test|pisa|cognitive|academic|student|overall) "
                     r"(scores?|results?|performance|achievement|outcomes?)\b|"
                     r"\b(achievement|attainment)\b|\bplausible values?\b|\bpvs?\b|"
                     r"\b(outcome|dependent) variables?\b|\bscores? variables?\b", re.I),
          _all("PV1-10MATH", "PV1-10READ", "PV1-10SCIE"), note=PV_NOTE),
    Topic("mathematics subscales (2022)",
          re.compile(r"\b(subscales?|content (areas?|categor\w+)|process(es)? (categor\w+|subscales?)|"
                     r"change and relationships|space and shape|uncertainty and data|"
                     r"quantity|formulating|employing|interpreting|reasoning)\b", re.I),
          {"2022": ("PV1-10MCCR", "PV1-10MCQN", "PV1-10MCSS", "PV1-10MCUD",
                    "PV1-10MPEM", "PV1-10MPFS", "PV1-10MPIN", "PV1-10MPRE")},
          note="content subscales (change and relationships, quantity, space and shape, "
               "uncertainty and data) and process subscales (employing, formulating, "
               "interpreting, reasoning) of the 2022 mathematics scale",
          requires=re.compile(r"\bmath", re.I)),
    Topic("reading subscales (2018)",
          re.compile(r"\b(subscales?|locate information|understand(ing)?|evaluate and reflect|"
                     r"single text|multiple text|text structure|cognitive process)\b", re.I),
          {"2018": ("PV1-10RCLI", "PV1-10RCUN", "PV1-10RCER", "PV1-10RTSN", "PV1-10RTML")},
          note="cognitive-process subscales (locate information, understand, evaluate and "
               "reflect) and text-structure subscales (single, multiple) of the 2018 reading scale",
          requires=re.compile(r"\bread", re.I)),
    Topic("science subscales (2025)",
          re.compile(r"\b(subscales?|explain phenomena|evaluate designs|scientific enquiry|"
                     r"scientific information|environmental science|competenc\w+ subscale)\b", re.I),
          {"2025": ("PV1-10SEPS", "PV1-10SEDE", "PV1-10SEID", "PV1-10SENV")},
          note="competency subscales of the 2025 science scale (explain phenomena, evaluate "
               "designs for enquiry, evaluate information for decisions) and environmental science",
          requires=re.compile(r"\bscien", re.I)),
    Topic("global competence (2018 innovative domain)",
          re.compile(r"\bglobal competenc[ey]\b|\bglobal[- ]mindedness\b", re.I),
          {"2018": ("PV1-10GLCM", "GLOBMIND")},
          note="the cognitive test was taken by 27 economies (PV1-10GLCM); GLOBMIND is the "
               "questionnaire index of global-mindedness (WLE)"),
    Topic("creative thinking (2022 innovative domain)",
          re.compile(r"\bcreativ(e|ity)\b", re.I),
          {"2022": ("PV1-10CRTH_NC",)}, instrument="crt_cog",
          note="creative-thinking plausible values (0-60 'number correct' scale) are in the "
               "2022 creative-thinking cognitive file crt_cog_2022, joined to students as "
               "stu_crt; CREATEFF (creative self-efficacy, WLE) is in stu_qqq_2022"),
    Topic("financial literacy",
          re.compile(r"\bfinancial (literacy|knowledge|education|skills)\b", re.I),
          {"2018": ("PV1-10FLIT",), "2022": ("PV1-10FLIT",)}, instrument="flt_qqq",
          note="the financial-literacy files (flt_qqq) hold the subsample that took the test, "
               "with their own W_FSTUWT and replicate weights; 2018 and 2022 only"),
    Topic("Learning in the Digital World (2025 innovative domain)",
          re.compile(r"\blearning in the digital world\b|\bldw\b|\bcomputational (thinking|problem|practices)\b|"
                     r"\bself-?regulated learning\b", re.I),
          {"2025": ("PV1-10CMPS", "PV1-10CPPK", "PV1-10CMOD", "PV1-10CPRO", "SELFREG")},
          note="the four LDW plausible-value scales (computational problem solving, "
               "computational practices prior knowledge, modelling, programming) are in "
               "stu_qqq_2025 for computer-based economies; item-level responses are in ldw_cog_2025"),
    Topic("weights",
          re.compile(r"\bweights?\b|\bweighting\b|\breplicates?\b|\bbrr\b|\bfay\b|\bw_fstuwt\b|"
                     r"\bsenate\b|\bsampling (design|variance)\b|\bstandard errors?\b", re.I),
          _all("W_FSTUWT", "W_FSTURWT1-80", "SENWT"),
          note="W_FSTUWT is the final student weight for every point estimate; the 80 "
               "Fay-BRR replicate weights (k = 0.5) give the sampling variance as the sum of "
               "squared replicate deviations divided by 20; SENWT (senate weight) makes every "
               "economy count equally when pooling. School files carry W_SCHGRNRABWT (2018, 2022)"),
    Topic("identifiers for merging files",
          re.compile(r"\b(identifiers?|ids?|student id|school id|country code|merge|merging|"
                     r"join(ing)?|link(ing)? (the )?(student|school|files?))\b", re.I),
          _all("CNT", "CNTRYID", "CNTSCHID", "CNTSTUID", "STRATUM", "OECD", "ADMINMODE"),
          note="students join their school on CNT + CNTSCHID (the app's stu_sch view does "
               "this); CNTSTUID is unique within a cycle only; STRATUM is the sampling stratum"),
    Topic("socio-economic status",
          re.compile(r"socio-?economic|\bescs\b|\bses\b|economic status|social background|"
                     r"family background|\bwealth\b|parental (education|occupation)|"
                     r"home possessions|\bdisadvantaged\b|\badvantaged\b|\bpoverty\b|\bincome\b", re.I),
          {"2018": ("ESCS", "HISEI", "PAREDINT", "PARED", "HISCED", "HOMEPOS", "WEALTH", "CULTPOSS", "HEDRES"),
           "2022": ("ESCS", "HISEI", "PAREDINT", "HISCED", "HOMEPOS"),
           "2025": ("ESCS", "HISEI", "PAREDINT", "HISCED", "HOMEPOS")},
          note="ESCS is the OECD index (mean 0, SD 1 across OECD countries in each cycle, so "
               "its level is not comparable across cycles); HISEI, parental education and "
               "HOMEPOS are its three components"),
    Topic("gender",
          re.compile(r"\b(gender|girls?|boys?|female|male|sex)\b", re.I),
          {"2018": ("ST004D01T",), "2022": ("ST004D01T",), "2025": ("MALE", "ST004D01T")},
          note="ST004D01T: 1 = female, 2 = male. In 2025 fourteen economies release no "
               "ST004D01T; the derived MALE flag (1 = male, 0 = female/other) is complete for "
               "all 90 and is the 2025 convention"),
    Topic("immigrant background and home language",
          re.compile(r"immigra|migrant|native[- ]born|foreign[- ]born|language (spoken )?at home|home language", re.I),
          _all("IMMIG", "ST022Q01TA", "LANGN"),
          note="IMMIG: 1 native, 2 second-generation, 3 first-generation (immigrant = 2 or 3); "
               "ST022Q01TA: 1 = language of the test at home, 2 = another language"),
    Topic("grade repetition",
          re.compile(r"repeat(ed|ing)? (a |the )?(grade|year|class)|grade repetition|repeaters?", re.I),
          _all("REPEAT"), note="1 = repeated a grade at least once, 0 = never"),
    Topic("grade and age",
          re.compile(r"\b(grade level|which grade|what grade|modal grade|students?'? age|age of (the )?students?|"
                     r"year of birth|birth (month|year))\b", re.I),
          _all("GRADE", "ST001D01T", "AGE"),
          note="GRADE is the student's grade relative to the modal grade for 15-year-olds in "
               "the economy; PISA samples 15-year-olds only"),
    Topic("social and emotional skills",
          re.compile(r"social[- ]?(and|&)?[- ]?emotional|socio-?emotional|\bsel\b|non-?cognitive|"
                     r"soft skills|character skills|big five|personality|21st[- ]century skills|"
                     r"transversal|life skills|\bpersever|\bgrit\b|\bcuriosity\b|\bcooperation\b|"
                     r"\bempathy\b|\bassertiveness\b|stress resistance|emotional control|"
                     r"self-?regulation|goal setting", re.I),
          {"2018": ("RESILIENCE", "WORKMAST", "MASTGOAL", "GFOFAIL", "COMPETE", "PERCOOP",
                    "PERCOMP", "EMOSUPS", "SWBP", "EUDMO", "BELONG"),
           "2022": ("PERSEVAGR", "CURIOAGR", "COOPAGR", "EMPATAGR", "ASSERAGR", "STRESAGR",
                    "EMOCOAGR", "GROSAGR", "ANXMAT", "MATHEFF", "MATHEF21", "CREATEFF",
                    "SDLEFF", "BELONG"),
           "2025": ("PERSEV", "CURIO", "SELFREG", "GOALSET", "COGABIL", "ENPROBS", "EFFSCIE",
                    "FAMSUP", "BELONG")},
          note="2022 has the social and emotional skills module proper: seven agreement scales "
               "(perseverance, curiosity, cooperation, empathy, assertiveness, stress resistance, "
               "emotional control; WLE, OECD mean 0, SD 1) plus growth mindset, mathematics "
               "anxiety and self-efficacy. 2018 has motivation and attitude indices (resilience, "
               "work mastery, mastery goals, fear of failure, competitiveness); 2025 has "
               "perseverance, curiosity, self-regulation, goal setting, cognitive adaptability "
               "and science self-efficacy. Different scales per cycle: levels only, no "
               "cross-cycle change"),
    Topic("well-being",
          re.compile(r"well-?being|wellbeing|life satisfaction|\bhapp(y|iness)\b|mental health|"
                     r"feel(ing)? safe|\bsafety\b|psycholog", re.I),
          {"2018": ("ST016Q01NA", "SWBP", "EUDMO", "BELONG", "BEINGBULLIED"),
           "2022": ("ST016Q01NA", "LIFESAT", "EXPWB", "BELONG", "BULLIED", "FEELSAFE"),
           "2025": ("ST016Q01NA", "BELONG", "BULLIED", "FEELSAFE", "FAMSUP")},
          note="ST016Q01NA is the 0-10 life-satisfaction item asked in every cycle (the OECD's "
               "headline well-being indicator); the WLE indices differ by cycle; EXPWB (2022) "
               "comes from the optional well-being questionnaire, so only some economies have it"),
    Topic("bullying",
          re.compile(r"bull(y|ied|ying|ies)", re.I),
          {"2018": ("BEINGBULLIED", "ST038Q03NA", "ST038Q04NA"),
           "2022": ("BULLIED", "ST038Q03NA", "ST038Q04NA"),
           "2025": ("BULLIED", "ST038Q04NA")},
          note="the exposure-to-bullying index (WLE) was renamed BEINGBULLIED (2018) to "
               "BULLIED (2022, 2025); ST038 items are the underlying frequency questions"),
    Topic("sense of belonging",
          re.compile(r"belong", re.I), _all("BELONG", "ST034Q01TA"),
          note="BELONG (WLE) exists in every cycle; ST034 items are its questions"),
    Topic("growth mindset",
          re.compile(r"growth mindset|\bmindset\b|intelligence (is|can) (something|change)", re.I),
          {"2018": ("ST184Q01HA",), "2022": ("GROSAGR",)},
          note="GROSAGR (WLE) exists in 2022 only; 2018 has the single item ST184Q01HA "
               "(your intelligence is something about you that you cannot change very much); "
               "no growth-mindset measure in 2025"),
    Topic("mathematics anxiety and self-efficacy",
          re.compile(r"anxiet|anxious|self-?efficacy|confiden(ce|t)\b", re.I),
          {"2022": ("ANXMAT", "MATHEFF", "MATHEF21", "SDLEFF", "ICTEFFIC"),
           "2025": ("EFFSCIE",)},
          note="mathematics anxiety and mathematics self-efficacy were collected in 2022 "
               "(major domain mathematics); 2025 has science self-efficacy EFFSCIE; 2018 "
               "(major domain reading) has neither"),
    Topic("enjoyment and motivation",
          re.compile(r"\benjoy|\bmotivat|\binterest(ed)? in\b|\bengag(e|ed|ement)\b", re.I),
          {"2018": ("JOYREAD", "MASTGOAL", "WORKMAST", "COMPETE"),
           "2022": ("MATHMOT", "PERSEVAGR"),
           "2025": ("JOYSCIE", "ENPROBS", "ENGSCIPR", "PERSEV")},
          note="each cycle asks about its major domain: enjoyment of reading (2018), "
               "motivation in mathematics (2022), enjoyment of science (2025)"),
    Topic("classroom climate and teacher support",
          re.compile(r"disciplin|classroom climate|teacher support|cognitive activation|"
                     r"teacher[- ]student relation", re.I),
          {"2018": ("DISCLIMA", "TEACHSUP"),
           "2022": ("TEACHSUP", "COGACMCO", "COGACRCO", "RELATST"),
           "2025": ("DISCLISCI", "TEACHSUP", "COGACSC")},
          note="asked about lessons in the cycle's major domain (reading 2018, mathematics "
               "2022, science 2025)"),
    Topic("school type and location",
          re.compile(r"\b(public|private|state|government)\b.{0,40}\bschool|\bschool.{0,40}\b(public|private|type)\b|"
                     r"\b(rural|urban|village|city|town|location|community size)\b", re.I),
          _all("SC013Q01TA", "SCHLTYPE", "PRIVATESCH", "SC001Q01TA"), instrument="sch_qqq",
          note="school-questionnaire variables: join students to schools on CNT + CNTSCHID "
               "(the app's stu_sch view). SC013Q01TA: 1 public, 2 private; SC001Q01TA: "
               "community size from 1 village to 6 megacity"),
    Topic("school resources and size",
          re.compile(r"school (size|resources)|class size|student-?teacher ratio|shortage|"
                     r"teacher shortage|teaching staff|educational material", re.I),
          {"2018": ("STAFFSHORT", "EDUSHORT", "PROATCE", "CLSIZE", "SCHSIZE", "STRATIO"),
           "2022": ("STAFFSHORT", "EDUSHORT", "PROATCE", "CLSIZE", "SCHSIZE", "STRATIO"),
           "2025": ("STAFFSHORT", "EDUSHORT", "PROATCE", "CLSIZE")}, instrument="sch_qqq",
          note="principal-reported (sch_qqq); SCHSIZE and STRATIO are not in the 2025 file"),
    Topic("ICT and digital devices",
          re.compile(r"\bict\b|\bdigital\b|\bcomputers?\b|\binternet\b|\btechnolog|\bdevices?\b|"
                     r"screen time|\bonline\b|\bsmartphones?\b", re.I),
          {"2018": ("ICTHOME", "ICTSCH", "ICTRES", "ENTUSE", "HOMESCH", "USESCH", "ICTCLASS"),
           "2022": ("ICTRES", "ICTHOME", "ICTSCH", "ICTAVSCH", "ICTOUT", "ICTEFFIC", "ICTDISTR"),
           "2025": ("ICTRES", "ICTHOME", "ICTSCH", "ICTAVHOME", "ICTAVSCH", "ICTOUT", "ICTDISTR", "AIUSESCH")},
          note="most ICT indices come from the optional ICT familiarity questionnaire, "
               "administered by a subset of economies in each cycle (the app's coverage "
               "notes say which); ICTRES (ICT resources at home) is in the core questionnaire"),
    Topic("AI use",
          re.compile(r"\b(ai|a\.i\.|artificial intelligence|chatgpt|chatbots?|generative ai)\b", re.I),
          {"2025": ("ST438Q01DA", "ST438Q02DA", "ST438Q03DA", "ST438Q04DA", "AIUSESCH", "IC170Q10DA")},
          note="2025 only: ST438Q01DA-Q04DA (how often students use AI chatbots for schoolwork, "
               "84 of 90 economies), AIUSESCH (WLE index), IC170Q10DA (ICT questionnaire, 44 "
               "economies). No PISA cycle has assessed AI literacy"),
    Topic("educational and career expectations",
          re.compile(r"expectations?|aspirations?|expected (education|occupation)|career|future job", re.I),
          {"2018": ("BSMJ",), "2022": ("EXPECEDU", "BSMJ", "SISCO"), "2025": ("EXPECEDU", "BSMJ", "SISCO")},
          note="BSMJ: expected occupational status (ISEI) at age 30; EXPECEDU: highest "
               "expected level of education; SISCO: has a clear idea about a future job"),
    Topic("truancy and lateness",
          re.compile(r"skip\w* (school|class)|truan|absentee|arriv\w* late|lateness|tardiness", re.I),
          {"2018": ("ST062Q01TA",), "2022": ("SKIPPING", "TARDYSD", "ST062Q01TA"),
           "2025": ("SKIPPING", "TARDYSD", "ST062Q01TA")},
          note="derived indices exist in 2022 and 2025; 2018 has the ST062 items only"),
    Topic("learning time (2018)",
          re.compile(r"learning time|instruction time|minutes per week|hours of (class|instruction|lessons)|"
                     r"class time|lesson time", re.I),
          {"2018": ("MMINS", "LMINS", "SMINS", "TMINS")},
          note="minutes per week in mathematics, test-language and science lessons, and in "
               "total (2018); 2022 asks homework time in the ST296 items instead"),
    Topic("proficiency levels",
          re.compile(r"proficien\w*|\blevel [1-6]\b|below level|baseline|top performers|low performers", re.I),
          {},
          note="there is no proficiency-level variable in the files: levels are cut from the "
               "plausible values at the OECD's fixed score thresholds (Level 2 begins at "
               "420.07 in mathematics, 407.47 in reading, 409.54 in science). Ask the app "
               "for a share (share of students below Level 2 in mathematics in Peru) and "
               "it applies the cutoffs to all ten PVs"),
]


def matching(question: str) -> list[Topic]:
    text = question or ""
    return [t for t in TOPICS if t.matches(text)]


def rows(topics: list[Topic], cycles=CYCLES, label_of=None) -> list[dict]:
    """One table row per (topic, code family) over the requested cycles:
    variable (displayed), label, tables, cycles, values, role, construct."""
    out, seen = [], set()
    for t in topics:
        per_code: dict[str, list[str]] = {}
        for cycle in cycles:
            for code in t.variables.get(cycle, ()):
                per_code.setdefault(code, []).append(cycle)
        for code, cyc in per_code.items():
            if (code, tuple(cyc)) in seen:
                continue
            seen.add((code, tuple(cyc)))
            members = expand(code)
            label = ""
            if label_of:
                label = label_of(members[0], cyc[-1]) or ""
                if len(members) > 1 and label:
                    label = re.sub(r"Plausible Value 1", f"Plausible values 1-{len(members)}", label)
                    label = re.sub(r"WEIGHTS? 1$", f"weights 1-{len(members)}", label, flags=re.I)
            out.append({"variable": display(code), "_code": code, "label": label,
                        "tables": ", ".join(f"{t.instrument}_{c}" for c in cyc),
                        "cycles": ", ".join(cyc),
                        "values": ("%d variables" % len(members)) if len(members) > 1 else "",
                        "role": "standard", "construct": t.construct})
    return out


def answer_lines(topics: list[Topic], cycles=CYCLES) -> list[str]:
    """One sentence per topic: the codes per cycle and the note."""
    lines = []
    for t in topics:
        parts = []
        by_cycle = {c: t.variables.get(c, ()) for c in cycles}
        present = {c: v for c, v in by_cycle.items() if v}
        if not present:
            if t.variables:
                have = sorted(t.variables)
                codes = "; ".join(display(c) for c in t.variables[have[-1]])
                lines.append(f"{t.construct}: not in PISA {', '.join(cycles)} — it exists in "
                             f"{', '.join(have)} ({codes}).")
            elif t.note:
                lines.append(f"{t.construct}: {t.note}.")
            continue
        if len(set(present.values())) == 1 and len(present) == len(cycles):
            codes = ", ".join(display(c) for c in next(iter(present.values())))
            parts.append(f"{codes} in every cycle" if len(cycles) > 1 else codes)
        else:
            for c, v in present.items():
                parts.append(f"{c}: {', '.join(display(x) for x in v)}")
            missing = [c for c, v in by_cycle.items() if not v]
            if missing:
                parts.append(f"nothing in {', '.join(missing)}")
        table = "" if t.instrument == "stu_qqq" else f" [{t.instrument} files]"
        lines.append(f"{t.construct}{table}: " + "; ".join(parts) + (f" — {t.note}" if t.note else "") + ".")
    return lines
