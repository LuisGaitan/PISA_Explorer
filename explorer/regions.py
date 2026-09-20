"""Fixed geographic and institutional groupings of PISA economies.

PISA has no region variable, and asking a language model to enumerate "Latin
American countries" produced a different, incomplete list each time. Here the
membership is fixed in code; the planner names a region and `expand()` turns
it into the exact CNT codes present in each cycle's table, so a region means
the same thing in every answer and never includes an economy that is not in
the data.

Groupings follow common usage (World Bank / UN geoscheme) restricted to
economies that have taken part in PISA 2018, 2022 or 2025. An economy may sit
in more than one region (Mexico is Latin American and North American;
Türkiye is European and Middle Eastern). "OECD" comes from the data's own
OECD flag, not from this file.
"""

import re

REGIONS: dict[str, list[str]] = {
    "Latin America and the Caribbean": [
        "ARG", "BRA", "CHL", "COL", "CRI", "DOM", "ECU", "GTM", "JAM", "MEX",
        "PAN", "PER", "PRY", "SLV", "URY"],
    "South America": ["ARG", "BRA", "CHL", "COL", "ECU", "PER", "PRY", "URY"],
    "Central America": ["CRI", "GTM", "PAN", "SLV"],
    "Caribbean": ["DOM", "JAM"],
    "North America": ["CAN", "MEX", "USA"],
    "East Asia": ["HKG", "JPN", "KOR", "MAC", "MNG", "QCI", "TAP"],
    "Southeast Asia": ["BRN", "IDN", "KHM", "MYS", "PHL", "SGP", "THA", "VNM"],
    "Central Asia": ["KAZ", "KGZ", "QTJ", "UZB"],
    "Caucasus": ["ARM", "AZE", "GEO", "QAZ"],
    "Middle East": ["ARE", "ISR", "JOR", "LBN", "PSE", "QAT", "QKI", "SAU", "TUR"],
    "North Africa": ["MAR"],
    "Sub-Saharan Africa": ["KEN", "MUS", "RWA", "ZMB"],
    "Europe": [
        "ALB", "AUT", "BEL", "BGR", "BIH", "BLR", "CHE", "CZE", "DEU", "DNK",
        "ESP", "EST", "FIN", "FRA", "GBR", "GRC", "HRV", "HUN", "IRL", "ISL",
        "ITA", "KSV", "LTU", "LUX", "LVA", "MDA", "MKD", "MLT", "MNE", "NLD",
        "NOR", "POL", "PRT", "QMR", "QRT", "QUA", "QUR", "ROU", "RUS", "SRB",
        "SVK", "SVN", "SWE", "TUR", "UKR"],
    "European Union": [
        "AUT", "BEL", "BGR", "CZE", "DEU", "DNK", "ESP", "EST", "FIN", "FRA",
        "GRC", "HRV", "HUN", "IRL", "ITA", "LTU", "LUX", "LVA", "MLT", "NLD",
        "POL", "PRT", "ROU", "SVK", "SVN", "SWE"],
    "Nordic countries": ["DNK", "FIN", "ISL", "NOR", "SWE"],
    "Baltic states": ["EST", "LTU", "LVA"],
    "Western Balkans": ["ALB", "BIH", "KSV", "MKD", "MNE", "SRB"],
    "Oceania": ["AUS", "NZL"],
}
# Unions expressed through the parts above.
REGIONS["Asia"] = sorted(set(REGIONS["East Asia"] + REGIONS["Southeast Asia"]
                             + REGIONS["Central Asia"] + REGIONS["Caucasus"]
                             + REGIONS["Middle East"]))
REGIONS["Middle East and North Africa"] = sorted(set(REGIONS["Middle East"]
                                                     + REGIONS["North Africa"]))
REGIONS["Africa"] = sorted(set(REGIONS["North Africa"] + REGIONS["Sub-Saharan Africa"]))
REGIONS["Americas"] = sorted(set(REGIONS["Latin America and the Caribbean"]
                                 + REGIONS["North America"]))

ALIASES: dict[str, str] = {
    "latin america": "Latin America and the Caribbean",
    "latin american": "Latin America and the Caribbean",
    "latin america and caribbean": "Latin America and the Caribbean",
    "latam": "Latin America and the Caribbean",
    "lac": "Latin America and the Caribbean",
    "eu": "European Union",
    "european union": "European Union",
    "mena": "Middle East and North Africa",
    "middle east": "Middle East",
    "gulf": "Middle East",
    "nordic": "Nordic countries",
    "nordics": "Nordic countries",
    "scandinavia": "Nordic countries",
    "baltic": "Baltic states",
    "baltics": "Baltic states",
    "balkans": "Western Balkans",
    "western balkans": "Western Balkans",
    "east asian": "East Asia",
    "southeast asian": "Southeast Asia",
    "south east asia": "Southeast Asia",
    "central asian": "Central Asia",
    "sub saharan africa": "Sub-Saharan Africa",
    "subsaharan africa": "Sub-Saharan Africa",
    "australia and new zealand": "Oceania",
}


# Common names that differ from the OECD label in the codebook ("Chinese
# Taipei", "Türkiye", "Korea", "Macao (China)"), so a question written the
# everyday way still resolves to the economy that IS in the data.
ECONOMY_ALIASES: dict[str, list[str]] = {
    "TAP": ["taiwan", "chinese taipei", "taipei"],
    "TUR": ["turkey", "turkiye", "türkiye"],
    "KOR": ["south korea", "korea", "republic of korea"],
    "ARE": ["uae", "united arab emirates", "emirates"],
    "GBR": ["uk", "united kingdom", "britain", "great britain", "england", "scotland", "wales"],
    "USA": ["usa", "united states"],
    "VNM": ["vietnam", "viet nam"],
    "MAC": ["macau", "macao"],
    "HKG": ["hong kong"],
    "QCI": ["b s j z", "bsjz", "beijing", "shanghai", "jiangsu", "zhejiang"],
    "PSE": ["palestine", "palestinian authority", "palestinian"],
    "KSV": ["kosovo"],
    "MKD": ["north macedonia", "macedonia"],
    "MDA": ["moldova"],
    "CZE": ["czechia", "czech republic"],
    "SVK": ["slovakia", "slovak republic"],
    "NLD": ["netherlands", "holland"],
    "BRN": ["brunei"],
    "QKI": ["kurdistan", "iraq"],
    "QTJ": ["dushanbe", "tajikistan"],
    "QUA": ["ukraine", "ukrainian regions"],
    "QUR": ["ukraine", "ukrainian regions"],
    "UKR": ["ukraine"],
    "QAZ": ["baku"],
    "QMR": ["moscow"],
    "QRT": ["tatarstan"],
    "RUS": ["russia", "russian federation"],
    "DOM": ["dominican republic"],
    "CRI": ["costa rica"],
    "SLV": ["el salvador"],
    "BIH": ["bosnia", "bosnia and herzegovina"],
    "IDN": ["indonesia"],
    "SAU": ["saudi arabia", "saudi"],
}


# Countries people ask about that have never taken part in PISA 2018, 2022 or
# 2025 (or only as a sub-national region held under another code). Naming
# one gets the fixed "not in the databases" answer instead of a planner that
# invents a code. China and Ukraine are handled through their region codes.
NON_PISA: dict[str, str] = {
    "india": "India", "pakistan": "Pakistan", "bangladesh": "Bangladesh", "nigeria": "Nigeria",
    "egypt": "Egypt", "ethiopia": "Ethiopia", "south africa": "South Africa", "iran": "Iran",
    "venezuela": "Venezuela", "bolivia": "Bolivia", "cuba": "Cuba", "nepal": "Nepal",
    "sri lanka": "Sri Lanka", "myanmar": "Myanmar", "ghana": "Ghana", "tanzania": "Tanzania",
    "uganda": "Uganda", "angola": "Angola", "algeria": "Algeria", "tunisia": "Tunisia",
    "libya": "Libya", "sudan": "Sudan", "afghanistan": "Afghanistan", "syria": "Syria",
    "yemen": "Yemen", "kuwait": "Kuwait", "bahrain": "Bahrain", "oman": "Oman",
    "cyprus": "Cyprus", "liechtenstein": "Liechtenstein", "puerto rico": "Puerto Rico",
    "haiti": "Haiti", "honduras": "Honduras", "nicaragua": "Nicaragua", "laos": "Laos",
    "mozambique": "Mozambique", "senegal": "Senegal", "zimbabwe": "Zimbabwe",
    "cameroon": "Cameroon", "ivory coast": "Côte d'Ivoire", "mongolia 2018": "Mongolia (before 2022)",
}


def non_pisa_named(text: str) -> list[str]:
    """Display names of NON_PISA countries named in the text."""
    low = " " + re.sub(r"[^a-z ]", " ", (text or "").lower()) + " "
    low = re.sub(r"\s+", " ", low)
    return [name for key, name in NON_PISA.items() if f" {key} " in low]


# Economies whose 2018-only codes carry no value label in the SAS release.
NAME_FALLBACK: dict[str, str] = {
    "BIH": "Bosnia and Herzegovina", "BLR": "Belarus", "RUS": "Russian Federation",
    "UKR": "Ukraine", "QMR": "Moscow region (Russia)", "QRT": "Tatarstan (Russia)",
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z ]", " ", name.lower()).replace("  ", " ").strip()


def canonical(name: str) -> str | None:
    """Resolve a user/planner region name to a REGIONS key, or None."""
    key = _norm(name)
    if key in ALIASES:
        return ALIASES[key]
    for region in REGIONS:
        if _norm(region) == key:
            return region
    for region in REGIONS:            # "Latin America" inside the full name
        if key and key in _norm(region):
            return region
    return None


def expand(names: list[str], present: dict[str, set[str]]) -> dict[str, dict]:
    """Per cycle, the region members present in that cycle's data.

    Returns {cycle: {"codes": [..present..], "absent": [..members not in
    that cycle..]}}; raises ValueError for an unknown region name."""
    members: set[str] = set()
    for name in names:
        region = canonical(name)
        if region is None:
            raise ValueError(f"unknown region {name!r}; known regions: "
                             + ", ".join(sorted(REGIONS)))
        members |= set(REGIONS[region])
    out = {}
    for cycle, codes in present.items():
        out[cycle] = {"codes": sorted(members & codes),
                      "absent": sorted(members - codes)}
    return out


def prompt_block(present: dict[str, set[str]]) -> str:
    """The region list the planner is shown, with per-cycle coverage."""
    lines = []
    for region, members in REGIONS.items():
        cov = []
        for cycle, codes in present.items():
            n = len(set(members) & codes)
            if n:
                cov.append(f"{cycle}: {n}")
        lines.append(f"- {region} ({len(members)} economies; in data — "
                     + ", ".join(cov) + ")")
    return "\n".join(lines)
