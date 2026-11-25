# -*- coding: utf-8 -*-
"""Map Web of Science C1 affiliation blocks to ISO-3 country codes."""
import re

# Canonical WoS country names and lowercase aliases mapped to ISO-3 codes
ALIAS = {
    "usa": "USA", "united states": "USA", "u s a": "USA",
    "brazil": "BRA", "france": "FRA", "india": "IND",
    "peoples r china": "CHN", "china": "CHN",
    "england": "GBR", "scotland": "GBR", "wales": "GBR",
    "northern ireland": "GBR", "great britain": "GBR", "united kingdom": "GBR",
    "germany": "DEU", "australia": "AUS", "italy": "ITA", "colombia": "COL",
    "mexico": "MEX", "thailand": "THA", "singapore": "SGP", "spain": "ESP",
    "malaysia": "MYS", "japan": "JPN", "netherlands": "NLD", "canada": "CAN",
    "belgium": "BEL", "indonesia": "IDN", "switzerland": "CHE", "kenya": "KEN",
    "saudi arabia": "SAU", "pakistan": "PAK", "argentina": "ARG", "sweden": "SWE",
    "iran": "IRN", "taiwan": "TWN", "south korea": "KOR", "korea": "KOR",
    "austria": "AUT", "bangladesh": "BGD", "south africa": "ZAF", "portugal": "PRT",
    "cameroon": "CMR", "tanzania": "TZA", "vietnam": "VNM", "greece": "GRC",
    "panama": "PAN", "peru": "PER", "senegal": "SEN", "ecuador": "ECU",
    "nigeria": "NGA", "finland": "FIN", "estonia": "EST", "gabon": "GAB",
    "ethiopia": "ETH", "venezuela": "VEN", "egypt": "EGY", "sri lanka": "LKA",
    "french guiana": "GUF", "cambodia": "KHM", "russia": "RUS",
    "israel": "ISR", "burkina faso": "BFA", "nicaragua": "NIC", "jamaica": "JAM",
    "honduras": "HND", "paraguay": "PRY", "ghana": "GHA", "new caledonia": "NCL",
    "turkey": "TUR", "dem rep congo": "COD", "democratic republic congo": "COD",
    "sudan": "SDN", "cuba": "CUB", "chile": "CHL", "denmark": "DNK",
    "norway": "NOR", "poland": "POL", "czech republic": "CZE", "czechia": "CZE",
    "philippines": "PHL", "morocco": "MAR", "algeria": "DZA", "tunisia": "TUN",
    "uganda": "UGA", "ivory coast": "CIV", "cote d ivoire": "CIV",
    "mali": "MLI", "nepal": "NPL", "ireland": "IRL", "new zealand": "NZL",
    "lebanon": "LBN", "jordan": "JOR", "kuwait": "KWT", "qatar": "QAT",
    "united arab emirates": "ARE", "u arab emirates": "ARE", "iraq": "IRQ",
    "malawi": "MWI", "mozambique": "MOZ", "madagascar": "MDG", "angola": "AGO",
    "bolivia": "BOL", "costa rica": "CRI", "guatemala": "GTM", "el salvador": "SLV",
    "dominican republic": "DOM", "haiti": "HTI", "trinidad and tobago": "TTO",
    "barbados": "BRB", "bahamas": "BHS", "suriname": "SUR", "guyana": "GUY",
    "belize": "BLZ", "uruguay": "URY", "ukraine": "UKR", "romania": "ROU",
    "hungary": "HUN", "bulgaria": "BGR", "croatia": "HRV", "serbia": "SRB",
    "slovakia": "SVK", "slovenia": "SVN", "lithuania": "LTU", "latvia": "LVA",
    "belarus": "BLR", "kazakhstan": "KAZ", "myanmar": "MMR", "laos": "LAO",
    "mongolia": "MNG", "zimbabwe": "ZWE", "zambia": "ZMB", "namibia": "NAM",
    "botswana": "BWA", "rwanda": "RWA", "benin": "BEN", "togo": "TGO",
    "niger": "NER", "chad": "TCD", "central african republic": "CAF",
    "central african rep": "CAF", "congo": "COG", "republic congo": "COG",
    "guinea": "GIN", "sierra leone": "SLE", "liberia": "LBR", "mauritania": "MRT",
    "gambia": "GMB", "somalia": "SOM", "libya": "LBY", "syria": "SYR",
    "afghanistan": "AFG", "yemen": "YEM", "oman": "OMN", "bahrain": "BHR",
    "reunion": "REU", "guadeloupe": "GLP", "martinique": "MTQ", "mayotte": "MYT",
    "puerto rico": "PRI", "french polynesia": "PYF", "papua new guinea": "PNG",
    "fiji": "FJI", "solomon islands": "SLB", "samoa": "WSM", "tonga": "TON",
    "vanuatu": "VUT", "cyprus": "CYP", "luxembourg": "LUX", "malta": "MLT",
    "iceland": "ISL", "north macedonia": "MKD", "macedonia": "MKD",
    "bosnia and herzegovina": "BIH", "bosnia herzeg": "BIH", "montenegro": "MNE",
    "albania": "ALB", "georgia": "GEO", "armenia": "ARM", "azerbaijan": "AZE",
    "sri lanka": "LKA", "south sudan": "SSD", "eritrea": "ERI", "djibouti": "DJI",
    "equatorial guinea": "GNQ", "eswatini": "SWZ", "swaziland": "SWZ",
    "lesotho": "LSO", "burundi": "BDI",
    "cent afr republ": "CAF", "cent african republ": "CAF",
    "cote ivoire": "CIV", "czechoslovakia": "CZE", "ussr": "RUS",
    "trinidad tobago": "TTO", "turkiye": "TUR",
    "grenada": "GRD", "mauritius": "MUS", "dominican rep": "DOM",
    "sarajevo": "BIH",
}

# Sort aliases longest-first to avoid partial substring matches
_KEYS = sorted(ALIAS, key=len, reverse=True)
_US_RE = re.compile(r"\bUSA\b")
_ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")


def _clean_tail(seg):
    tail = seg.split(",")[-1].strip()
    tail = _ZIP_RE.sub("", tail)
    tail = " ".join(t for t in tail.split() if t.isalpha())
    return tail.lower()


def _match_in(text):
    for k in _KEYS:
        if re.search(r"\b" + re.escape(k) + r"\b", text):
            return ALIAS[k]
    return None


def parse_countries(c1):
    """Return the deduplicated set of ISO-3 country codes for a record."""
    found = set()
    for seg in c1.split(";"):
        seg = seg.strip()
        if seg.startswith("["):
            rb = seg.find("]")
            if rb != -1:
                seg = seg[rb + 1:].strip()
        if not seg:
            continue
        low = seg.lower()
        if _US_RE.search(seg):
            found.add("USA")
            continue
        tail = _clean_tail(seg)
        iso = ALIAS.get(tail)
        if iso is None:
            iso = _match_in(low)
        if iso:
            found.add(iso)
    return found
