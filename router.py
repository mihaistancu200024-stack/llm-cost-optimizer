import re

EXPANSION_RULES = {
    "ocean_view": (
        r"\b(ocean\s*view|ocean|sea\s*view)\b",
        ["waterfront", "beachfront", "seaview", "ocean view"],
    ),
    "penthouse": (
        r"\bpenthouse\b",
        ["top floor", "rooftop", "luxury penthouse"],
    ),
    "pool": (
        r"\bpool\b",
        ["swimming pool", "heated pool"],
    ),
    "gym": (
        r"\b(gym|fitness)\b",
        ["fitness center", "gym", "workout room"],
    ),
    "downtown": (
        r"\b(downtown|city\s*center)\b",
        ["urban", "city", "central"],
    ),
    "quiet": (
        r"\b(quiet|peaceful)\b",
        ["suburban", "cul-de-sac", "low traffic"],
    ),
    "parking": (
        r"\b(parking|garage)\b",
        ["covered parking", "garage", "carport"],
    ),
    "modern": (
        r"\b(new|modern)\b",
        ["newly built", "renovated", "contemporary"],
    ),
    "pet": (
        r"\b(pet|dog|cat)\b",
        ["pet-friendly", "fenced yard", "dog park"],
    ),
    "spacious": (
        r"\b(large|spacious)\b",
        ["open floor plan", "high ceilings"],
    ),
}

COMPLEX_KEYWORDS = re.compile(
    r"\b(compare|vs|difference|negotiate|investment|roi|rental\s*income|cap\s*rate)\b",
    re.IGNORECASE,
)


def expand_query(query: str) -> tuple[str, list[str]]:
    extra_terms: list[str] = []
    fired_rules: list[str] = []

    for rule_name, (pattern, keywords) in EXPANSION_RULES.items():
        if re.search(pattern, query, re.IGNORECASE):
            fired_rules.append(rule_name)
            extra_terms.extend(keywords)

    unique_extras = list(dict.fromkeys(extra_terms))
    expanded = query if not unique_extras else query + " " + " ".join(unique_extras)
    return expanded, fired_rules


def is_complex_query(query: str) -> bool:
    if len(query.split()) > 8:
        return True
    return bool(COMPLEX_KEYWORDS.search(query))
