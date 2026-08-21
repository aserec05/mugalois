# src/mugalois/core/parser.py
import json
import re
from mugalois.core.types import Triple

JUNK_MARKERS = {
    "null", "none", "n/a", "na", "unknown", "tbd", "tba",
    "to be determined", "to be announced", "pending", "",
}

_SAINT_PREFIX = re.compile(r"^(Saint|St\.?)\s+", re.IGNORECASE)


def _normalize_name(v: str) -> str:
    """Normalize entity name variants for consistent matching."""
    # "Saint X" / "St. X" -> "Pope X"
    v = _SAINT_PREFIX.sub("Pope ", v.strip())
    # Strip parenthetical suffixes: "Anacletus (Cletus)" -> "Anacletus"
    v = re.sub(r"\s*\(.*?\)\s*$", "", v).strip()
    return v


def _is_real_value(v) -> bool:
    """True if v is a non-junk string value worth keeping."""
    if not isinstance(v, str):
        return False
    s = v.strip()
    if not s:
        return False
    if s.lower() in JUNK_MARKERS:
        return False
    return True


def json_to_triples(response_text: str) -> set[Triple]:
    """Parse LLM JSON response into a set of Triple objects.

    Expected: {"triples": [{"s": "...", "p": "...", "o": "..."}]}
    Falls back to regex for small models with malformed output.
    """
    text = response_text.strip()
    try:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > 0:
            data = json.loads(text[start:end])
            triples = {
                Triple(t["s"], t["p"], t["o"])
                for t in data.get("triples", [])
                if _is_real_value(t.get("s")) and _is_real_value(t.get("o"))
            }
            if triples:
                return triples
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        pass

    triples = set()
    pattern = r'\{\s*"s"\s*:\s*"([^"]+)"\s*,\s*"p"\s*:\s*"([^"]+)"\s*,\s*"o"\s*:\s*"([^"]+)"\s*\}'
    for match in re.finditer(pattern, text):
        s, p, o = match.group(1), match.group(2), match.group(3)
        if _is_real_value(s) and _is_real_value(o):
            triples.add(Triple(s, p, o))
    return triples


def json_to_values(response_text: str) -> set[str]:
    """Parse LLM JSON response into a set of string values.

    Expected: {"values": ["...", "..."]}
    Strategy:
    1. Try clean JSON parse
    2. Try bare array
    3. Fallback: regex extract all quoted strings

    Junk/placeholder values (e.g. "None", "null", "") are filtered out
    in every branch. "Saint X" / "St. X" are normalized to "Pope X".
    """
    text = response_text.strip()

    # 1. clean JSON parse
    try:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > 0:
            data = json.loads(text[start:end])
            if "values" in data:
                vals = data["values"]
                if isinstance(vals, dict):
                    vals = list(vals.values())
                if isinstance(vals, list):
                    return {_normalize_name(str(v)) for v in vals if _is_real_value(v)}
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        pass

    # 2. bare array
    try:
        start = text.find("[")
        end   = text.rfind("]") + 1
        if start != -1 and end > 0:
            data = json.loads(text[start:end])
            if isinstance(data, list):
                return {_normalize_name(str(v)) for v in data if _is_real_value(v)}
    except (json.JSONDecodeError, TypeError):
        pass

    # 3. regex fallback — handles truncated JSON
    JSON_KEYS = {"values", "triples", "s", "p", "o"}
    candidates = re.findall(r'"([^"]+)"', text)
    result = {
        _normalize_name(v) for v in candidates
        if v not in JSON_KEYS and len(v) > 1 and _is_real_value(v)
    }
    return result