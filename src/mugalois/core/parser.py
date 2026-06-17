import json
import re
from mugalois.core.types import Triple


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
            }
            if triples:
                return triples
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        pass

    triples = set()
    pattern = r'\{\s*"s"\s*:\s*"([^"]+)"\s*,\s*"p"\s*:\s*"([^"]+)"\s*,\s*"o"\s*:\s*"([^"]+)"\s*\}'
    for match in re.finditer(pattern, text):
        triples.add(Triple(match.group(1), match.group(2), match.group(3)))
    return triples


def json_to_values(response_text: str) -> set[str]:
    """Parse LLM JSON response into a set of string values.

    Expected: {"values": ["...", "..."]}

    Strategy:
    1. Try clean JSON parse
    2. Try bare array
    3. Fallback: regex extract all quoted strings
       (handles truncated JSON from token limits)
    """
    text = response_text.strip()

    # 1. clean JSON parse
    try:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > 0:
            data = json.loads(text[start:end])
            if "values" in data:
                return {str(v) for v in data["values"] if v}
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        pass

    # 2. bare array
    try:
        start = text.find("[")
        end   = text.rfind("]") + 1
        if start != -1 and end > 0:
            data = json.loads(text[start:end])
            if isinstance(data, list):
                return {str(v) for v in data if isinstance(v, str) and v}
    except (json.JSONDecodeError, TypeError):
        pass

    # 3. regex fallback — handles truncated JSON
    # extract all quoted strings, exclude JSON keys
    JSON_KEYS = {"values", "triples", "s", "p", "o"}
    candidates = re.findall(r'"([^"]+)"', text)
    result = {v for v in candidates if v not in JSON_KEYS and len(v) > 1}
    return result