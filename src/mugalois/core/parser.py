import json
import re
from mugalois.core.types import Triple


def json_to_triples(response_text: str) -> set[Triple]:
    """Parse the LLM JSON response into a set of Triple objects.

    First attempts clean JSON parsing, then falls back to regex extraction
    to handle small models that add prefixes or malformed content.
    Expected format: {"triples": [{"s": "...", "p": "...", "o": "..."}]}
    """
    text = response_text.strip()

    try:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > 0:
            data = json.loads(text[start:end])
            return {
                Triple(t["s"], t["p"], t["o"])
                for t in data.get("triples", [])
            }
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        pass

    # temporaly with the little models
    triples = set()
    pattern = r'\{\s*"s"\s*:\s*"([^"]+)"\s*,\s*"p"\s*:\s*"([^"]+)"\s*,\s*"o"\s*:\s*"([^"]+)"\s*\}'
    for match in re.finditer(pattern, text):
        triples.add(Triple(match.group(1), match.group(2), match.group(3)))
    return triples