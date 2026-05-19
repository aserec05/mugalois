import json
from mugalois.core.types import Triple
 
 
def json_to_triples(response_text: str) -> set[Triple]:
    """Parse the LLM JSON -> Triple objects.
 
    Expected format: {"triples": [{"s": "...", "p": "...", "o": "..."}]}
    Returns an empty set on any malformed or unexpected response.
    """
    try:
        data = json.loads(response_text)
        return {
            Triple(t["s"], t["p"], t["o"])
            for t in data.get("triples", [])
        }
    except (AttributeError, json.JSONDecodeError, KeyError, TypeError):
        return set()
