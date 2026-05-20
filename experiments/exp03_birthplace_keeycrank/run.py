import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mugalois.core.types import TriplePattern, Environment
from mugalois.scans.key_crank import LLMKeyCrank
from mugalois.llm.llm_client import OllamaClient

PREDICATE = "schema:birthPlace"
PATTERN   = TriplePattern("?person", PREDICATE, "?place")
MODEL     = os.getenv("OLLAMA_MODEL", "phi3")
BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

SEEDS_PERSON = {"Albert Einstein", "Marie Curie", "Napoleon Bonaparte"}
SEEDS_PLACE  = {"Ulm", "Warsaw", "Paris"}


def run_case(label, env, seeds_person, seeds_place):
    print(f"\n{'='*50}")
    print(f"Case: {label}")
    if seeds_person:
        print(f"  Seeds ?person : {sorted(seeds_person)}")
    if seeds_place:
        print(f"  Seeds ?place  : {sorted(seeds_place)}")

    llm = OllamaClient(model=MODEL, base_url=BASE_URL)
    T = LLMKeyCrank(PATTERN, env, llm=llm)

    print(f"\nTriples found: {len(T)}")
    for t in sorted(T, key=lambda x: x.s):
        print(f"  {t}")

    out_of_seeds = []
    if seeds_person and seeds_place:
        out_of_seeds = [
            t for t in T
            if t.s not in seeds_person or t.o not in seeds_place
        ]
        if out_of_seeds:
            print(f"\nWarning — triples outside seeds ({len(out_of_seeds)}):")
            for t in out_of_seeds:
                print(f"  {t}")
        else:
            print("\nAll triples respect the seed constraints.")

    return {
        "label":        label,
        "n_triples":    len(T),
        "triples":      [{"s": t.s, "p": t.p, "o": t.o} for t in T],
        "out_of_seeds": [{"s": t.s, "p": t.p, "o": t.o} for t in out_of_seeds],
    }


if __name__ == "__main__":
    print(f"Running KeyCrank on {PREDICATE} with {MODEL}...")
    results = []

    # Cas A — seeds left
    env_a = Environment()
    env_a.set("?person", SEEDS_PERSON)
    results.append(run_case("L->R (seeds on subject side)", env_a, SEEDS_PERSON, set()))

    # Cas B — seeds right
    env_b = Environment()
    env_b.set("?place", SEEDS_PLACE)
    results.append(run_case("R->L (seeds on object side)", env_b, set(), SEEDS_PLACE))

    # Cas C — two seeds
    env_c = Environment()
    env_c.set("?person", SEEDS_PERSON)
    env_c.set("?place", SEEDS_PLACE)
    results.append(run_case("Both sides (direction by cardinality)", env_c, SEEDS_PERSON, SEEDS_PLACE))

    out = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out, "w") as f:
        json.dump({
            "predicate": PREDICATE,
            "model":     MODEL,
            "results":   results,
        }, f, indent=2)
    print(f"\nResults saved to {out}")