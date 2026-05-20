import json
import os
import sys
 
from dotenv import load_dotenv
load_dotenv()
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
 
from mugalois.core.types import TriplePattern, Environment
from mugalois.scans.seed_crank import LLMSeedCrank
from mugalois.llm.llm_client import OllamaClient
 
PREDICATE = "schema:spouse"
PATTERN   = TriplePattern("?person", PREDICATE, "?spouse")
MODEL     = os.getenv("OLLAMA_MODEL", "phi3")
BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MAX_ITER  = 2
 
SEEDS_PERSON = {"Elvis Presley", "John Lennon", "Paul McCartney"}
SEEDS_SPOUSE = {"Priscilla Presley", "Yoko Ono", "Linda Eastman"}
 
if __name__ == "__main__":
    llm = OllamaClient(model=MODEL, base_url=BASE_URL)
    env = Environment()
    env.set("?person", SEEDS_PERSON)
    env.set("?spouse", SEEDS_SPOUSE)
 
    print(f"Running SeedCrank on {PREDICATE} with {MODEL}...")
    print(f"  Seeds ?person : {sorted(SEEDS_PERSON)}")
    print(f"  Seeds ?spouse : {sorted(SEEDS_SPOUSE)}")
 
    T = LLMSeedCrank(PATTERN, env, llm=llm, max_iter=MAX_ITER)
 
    print(f"\nTriples found: {len(T)}")
    for t in sorted(T, key=lambda x: x.s):
        print(f"  {t}")
 
    out_of_seeds = [
        t for t in T
        if t.s not in SEEDS_PERSON or t.o not in SEEDS_SPOUSE
    ]
    if out_of_seeds:
        print(f"\nWarning — triples outside seeds ({len(out_of_seeds)}):")
        for t in out_of_seeds:
            print(f"  {t}")
    else:
        print("\nAll triples respect the seed constraints.")
 
    results = {
        "predicate":    PREDICATE,
        "model":        MODEL,
        "max_iter":     MAX_ITER,
        "seeds_person": sorted(SEEDS_PERSON),
        "seeds_spouse": sorted(SEEDS_SPOUSE),
        "n_triples":    len(T),
        "triples":      [{"s": t.s, "p": t.p, "o": t.o} for t in T],
        "out_of_seeds": [{"s": t.s, "p": t.p, "o": t.o} for t in out_of_seeds],
    }
 
    out = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")
