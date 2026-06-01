import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mugalois.core.types import TriplePattern, Environment
from mugalois.scans.table_scan import LLMTableScan
from mugalois.scans.seed_crank import LLMSeedCrank
from mugalois.scans.key_crank import LLMKeyCrank
from mugalois.scans.llm_triplet_scan import LLMTripletScan
from mugalois.llm.llm_client import AzureOpenAIClient

MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini-2")
TAU   = 0.7

PREDICATE    = "schema:birthPlace"
PATTERN      = TriplePattern("?person", PREDICATE, "?place")

SEEDS_PERSON = {
    "Albert Einstein", "Marie Curie", "Napoleon Bonaparte",
    "Leonardo da Vinci", "Isaac Newton", "Charles Darwin",
    "Galileo Galilei", "Nikola Tesla", "Louis Pasteur", "Ada Lovelace"
}


def run_strategy(label, fn, pattern, env, llm):
    print(f"\n{'='*55}")
    print(f"Strategy: {label}")
    T = fn(pattern, env, llm)

    print(f"\nTriples found: {len(T)}")
    for t in sorted(T, key=lambda x: x.s):
        print(f"  {t}")

    return {
        "strategy":  label,
        "n_triples": len(T),
        "triples":   [{"s": t.s, "p": t.p, "o": t.o} for t in T],
    }


if __name__ == "__main__":
    print(f"Experiment 04 — Strategy Comparison on {PREDICATE}")
    print(f"Model  : {MODEL}")
    print(f"Seeds  : {sorted(SEEDS_PERSON)}")

    results = []

    # A — TableScan : aucun seed, le LLM liste librement
    env_a = Environment()
    results.append(run_strategy(
        "TableScan (no seeds)",
        lambda p, e, l: LLMTableScan(p, e, l, max_iter=3),
        PATTERN, env_a, AzureOpenAIClient()
    ))

    # B — SeedCrank : seeds sujet, un seul appel avec tous les seeds
    env_b = Environment()
    env_b.set("?person", SEEDS_PERSON)
    results.append(run_strategy(
        "SeedCrank (subject seeds, single call)",
        lambda p, e, l: LLMSeedCrank(p, e, l, max_iter=3),
        PATTERN, env_b, AzureOpenAIClient()
    ))

    # C — KeyCrank : seeds sujet, un appel par seed
    env_c = Environment()
    env_c.set("?person", SEEDS_PERSON)
    results.append(run_strategy(
        "KeyCrank (subject seeds, one call per seed)",
        lambda p, e, l: LLMKeyCrank(p, e, l),
        PATTERN, env_c, AzureOpenAIClient()
    ))

    # D — TripletScan : dispatcher automatique
    env_d = Environment()
    env_d.set("?person", SEEDS_PERSON)
    results.append(run_strategy(
        "TripletScan (automatic routing)",
        lambda p, e, l: LLMTripletScan(p, e, l, tau_strategie=TAU),
        PATTERN, env_d, AzureOpenAIClient()
    ))

    print(f"\n{'='*55}")
    print(f"Summary:")
    for r in results:
        print(f"  {r['strategy']:<45} {r['n_triples']} triples")

    out = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out, "w") as f:
        json.dump({
            "predicate": PREDICATE,
            "model":     MODEL,
            "results":   results,
        }, f, indent=2)
    print(f"\nResults saved to {out}")