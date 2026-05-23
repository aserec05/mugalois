import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mugalois.core.types import TriplePattern, Environment
from mugalois.scans.table_scan import LLMTableScan
from mugalois.llm.llm_client import AzureOpenAIClient

PREDICATE = "schema:capital"
PATTERN   = TriplePattern("?country", PREDICATE, "?capital")
MODEL     = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini-2")
MAX_ITER  = 20

if __name__ == "__main__":
    llm = AzureOpenAIClient()
    env = Environment()

    print(f"Running TableScan on {PREDICATE} with {MODEL}...")
    T = LLMTableScan(PATTERN, env, llm=llm, max_iter=MAX_ITER)

    print(f"\nTriples found: {len(T)}")
    for t in sorted(T, key=lambda x: x.s):
        print(f"  {t}")

    print(f"\nEnvironment after:")
    print(f"  ?country : {sorted(env.get('?country'))}")
    print(f"  ?capital : {sorted(env.get('?capital'))}")

    results = {
        "predicate": PREDICATE,
        "model":     MODEL,
        "max_iter":  MAX_ITER,
        "n_triples": len(T),
        "triples":   [{"s": t.s, "p": t.p, "o": t.o} for t in T],
    }

    out = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")