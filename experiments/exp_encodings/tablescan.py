"""
experiments/exp_tablescan_encoding/run.py
==========================================

Research question
-----------------
Does the prompt encoding affect TableScan performance ?

Three encodings compared on two predicates, same GT, same LLM :
- SPARQL   : SELECT ?x ?y WHERE { ?x p ?y }
- Pattern  : (?x, p, ?y)
- Current  : NL-style "List ALL triples..."

Predicates :
- dbo:spouse    — 43 triples  (lightweight)
- dbo:deathPlace — 376 triples (medium)

Run
---
    python3 -m experiments.exp_tablescan_encoding.run
"""

import os
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mugalois.core.types import TriplePattern, Environment
from mugalois.scans.table_scan import LLMTableScan
from mugalois.llm.llm_client import AzureOpenAIClient

from evaluation.dataset import ScientistDataset
from evaluation.evaluator import Evaluator
from evaluation.report import Report

# ─── Config ───────────────────────────────────────────────────────────────────

MODEL    = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini-2")
MAX_ITER = 3

GT_PATH     = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "ground_truth", "ground_truth.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

ds = ScientistDataset(GT_PATH)

PREDICATES = [
    {
        "uri":     "http://dbpedia.org/ontology/spouse",
        "label":   "dbo:spouse",
        "pattern": TriplePattern("?scientist", "dbo:spouse", "?spouse"),
        "name":    "spouse",
    },
    {
        "uri":     "http://dbpedia.org/ontology/deathPlace",
        "label":   "dbo:deathPlace",
        "pattern": TriplePattern("?scientist", "dbo:deathPlace", "?place"),
        "name":    "deathplace",
    },
]

# ─── Main ─────────────────────────────────────────────────────────────────────

def run_pred(pred_cfg, llm):
    expected = ds.T3(pred_cfg["uri"])
    pattern  = pred_cfg["pattern"]
    label    = pred_cfg["label"]

    print(f"\n{'─'*60}")
    print(f"  Predicate : {label}  ({len(expected)} triples)")
    print(f"{'─'*60}")

    results = {}
    for encoding in ["sparql", "pattern", "current"]:
        print(f"  TableScan {encoding} ...")
        env = Environment()
        triples = LLMTableScan(pattern, env, llm, max_iter=MAX_ITER, encoding=encoding)
        print(f"  → {len(triples)} triples")
        results[f"TableScan {encoding}"] = triples

    report = Report(predicate=f"{label} — TableScan encoding")
    for strat, triples in results.items():
        report.add(strat, Evaluator(triples, expected).evaluate())
    report.print_table()
    return report


def main():
    print(f"\n{'='*60}")
    print(f"  Experiment — TableScan encoding comparison")
    print(f"  Model : {MODEL}  |  max_iter={MAX_ITER}")
    print(f"{'='*60}")

    llm = AzureOpenAIClient()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for pred_cfg in PREDICATES:
        rep = run_pred(pred_cfg, llm)
        rep.save_json(os.path.join(RESULTS_DIR, f"tablescan_encoding_{pred_cfg['name']}.json"))
        rep.save_csv(os.path.join(RESULTS_DIR,  f"tablescan_encoding_{pred_cfg['name']}.csv"))


if __name__ == "__main__":
    main()