"""
experiments/exp_t3_birthplace/run.py
=====================================

Research question
-----------------
Pattern T3 : (?x, dbo:birthPlace, ?y) — both variables free

Experiments
-----------
A — no seed
B — seed ?x small  (10 scientists)
C — seed ?x large  (100 scientists)
D — seed ?y small  (10 places)
E — seed ?y large  (100 places)
F — seed ?x + ?y   many-to-one  (several scientists → same place)
G — seed ?x + ?y   one-to-many  (one scientist → several places)
H — seed ?x + ?y   large both

Ground truth : T3["dbo:birthPlace"] = 684 triples

Run
---
    python3 -m experiments.exp_t3_birthplace.run
"""

import os
import sys
import random

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mugalois.core.types import Triple, TriplePattern, Environment
from mugalois.core.parser import json_to_triples
from mugalois.core.prompts import build_messages
from mugalois.core.operators import filter_by_subject, filter_by_object
from mugalois.scans.table_scan import LLMTableScan
from mugalois.scans.seed_crank import LLMSeedCrank
from mugalois.scans.key_crank import LLMKeyCrank
from mugalois.scans.llm_triplet_scan import LLMTripletScan
from mugalois.llm.llm_client import AzureOpenAIClient

from evaluation.dataset import ScientistDataset
from evaluation.evaluator import Evaluator
from evaluation.report import Report

# ─── Config ───────────────────────────────────────────────────────────────────

PRED_URI = "http://dbpedia.org/ontology/birthPlace"
MODEL    = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini-2")
TAU      = 0.7
MAX_ITER = 10

GT_PATH     = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "ground_truth", "ground_truth.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ─── Ground truth ─────────────────────────────────────────────────────────────

ds           = ScientistDataset(GT_PATH)
EXPECTED     = ds.T3(PRED_URI)
ALL_SUBJECTS = ds.subjects(PRED_URI)
ALL_OBJECTS  = ds.objects(PRED_URI)

random.seed(42)

# Seeds ?x
SX_SMALL = set(random.sample(sorted(ALL_SUBJECTS), 10))
SX_LARGE = set(random.sample(sorted(ALL_SUBJECTS), 100))

# Seeds ?y
SY_SMALL = set(random.sample(sorted(ALL_OBJECTS), 10))
SY_LARGE = set(random.sample(sorted(ALL_OBJECTS), 100))

# Exp F — seed ?x=1, seed ?y=n  (one scientist, many place candidates)
SX_F = {"Albert Einstein"}
SY_F = SY_LARGE

# Exp G — seed ?x=n, seed ?y=1  (many scientists, one place candidate)
SX_G = SX_LARGE
SY_G = {"Ulm"}

# ─── Helpers ──────────────────────────────────────────────────────────────────

PATTERN = TriplePattern("?scientist", "dbo:birthPlace", "?place")


def ref_s(seeds): return {Triple(s, "p", "x") for s in seeds}
def ref_o(seeds): return {Triple("x", "p", o) for o in seeds}


def baseline_nl(seeds_x=None, seeds_y=None):
    base = (
        "List the birth place of scientists. "
        "Return one triple per scientist. "
        "Return only a JSON object: "
        "{\"triples\": [{\"s\": \"scientist\", \"p\": \"dbo:birthPlace\", \"o\": \"place\"}]}"
    )
    if seeds_x:
        base += f" Scientists: {', '.join(sorted(seeds_x))}."
    if seeds_y:
        base += f" Places: {', '.join(sorted(seeds_y))}."
    return base


def baseline_sparql(seeds_x=None, seeds_y=None):
    base = (
        "Using your knowledge, answer: "
        "SELECT ?scientist ?place WHERE { ?scientist dbo:birthPlace ?place }. "
        "Return one triple per scientist. "
        "Return only a JSON object: "
        "{\"triples\": [{\"s\": \"scientist\", \"p\": \"dbo:birthPlace\", \"o\": \"place\"}]}"
    )
    if seeds_x:
        base += f" Scientists: {', '.join(sorted(seeds_x))}."
    if seeds_y:
        base += f" Places: {', '.join(sorted(seeds_y))}."
    return base


def run_exp(label, seeds_x, seeds_y, llm, expected_subset=None):
    """Run all strategies for a given seed configuration."""
    gt = expected_subset if expected_subset is not None else EXPECTED

    print(f"\n{'─'*60}")
    print(f"  {label}")
    if seeds_x: print(f"  Seeds ?x : {len(seeds_x)}")
    if seeds_y: print(f"  Seeds ?y : {len(seeds_y)}")
    print(f"  GT size  : {len(gt)}")
    print(f"{'─'*60}")

    results = {}

    # Baselines (no seed strategies)
    if not seeds_x and not seeds_y:
        r = llm.chat(build_messages(baseline_nl()))
        results["Baseline NL"] = json_to_triples(r.text)
        print(f"  Baseline NL → {len(results['Baseline NL'])} triples")

        r2 = llm.chat(build_messages(baseline_sparql()))
        results["Baseline SPARQL"] = json_to_triples(r2.text)
        print(f"  Baseline SPARQL → {len(results['Baseline SPARQL'])} triples")

        env_ts = Environment()
        results["TableScan"] = LLMTableScan(PATTERN, env_ts, llm, max_iter=MAX_ITER)
        print(f"  TableScan → {len(results['TableScan'])} triples")

    else:
        # Baselines + Filter
        r = llm.chat(build_messages(baseline_nl(seeds_x, seeds_y)))
        raw = json_to_triples(r.text)
        if seeds_x: raw = filter_by_subject(raw, ref_s(seeds_x))
        if seeds_y: raw = filter_by_object(raw, ref_o(seeds_y))
        results["Baseline NL+Filter"] = raw
        print(f"  Baseline NL+Filter → {len(raw)} triples")

        r2 = llm.chat(build_messages(baseline_sparql(seeds_x, seeds_y)))
        raw2 = json_to_triples(r2.text)
        if seeds_x: raw2 = filter_by_subject(raw2, ref_s(seeds_x))
        if seeds_y: raw2 = filter_by_object(raw2, ref_o(seeds_y))
        results["Baseline SPARQL+Filter"] = raw2
        print(f"  Baseline SPARQL+Filter → {len(raw2)} triples")

        env_ts = Environment()
        raw_ts = LLMTableScan(PATTERN, env_ts, llm, max_iter=MAX_ITER)
        if seeds_x: raw_ts = filter_by_subject(raw_ts, ref_s(seeds_x))
        if seeds_y: raw_ts = filter_by_object(raw_ts, ref_o(seeds_y))
        results["TableScan+Filter"] = raw_ts
        print(f"  TableScan+Filter → {len(raw_ts)} triples")

        # µ-Galois strategies
        env_sc = Environment()
        if seeds_x: env_sc.set("?scientist", seeds_x)
        if seeds_y: env_sc.set("?place", seeds_y)
        results["SeedCrank"] = LLMSeedCrank(PATTERN, env_sc, llm, max_iter=MAX_ITER)
        print(f"  SeedCrank → {len(results['SeedCrank'])} triples")

        env_kc = Environment()
        if seeds_x: env_kc.set("?scientist", seeds_x)
        if seeds_y: env_kc.set("?place", seeds_y)
        results["KeyCrank"] = LLMKeyCrank(PATTERN, env_kc, llm)
        print(f"  KeyCrank → {len(results['KeyCrank'])} triples")

        env_tc = Environment()
        if seeds_x: env_tc.set("?scientist", seeds_x)
        if seeds_y: env_tc.set("?place", seeds_y)
        results["TripletScan"] = LLMTripletScan(PATTERN, env_tc, llm,
                                                 max_iter=MAX_ITER, tau_strategie=TAU)
        print(f"  TripletScan → {len(results['TripletScan'])} triples")

    report = Report(predicate=f"birthPlace — {label}")
    for strat, triples in results.items():
        report.add(strat, Evaluator(triples, gt).evaluate())
    report.print_table()
    return report


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  Experiment — T3 : (?x, birthPlace, ?y)")
    print(f"  Model   : {MODEL}  |  τ={TAU}  |  max_iter={MAX_ITER}")
    print(f"  GT size : {len(EXPECTED)} triples")
    print(f"{'='*60}")

    llm = AzureOpenAIClient()

    reports = {
        "A": run_exp("Exp A — no seed",           None,        None,        llm),
        "B": run_exp("Exp B — seed ?x small",      SX_SMALL,    None,        llm),
        "C": run_exp("Exp C — seed ?x large",      SX_LARGE,    None,        llm),
        "D": run_exp("Exp D — seed ?y small",      None,        SY_SMALL,    llm),
        "E": run_exp("Exp E — seed ?y large",      None,        SY_LARGE,    llm),
        "F": run_exp("Exp F — seed ?x=1, ?y=large", SX_F, SY_F, llm,
                     expected_subset={t for t in EXPECTED if t.s in SX_F}),
        "G": run_exp("Exp G — seed ?x=large, ?y=1", SX_G, SY_G, llm,
                     expected_subset={t for t in EXPECTED if t.o in SY_G}),
        "H": run_exp("Exp H — seed ?x+?y large",  SX_LARGE, SY_LARGE, llm,
                     expected_subset={t for t in EXPECTED
                                      if t.s in SX_LARGE and t.o in SY_LARGE}),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for key, rep in reports.items():
        rep.save_json(os.path.join(RESULTS_DIR, f"t3_exp_{key.lower()}.json"))
        rep.save_csv(os.path.join(RESULTS_DIR,  f"t3_exp_{key.lower()}.csv"))


if __name__ == "__main__":
    main()