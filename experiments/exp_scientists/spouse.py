"""
experiments/exp_t3_spouse/run.py
=================================

Research question
-----------------
Pattern T3 : (?x, dbo:spouse, ?y) — both variables free
43 triples, 40 subjects, 43 objects — lightweight predicate for clean evaluation.

Experiments
-----------
A — no seed
B — seed ?x small  (5 scientists)
C — seed ?x large  (20 scientists)
D — seed ?y small  (5 spouses)
E — seed ?y large  (20 spouses)
F — seed ?x=1, seed ?y=large  (one scientist, many spouse candidates)
G — seed ?x=large, seed ?y=1  (many scientists, one spouse candidate)
H — seed ?x+?y large

Ground truth : T3["dbo:spouse"] = 43 triples

Run
---
    python3 -m experiments.exp_t3_spouse.run
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

PRED_URI = "http://dbpedia.org/ontology/spouse"
MODEL    = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini-2")
TAU      = 0.7
MAX_ITER = 3

GT_PATH     = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "ground_truth", "ground_truth.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ─── Ground truth ─────────────────────────────────────────────────────────────

ds           = ScientistDataset(GT_PATH)
EXPECTED     = ds.T3(PRED_URI)
ALL_SUBJECTS = ds.subjects(PRED_URI)   # 40 scientists
ALL_OBJECTS  = ds.objects(PRED_URI)    # 43 spouses

random.seed(42)

SX_SMALL = set(random.sample(sorted(ALL_SUBJECTS), 5))
SX_LARGE = set(random.sample(sorted(ALL_SUBJECTS), 20))
SY_SMALL = set(random.sample(sorted(ALL_OBJECTS),  5))
SY_LARGE = set(random.sample(sorted(ALL_OBJECTS),  20))

# F : ?x=1, ?y=large
SX_F = {"Albert Einstein"}
SY_F = SY_LARGE

# G : ?x=large, ?y=1
SX_G = SX_LARGE
SY_G = {"Elsa Einstein"}

# ─── Helpers ──────────────────────────────────────────────────────────────────

PATTERN = TriplePattern("?scientist", "dbo:spouse", "?spouse")


def ref_s(seeds): return {Triple(s, "p", "x") for s in seeds}
def ref_o(seeds): return {Triple("x", "p", o) for o in seeds}


def gt_for(seeds_x=None, seeds_y=None):
    """Ground truth restricted to the seeds."""
    result = EXPECTED
    if seeds_x:
        result = {t for t in result if t.s in seeds_x}
    if seeds_y:
        result = {t for t in result if t.o in seeds_y}
    return result


def baseline_nl(seeds_x=None, seeds_y=None):
    base = (
        "List the spouse of scientists you know. "
        "Return one triple per scientist. "
        "Return only a JSON object: "
        "{\"triples\": [{\"s\": \"scientist\", \"p\": \"dbo:spouse\", \"o\": \"spouse name\"}]}"
    )
    if seeds_x:
        base += f" Scientists: {', '.join(sorted(seeds_x))}."
    if seeds_y:
        base += f" Spouses: {', '.join(sorted(seeds_y))}."
    return base


def baseline_sparql(seeds_x=None, seeds_y=None):
    base = (
        "Using your knowledge, answer: "
        "SELECT ?scientist ?spouse WHERE { ?scientist dbo:spouse ?spouse }. "
        "Return one triple per scientist. "
        "Return only a JSON object: "
        "{\"triples\": [{\"s\": \"scientist\", \"p\": \"dbo:spouse\", \"o\": \"spouse name\"}]}"
    )
    if seeds_x:
        base += f" Scientists: {', '.join(sorted(seeds_x))}."
    if seeds_y:
        base += f" Spouses: {', '.join(sorted(seeds_y))}."
    return base


def run_exp(label, seeds_x, seeds_y, llm):
    gt = gt_for(seeds_x, seeds_y)

    print(f"\n{'─'*60}")
    print(f"  {label}")
    if seeds_x: print(f"  Seeds ?x : {len(seeds_x)}")
    if seeds_y: print(f"  Seeds ?y : {len(seeds_y)}")
    print(f"  GT size  : {len(gt)}")
    print(f"{'─'*60}")

    results = {}

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

        env_sc = Environment()
        if seeds_x: env_sc.set("?scientist", seeds_x)
        if seeds_y: env_sc.set("?spouse", seeds_y)
        results["SeedCrank"] = LLMSeedCrank(PATTERN, env_sc, llm, max_iter=MAX_ITER)
        print(f"  SeedCrank → {len(results['SeedCrank'])} triples")

        env_kc = Environment()
        if seeds_x: env_kc.set("?scientist", seeds_x)
        if seeds_y: env_kc.set("?spouse", seeds_y)
        results["KeyCrank"] = LLMKeyCrank(PATTERN, env_kc, llm)
        print(f"  KeyCrank → {len(results['KeyCrank'])} triples")

        env_tc = Environment()
        if seeds_x: env_tc.set("?scientist", seeds_x)
        if seeds_y: env_tc.set("?spouse", seeds_y)
        results["TripletScan"] = LLMTripletScan(
            PATTERN, env_tc, llm, max_iter=MAX_ITER, tau_strategie=TAU)
        print(f"  TripletScan → {len(results['TripletScan'])} triples")

    report = Report(predicate=f"spouse — {label}")
    for strat, triples in results.items():
        report.add(strat, Evaluator(triples, gt).evaluate())
    report.print_table()
    return report


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  Experiment — T3 : (?x, spouse, ?y)")
    print(f"  Model   : {MODEL}  |  τ={TAU}  |  max_iter={MAX_ITER}")
    print(f"  GT size : {len(EXPECTED)} triples")
    print(f"  Subjects: {len(ALL_SUBJECTS)}  |  Objects: {len(ALL_OBJECTS)}")
    print(f"{'='*60}")

    llm = AzureOpenAIClient()

    reports = {
        "A": run_exp("Exp A — no seed",              None,     None,     llm),
        "B": run_exp("Exp B — seed ?x small (5)",    SX_SMALL, None,     llm),
        "C": run_exp("Exp C — seed ?x large (20)",   SX_LARGE, None,     llm),
        "D": run_exp("Exp D — seed ?y small (5)",    None,     SY_SMALL, llm),
        "E": run_exp("Exp E — seed ?y large (20)",   None,     SY_LARGE, llm),
        "F": run_exp("Exp F — ?x=1, ?y=large",       SX_F,     SY_F,     llm),
        "G": run_exp("Exp G — ?x=large, ?y=1",       SX_G,     SY_G,     llm),
        "H": run_exp("Exp H — ?x+?y large",          SX_LARGE, SY_LARGE, llm),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for key, rep in reports.items():
        rep.save_json(os.path.join(RESULTS_DIR, f"t3_spouse_exp_{key.lower()}.json"))
        rep.save_csv(os.path.join(RESULTS_DIR,  f"t3_spouse_exp_{key.lower()}.csv"))


if __name__ == "__main__":
    main()