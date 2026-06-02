"""
experiments/exp_t2_birthplace/run.py
=====================================

Research question
-----------------
Pattern T2 : (?x, dbo:birthPlace, Ulm) — object fixed
Who was born in Ulm ?

Exp A — no seed on ?x
    Baseline NL, Baseline SPARQL, TableScan

Exp B — small seed on ?x (5 candidates)
    Baseline NL+Filter, Baseline SPARQL+Filter, TableScan+Filter,
    SeedCrank, KeyCrank, TripletScan

Exp C — large seed on ?x (50 candidates)
    Baseline NL+Filter, Baseline SPARQL+Filter, TableScan+Filter,
    SeedCrank, KeyCrank, TripletScan

Ground truth
------------
T2["dbo:birthPlace|dbo:Paris"] from scientists subgraph

Run
---
    python3 -m experiments.exp_t2_birthplace.run
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
from mugalois.core.operators import filter_by_subject
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
OBJ_URI  = "http://dbpedia.org/resource/Ulm"
MODEL    = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini-2")
TAU      = 0.7
MAX_ITER = 3

GT_PATH     = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "ground_truth", "ground_truth.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ─── Ground truth ─────────────────────────────────────────────────────────────

ds              = ScientistDataset(GT_PATH)
EXPECTED        = ds.T2(PRED_URI, OBJ_URI)
CORRECT_SUBJECTS = {t.s for t in EXPECTED}
ALL_SUBJECTS     = ds.subjects(PRED_URI) - CORRECT_SUBJECTS

print(f"\n  GT : {len(EXPECTED)} scientists born in Ulm → {CORRECT_SUBJECTS}")

random.seed(42)
SEEDS_SMALL = CORRECT_SUBJECTS | set(random.sample(sorted(ALL_SUBJECTS), 2))
SEEDS_LARGE = CORRECT_SUBJECTS | set(random.sample(sorted(ALL_SUBJECTS), 47))

# ─── Prompts ──────────────────────────────────────────────────────────────────

def baseline_nl_prompt(seeds: set[str] | None = None) -> str:
    base = (
        "Who are the scientists born in Ulm? "
        "Return one triple per scientist. "
        "Return only a JSON object: "
        "{\"triples\": [{\"s\": \"scientist name\", \"p\": \"dbo:birthPlace\", \"o\": \"Ulm\"}]}"
    )
    if seeds:
        base += f" Only consider these candidate scientists: {', '.join(sorted(seeds))}."
    return base


def baseline_sparql_prompt(seeds: set[str] | None = None) -> str:
    base = (
        "Using your knowledge, answer this SPARQL pattern: "
        "(?scientist, dbo:birthPlace, Ulm). "
        "List all scientists born in Ulm. "
        "Return one triple per scientist. "
        "Return only a JSON object: "
        "{\"triples\": [{\"s\": \"scientist name\", \"p\": \"dbo:birthPlace\", \"o\": \"Ulm\"}]}"
    )
    if seeds:
        base += f" Only consider these candidate scientists: {', '.join(sorted(seeds))}."
    return base


def make_seed_triples(seeds: set[str]) -> set[Triple]:
    return {Triple(s, "dbo:birthPlace", "x") for s in seeds}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  Experiment — T2 : (?x, birthPlace, Ulm)")
    print(f"  Model   : {MODEL}  |  τ={TAU}  |  max_iter={MAX_ITER}")
    print(f"  GT      : {CORRECT_SUBJECTS}")
    print(f"{'='*60}")

    llm = AzureOpenAIClient()
    pattern = TriplePattern("?scientist", "dbo:birthPlace", "Ulm")

    # ══════════════════════════════════════════════════════════════════════════
    # EXP A — no seed on ?x
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("  EXP A — no seed on ?x")
    print("─"*60)

    r = llm.chat(build_messages(baseline_nl_prompt()))
    a_nl = json_to_triples(r.text)
    print(f"  [1/3] Baseline NL → {len(a_nl)} triples : {[t.s for t in a_nl]}")

    r2 = llm.chat(build_messages(baseline_sparql_prompt()))
    a_sparql = json_to_triples(r2.text)
    print(f"  [2/3] Baseline SPARQL → {len(a_sparql)} triples : {[t.s for t in a_sparql]}")

    env_ts = Environment()
    a_table = LLMTableScan(pattern, env_ts, llm, max_iter=MAX_ITER)
    print(f"  [3/3] TableScan → {len(a_table)} triples : {[t.s for t in a_table]}")

    report_a = Report(predicate="birthPlace — T2 Exp A (no seed, ?x born in Ulm)")
    for label, triples in [
        ("Baseline NL",     a_nl),
        ("Baseline SPARQL", a_sparql),
        ("TableScan",       a_table),
    ]:
        report_a.add(label, Evaluator(triples, EXPECTED).evaluate())
    report_a.print_table()

    # ══════════════════════════════════════════════════════════════════════════
    # EXP B — small seed on ?x
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print(f"  EXP B — small seed on ?x ({len(SEEDS_SMALL)} candidates)")
    print(f"  Seeds : {sorted(SEEDS_SMALL)}")
    print("─"*60)

    seed_ref_small = make_seed_triples(SEEDS_SMALL)

    r = llm.chat(build_messages(baseline_nl_prompt(SEEDS_SMALL)))
    b_nl = filter_by_subject(json_to_triples(r.text), seed_ref_small)
    print(f"  [1/6] Baseline NL+Filter → {len(b_nl)} triples")

    r2 = llm.chat(build_messages(baseline_sparql_prompt(SEEDS_SMALL)))
    b_sparql = filter_by_subject(json_to_triples(r2.text), seed_ref_small)
    print(f"  [2/6] Baseline SPARQL+Filter → {len(b_sparql)} triples")

    env_ts2 = Environment()
    b_table = filter_by_subject(LLMTableScan(pattern, env_ts2, llm, max_iter=MAX_ITER), seed_ref_small)
    print(f"  [3/6] TableScan+Filter → {len(b_table)} triples")

    env_sc = Environment()
    env_sc.set("?scientist", SEEDS_SMALL)
    b_seed = LLMSeedCrank(pattern, env_sc, llm, max_iter=MAX_ITER)
    print(f"  [4/6] SeedCrank → {len(b_seed)} triples")

    env_kc = Environment()
    env_kc.set("?scientist", SEEDS_SMALL)
    b_key = LLMKeyCrank(pattern, env_kc, llm)
    print(f"  [5/6] KeyCrank → {len(b_key)} triples")

    env_tc = Environment()
    env_tc.set("?scientist", SEEDS_SMALL)
    b_triplet = LLMTripletScan(pattern, env_tc, llm, max_iter=MAX_ITER, tau_strategie=TAU)
    print(f"  [6/6] TripletScan → {len(b_triplet)} triples")

    report_b = Report(predicate="birthPlace — T2 Exp B (small seed on ?x)")
    for label, triples in [
        ("Baseline NL+Filter",     b_nl),
        ("Baseline SPARQL+Filter", b_sparql),
        ("TableScan+Filter",       b_table),
        ("SeedCrank",              b_seed),
        ("KeyCrank",               b_key),
        ("TripletScan",            b_triplet),
    ]:
        report_b.add(label, Evaluator(triples, EXPECTED).evaluate())
    report_b.print_table()

    # ══════════════════════════════════════════════════════════════════════════
    # EXP C — large seed on ?x
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print(f"  EXP C — large seed on ?x ({len(SEEDS_LARGE)} candidates)")
    print("─"*60)

    seed_ref_large = make_seed_triples(SEEDS_LARGE)

    r = llm.chat(build_messages(baseline_nl_prompt(SEEDS_LARGE)))
    c_nl = filter_by_subject(json_to_triples(r.text), seed_ref_large)
    print(f"  [1/6] Baseline NL+Filter → {len(c_nl)} triples")

    r2 = llm.chat(build_messages(baseline_sparql_prompt(SEEDS_LARGE)))
    c_sparql = filter_by_subject(json_to_triples(r2.text), seed_ref_large)
    print(f"  [2/6] Baseline SPARQL+Filter → {len(c_sparql)} triples")

    env_ts3 = Environment()
    c_table = filter_by_subject(LLMTableScan(pattern, env_ts3, llm, max_iter=MAX_ITER), seed_ref_large)
    print(f"  [3/6] TableScan+Filter → {len(c_table)} triples")

    env_sc2 = Environment()
    env_sc2.set("?scientist", SEEDS_LARGE)
    c_seed = LLMSeedCrank(pattern, env_sc2, llm, max_iter=MAX_ITER)
    print(f"  [4/6] SeedCrank → {len(c_seed)} triples")

    env_kc2 = Environment()
    env_kc2.set("?scientist", SEEDS_LARGE)
    c_key = LLMKeyCrank(pattern, env_kc2, llm)
    print(f"  [5/6] KeyCrank → {len(c_key)} triples")

    env_tc2 = Environment()
    env_tc2.set("?scientist", SEEDS_LARGE)
    c_triplet = LLMTripletScan(pattern, env_tc2, llm, max_iter=MAX_ITER, tau_strategie=TAU)
    print(f"  [6/6] TripletScan → {len(c_triplet)} triples")

    report_c = Report(predicate="birthPlace — T2 Exp C (large seed on ?x)")
    for label, triples in [
        ("Baseline NL+Filter",     c_nl),
        ("Baseline SPARQL+Filter", c_sparql),
        ("TableScan+Filter",       c_table),
        ("SeedCrank",              c_seed),
        ("KeyCrank",               c_key),
        ("TripletScan",            c_triplet),
    ]:
        report_c.add(label, Evaluator(triples, EXPECTED).evaluate())
    report_c.print_table()

    # ── Export ────────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for rep, name in [
        (report_a, "t2_exp_a_no_seed"),
        (report_b, "t2_exp_b_small_seed"),
        (report_c, "t2_exp_c_large_seed"),
    ]:
        rep.save_json(os.path.join(RESULTS_DIR, f"{name}.json"))
        rep.save_csv(os.path.join(RESULTS_DIR,  f"{name}.csv"))


if __name__ == "__main__":
    main()