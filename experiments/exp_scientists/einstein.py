"""
experiments/exp_einstein_birthplace/run.py
==========================================

Research question
-----------------
How do different strategies perform on T1 pattern:
(Albert Einstein, dbo:birthPlace, ?y) ?

Three experiments
-----------------
Exp A — no seed on ?y
    Baseline NL, Baseline SPARQL, TableScan

Exp B — small seed on ?y (5 candidates)
    Baseline NL + Filter, Baseline SPARQL + Filter, TableScan + Filter,
    SeedCrank, KeyCrank, TripletScan

Exp C — large seed on ?y (50 candidates)
    Baseline NL + Filter, Baseline SPARQL + Filter, TableScan + Filter,
    SeedCrank, KeyCrank, TripletScan

Ground truth
------------
T1["Albert_Einstein|birthPlace"] = {Ulm, Kingdom of Württemberg, German Empire}

Run
---
    python3 -m experiments.exp_einstein_birthplace.run
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
from mugalois.core.operators import filter_by_object
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
SUBJ_URI = "http://dbpedia.org/resource/Albert_Einstein"
MODEL    = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini-2")
TAU      = 0.7
MAX_ITER = 3

GT_PATH     = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "ground_truth", "ground_truth.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ─── Ground truth ─────────────────────────────────────────────────────────────

ds              = ScientistDataset(GT_PATH)
EXPECTED        = ds.T1(SUBJ_URI, PRED_URI)
CORRECT_OBJECTS = {t.o for t in EXPECTED}
ALL_OBJECTS     = ds.objects(PRED_URI) - CORRECT_OBJECTS

random.seed(42)
SEEDS_SMALL = CORRECT_OBJECTS | set(random.sample(sorted(ALL_OBJECTS), 2))
SEEDS_LARGE = CORRECT_OBJECTS | set(random.sample(sorted(ALL_OBJECTS), 47))

# ─── Prompts ──────────────────────────────────────────────────────────────────

def baseline_nl_prompt(seeds: set[str] | None = None) -> str:
    base = (
        "What are the birth places of Albert Einstein? "
        "Return one triple per birth place. "
        "Return only a JSON object: "
        "{\"triples\": [{\"s\": \"Albert Einstein\", \"p\": \"dbo:birthPlace\", \"o\": \"place name\"}]}"
    )
    if seeds:
        base += f" Only consider these candidate places: {', '.join(sorted(seeds))}."
    return base


def baseline_sparql_prompt(seeds: set[str] | None = None) -> str:
    base = (
        "Using your knowledge, answer this SPARQL pattern: "
        "(Albert Einstein, dbo:birthPlace, ?place). "
        "Return one triple per place. "
        "Return only a JSON object: "
        "{\"triples\": [{\"s\": \"Albert Einstein\", \"p\": \"dbo:birthPlace\", \"o\": \"place name\"}]}"
    )
    if seeds:
        base += f" Only consider these candidate places: {', '.join(sorted(seeds))}."
    return base


# ─── Helper ───────────────────────────────────────────────────────────────────

def run_strategy(label, fn):
    print(f"  {label} ...")
    result = fn()
    print(f"  → {len(result)} triples : {[t.o for t in result]}")
    return result


def make_seed_triples(seeds: set[str]) -> set[Triple]:
    """Build a reference set for filter_by_object."""
    return {Triple("x", "dbo:birthPlace", o) for o in seeds}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  Experiment — T1 : (Albert Einstein, birthPlace, ?y)")
    print(f"  Model   : {MODEL}  |  τ={TAU}  |  max_iter={MAX_ITER}")
    print(f"  GT      : {CORRECT_OBJECTS}")
    print(f"{'='*60}")

    llm = AzureOpenAIClient()
    pattern = TriplePattern("?scientist", "dbo:birthPlace", "?place")

    # ══════════════════════════════════════════════════════════════════════════
    # EXP A — no seed on ?y
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("  EXP A — no seed on ?y")
    print("─"*60)

    r = llm.chat(build_messages(baseline_nl_prompt()))
    a_nl = run_strategy("[1/3] Baseline NL", lambda: json_to_triples(r.text))

    r2 = llm.chat(build_messages(baseline_sparql_prompt()))
    a_sparql = run_strategy("[2/3] Baseline SPARQL", lambda: json_to_triples(r2.text))

    env_ts = Environment()
    a_table = run_strategy("[3/3] TableScan",
        lambda: LLMTableScan(pattern, env_ts, llm, max_iter=MAX_ITER))

    report_a = Report(predicate="birthPlace — Exp A (no seed)")
    for label, triples in [
        ("Baseline NL",    a_nl),
        ("Baseline SPARQL", a_sparql),
        ("TableScan",       a_table),
    ]:
        report_a.add(label, Evaluator(triples, EXPECTED).evaluate())
    report_a.print_table()

    # ══════════════════════════════════════════════════════════════════════════
    # EXP B — small seed on ?y
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print(f"  EXP B — small seed on ?y ({len(SEEDS_SMALL)} candidates)")
    print(f"  Seeds : {sorted(SEEDS_SMALL)}")
    print("─"*60)

    seed_ref_small = make_seed_triples(SEEDS_SMALL)

    r = llm.chat(build_messages(baseline_nl_prompt(SEEDS_SMALL)))
    b_nl = run_strategy("[1/6] Baseline NL + Filter",
        lambda: filter_by_object(json_to_triples(r.text), seed_ref_small))

    r2 = llm.chat(build_messages(baseline_sparql_prompt(SEEDS_SMALL)))
    b_sparql = run_strategy("[2/6] Baseline SPARQL + Filter",
        lambda: filter_by_object(json_to_triples(r2.text), seed_ref_small))

    env_ts2 = Environment()
    raw_table = LLMTableScan(pattern, env_ts2, llm, max_iter=MAX_ITER)
    b_table = run_strategy("[3/6] TableScan + Filter",
        lambda: filter_by_object(raw_table, seed_ref_small))

    env_sc = Environment()
    env_sc.set("?scientist", {"Albert Einstein"})
    env_sc.set("?place", SEEDS_SMALL)
    b_seed = run_strategy("[4/6] SeedCrank",
        lambda: LLMSeedCrank(pattern, env_sc, llm, max_iter=MAX_ITER))

    env_kc = Environment()
    env_kc.set("?scientist", {"Albert Einstein"})
    env_kc.set("?place", SEEDS_SMALL)
    b_key = run_strategy("[5/6] KeyCrank",
        lambda: LLMKeyCrank(pattern, env_kc, llm))

    env_tc = Environment()
    env_tc.set("?scientist", {"Albert Einstein"})
    env_tc.set("?place", SEEDS_SMALL)
    b_triplet = run_strategy("[6/6] TripletScan",
        lambda: LLMTripletScan(pattern, env_tc, llm, max_iter=MAX_ITER, tau_strategie=TAU))

    report_b = Report(predicate="birthPlace — Exp B (small seed)")
    for label, triples in [
        ("Baseline NL+Filter",    b_nl),
        ("Baseline SPARQL+Filter", b_sparql),
        ("TableScan+Filter",       b_table),
        ("SeedCrank",              b_seed),
        ("KeyCrank",               b_key),
        ("TripletScan",            b_triplet),
    ]:
        report_b.add(label, Evaluator(triples, EXPECTED).evaluate())
    report_b.print_table()

    # ══════════════════════════════════════════════════════════════════════════
    # EXP C — large seed on ?y
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print(f"  EXP C — large seed on ?y ({len(SEEDS_LARGE)} candidates)")
    print("─"*60)

    seed_ref_large = make_seed_triples(SEEDS_LARGE)

    r = llm.chat(build_messages(baseline_nl_prompt(SEEDS_LARGE)))
    c_nl = run_strategy("[1/6] Baseline NL + Filter",
        lambda: filter_by_object(json_to_triples(r.text), seed_ref_large))

    r2 = llm.chat(build_messages(baseline_sparql_prompt(SEEDS_LARGE)))
    c_sparql = run_strategy("[2/6] Baseline SPARQL + Filter",
        lambda: filter_by_object(json_to_triples(r2.text), seed_ref_large))

    env_ts3 = Environment()
    raw_table2 = LLMTableScan(pattern, env_ts3, llm, max_iter=MAX_ITER)
    c_table = run_strategy("[3/6] TableScan + Filter",
        lambda: filter_by_object(raw_table2, seed_ref_large))

    env_sc2 = Environment()
    env_sc2.set("?scientist", {"Albert Einstein"})
    env_sc2.set("?place", SEEDS_LARGE)
    c_seed = run_strategy("[4/6] SeedCrank",
        lambda: LLMSeedCrank(pattern, env_sc2, llm, max_iter=MAX_ITER))

    env_kc2 = Environment()
    env_kc2.set("?scientist", {"Albert Einstein"})
    env_kc2.set("?place", SEEDS_LARGE)
    c_key = run_strategy("[5/6] KeyCrank",
        lambda: LLMKeyCrank(pattern, env_kc2, llm))

    env_tc2 = Environment()
    env_tc2.set("?scientist", {"Albert Einstein"})
    env_tc2.set("?place", SEEDS_LARGE)
    c_triplet = run_strategy("[6/6] TripletScan",
        lambda: LLMTripletScan(pattern, env_tc2, llm, max_iter=MAX_ITER, tau_strategie=TAU))

    report_c = Report(predicate="birthPlace — Exp C (large seed)")
    for label, triples in [
        ("Baseline NL+Filter",    c_nl),
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
        (report_a, "exp_a_no_seed"),
        (report_b, "exp_b_small_seed"),
        (report_c, "exp_c_large_seed"),
    ]:
        rep.save_json(os.path.join(RESULTS_DIR, f"{name}.json"))
        rep.save_csv(os.path.join(RESULTS_DIR,  f"{name}.csv"))


if __name__ == "__main__":
    main()