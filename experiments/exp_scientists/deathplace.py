"""
experiments/exp_t3_deathplace/run.py
=====================================

Pattern T3 : (?x, dbo:deathPlace, ?y) — 376 triples

Experiments A-H same design as spouse experiment.

Run
---
    python3 -m experiments.exp_t3_deathplace.run
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

PRED_URI = "http://dbpedia.org/ontology/deathPlace"
MODEL    = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini-2")
TAU      = 0.7
MAX_ITER = 3

GT_PATH     = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "ground_truth", "ground_truth.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

ds           = ScientistDataset(GT_PATH)
EXPECTED     = ds.T3(PRED_URI)
ALL_SUBJECTS = ds.subjects(PRED_URI)
ALL_OBJECTS  = ds.objects(PRED_URI)

random.seed(42)

SX_SMALL = set(random.sample(sorted(ALL_SUBJECTS), 5))
SX_LARGE = set(random.sample(sorted(ALL_SUBJECTS), 50))
SY_SMALL = set(random.sample(sorted(ALL_OBJECTS),  5))
SY_LARGE = set(random.sample(sorted(ALL_OBJECTS),  50))
SX_F     = {"Albert Einstein"}
SY_F     = SY_LARGE
SX_G     = SX_LARGE
SY_G     = {"Princeton, New Jersey"}

PATTERN = TriplePattern("?scientist", "dbo:deathPlace", "?place")


def ref_s(seeds): return {Triple(s, "p", "x") for s in seeds}
def ref_o(seeds): return {Triple("x", "p", o) for o in seeds}
def gt_for(sx=None, sy=None):
    r = EXPECTED
    if sx: r = {t for t in r if t.s in sx}
    if sy: r = {t for t in r if t.o in sy}
    return r


def baseline_nl(sx=None, sy=None):
    base = (
        "List the death place of scientists you know. "
        "Return one triple per scientist. "
        "Return only a JSON object: "
        "{\"triples\": [{\"s\": \"scientist\", \"p\": \"dbo:deathPlace\", \"o\": \"place\"}]}"
    )
    if sx: base += f" Scientists: {', '.join(sorted(sx))}."
    if sy: base += f" Places: {', '.join(sorted(sy))}."
    return base


def baseline_sparql(sx=None, sy=None):
    base = (
        "Using your knowledge, answer: "
        "SELECT ?scientist ?place WHERE { ?scientist dbo:deathPlace ?place }. "
        "Return one triple per scientist. "
        "Return only a JSON object: "
        "{\"triples\": [{\"s\": \"scientist\", \"p\": \"dbo:deathPlace\", \"o\": \"place\"}]}"
    )
    if sx: base += f" Scientists: {', '.join(sorted(sx))}."
    if sy: base += f" Places: {', '.join(sorted(sy))}."
    return base


def run_exp(label, sx, sy, llm):
    gt = gt_for(sx, sy)
    print(f"\n{'─'*60}")
    print(f"  {label}")
    if sx: print(f"  Seeds ?x : {len(sx)}")
    if sy: print(f"  Seeds ?y : {len(sy)}")
    print(f"  GT size  : {len(gt)}")
    print(f"{'─'*60}")

    results = {}

    if not sx and not sy:
        r = llm.chat(build_messages(baseline_nl()))
        results["Baseline NL"] = json_to_triples(r.text)
        print(f"  Baseline NL → {len(results['Baseline NL'])} triples")

        r2 = llm.chat(build_messages(baseline_sparql()))
        results["Baseline SPARQL"] = json_to_triples(r2.text)
        print(f"  Baseline SPARQL → {len(results['Baseline SPARQL'])} triples")

        env = Environment()
        results["TableScan"] = LLMTableScan(PATTERN, env, llm, max_iter=MAX_ITER)
        print(f"  TableScan → {len(results['TableScan'])} triples")

    else:
        r = llm.chat(build_messages(baseline_nl(sx, sy)))
        raw = json_to_triples(r.text)
        if sx: raw = filter_by_subject(raw, ref_s(sx))
        if sy: raw = filter_by_object(raw, ref_o(sy))
        results["Baseline NL+Filter"] = raw
        print(f"  Baseline NL+Filter → {len(raw)} triples")

        r2 = llm.chat(build_messages(baseline_sparql(sx, sy)))
        raw2 = json_to_triples(r2.text)
        if sx: raw2 = filter_by_subject(raw2, ref_s(sx))
        if sy: raw2 = filter_by_object(raw2, ref_o(sy))
        results["Baseline SPARQL+Filter"] = raw2
        print(f"  Baseline SPARQL+Filter → {len(raw2)} triples")

        env = Environment()
        raw_ts = LLMTableScan(PATTERN, env, llm, max_iter=MAX_ITER)
        if sx: raw_ts = filter_by_subject(raw_ts, ref_s(sx))
        if sy: raw_ts = filter_by_object(raw_ts, ref_o(sy))
        results["TableScan+Filter"] = raw_ts
        print(f"  TableScan+Filter → {len(raw_ts)} triples")

        env_sc = Environment()
        if sx: env_sc.set("?scientist", sx)
        if sy: env_sc.set("?place", sy)
        results["SeedCrank"] = LLMSeedCrank(PATTERN, env_sc, llm, max_iter=MAX_ITER)
        print(f"  SeedCrank → {len(results['SeedCrank'])} triples")

        env_kc = Environment()
        if sx: env_kc.set("?scientist", sx)
        if sy: env_kc.set("?place", sy)
        results["KeyCrank"] = LLMKeyCrank(PATTERN, env_kc, llm)
        print(f"  KeyCrank → {len(results['KeyCrank'])} triples")

        env_tc = Environment()
        if sx: env_tc.set("?scientist", sx)
        if sy: env_tc.set("?place", sy)
        results["TripletScan"] = LLMTripletScan(
            PATTERN, env_tc, llm, max_iter=MAX_ITER, tau_strategie=TAU)
        print(f"  TripletScan → {len(results['TripletScan'])} triples")

    report = Report(predicate=f"deathPlace — {label}")
    for strat, triples in results.items():
        report.add(strat, Evaluator(triples, gt).evaluate())
    report.print_table()
    return report


def main():
    print(f"\n{'='*60}")
    print(f"  Experiment — T3 : (?x, deathPlace, ?y)")
    print(f"  Model   : {MODEL}  |  τ={TAU}  |  max_iter={MAX_ITER}")
    print(f"  GT size : {len(EXPECTED)} triples")
    print(f"  Subjects: {len(ALL_SUBJECTS)}  |  Objects: {len(ALL_OBJECTS)}")
    print(f"{'='*60}")

    llm = AzureOpenAIClient()

    reports = {
        "A": run_exp("Exp A — no seed",            None,     None,  llm),
        "B": run_exp("Exp B — seed ?x small (5)",  SX_SMALL, None,  llm),
        "C": run_exp("Exp C — seed ?x large (50)", SX_LARGE, None,  llm),
        "D": run_exp("Exp D — seed ?y small (5)",  None,  SY_SMALL, llm),
        "E": run_exp("Exp E — seed ?y large (50)", None,  SY_LARGE, llm),
        "F": run_exp("Exp F — ?x=1, ?y=large",     SX_F,     SY_F,  llm),
        "G": run_exp("Exp G — ?x=large, ?y=1",     SX_G,     SY_G,  llm),
        "H": run_exp("Exp H — ?x+?y large",        SX_LARGE, SY_LARGE, llm),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for key, rep in reports.items():
        rep.save_json(os.path.join(RESULTS_DIR, f"t3_deathplace_exp_{key.lower()}.json"))
        rep.save_csv(os.path.join(RESULTS_DIR,  f"t3_deathplace_exp_{key.lower()}.csv"))


if __name__ == "__main__":
    main()