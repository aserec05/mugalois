"""
experiments/exp_keycrank_prompt/run.py
=======================================

Research question
-----------------
Does changing "is one of" to "may be one of" + exit clause
reduce hallucinations in KeyCrank ?

Two variants compared on same seeds, same GT, same LLM :
- KeyCrank original  : "?place is one of: ..."
- KeyCrank improved  : "?place may be one of: ... If none apply, return empty."

Predicate : dbo:spouse — 43 triples (lightweight, fast evaluation)

Run
---
    python3 -m experiments.exp_keycrank_prompt.run
"""

import os
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mugalois.core.types import Triple, TriplePattern, Environment
from mugalois.core.helpers import seedsOf
from mugalois.core.parser import json_to_triples
from mugalois.core.prompts import build_messages, SYSTEM_PROMPT, _build_constraints
from mugalois.core.helpers import updateEnv
from mugalois.llm.llm_client import AzureOpenAIClient

from evaluation.dataset import ScientistDataset
from evaluation.evaluator import Evaluator
from evaluation.report import Report

# ─── Config ───────────────────────────────────────────────────────────────────

PRED_URI = "http://dbpedia.org/ontology/spouse"
MODEL    = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini-2")

GT_PATH     = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "ground_truth", "ground_truth.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

ds           = ScientistDataset(GT_PATH)
EXPECTED     = ds.T3(PRED_URI)
ALL_SUBJECTS = ds.subjects(PRED_URI)
ALL_OBJECTS  = ds.objects(PRED_URI)

import random
random.seed(42)
SEEDS_X = set(random.sample(sorted(ALL_SUBJECTS), 10))
SEEDS_Y = set(random.sample(sorted(ALL_OBJECTS),  10))

PATTERN = TriplePattern("?scientist", "dbo:spouse", "?spouse")


# ─── Prompt variants ──────────────────────────────────────────────────────────

def gen_keycrank_original(pattern, env, seed_value, direction):
    """Current prompt — hard constraint."""
    constraint = _build_constraints(pattern, env,
                                    side="o" if direction == "L->R" else "s")
    if direction == "L->R":
        return (
            f"Context: The subject is fixed: {seed_value}. "
            f"The predicate is {pattern.p}.\n\n"
            f"Task: List all triples ({seed_value}, {pattern.p}, {pattern.o}) "
            f"that factually hold.\n"
            f"{constraint}\n"
            f"Return only factual values."
        )
    else:
        return (
            f"Context: The object is fixed: {seed_value}. "
            f"The predicate is {pattern.p}.\n\n"
            f"Task: List all triples ({pattern.s}, {pattern.p}, {seed_value}) "
            f"that factually hold.\n"
            f"{constraint}\n"
            f"Return only factual values."
        )


def gen_keycrank_improved(pattern, env, seed_value, direction):
    """Improved prompt — soft constraint + exit clause."""
    constraint = _build_constraints(pattern, env,
                                    side="o" if direction == "L->R" else "s")
    constraint = constraint.replace("is one of", "may be one of")

    if direction == "L->R":
        return (
            f"Context: The subject is fixed: {seed_value}. "
            f"The predicate is {pattern.p}.\n\n"
            f"Task: List all triples ({seed_value}, {pattern.p}, {pattern.o}) "
            f"that factually hold.\n"
            f"{constraint}\n"
            f"Only return values from the list above that are factually correct. "
            f"If the subject has no valid {pattern.p}, or none of the candidate "
            f"values apply, return an empty list."
        )
    else:
        return (
            f"Context: The object is fixed: {seed_value}. "
            f"The predicate is {pattern.p}.\n\n"
            f"Task: List all triples ({pattern.s}, {pattern.p}, {seed_value}) "
            f"that factually hold.\n"
            f"{constraint}\n"
            f"Only return values from the list above that are factually correct. "
            f"If no subject has {pattern.p} equal to {seed_value}, or none of "
            f"the candidate values apply, return an empty list."
        )


# ─── KeyCrank runner ──────────────────────────────────────────────────────────

def run_keycrank(pattern, env, llm, prompt_fn, label):
    """Run KeyCrank with a custom prompt function."""
    seeds_s = seedsOf(pattern.s, env)
    seeds_o = seedsOf(pattern.o, env)

    if not seeds_o or (seeds_s and len(seeds_s) <= len(seeds_o)):
        direction  = "L->R"
        iter_seeds = seeds_s
    else:
        direction  = "R->L"
        iter_seeds = seeds_o

    T = set()
    for k in iter_seeds:
        prompt   = prompt_fn(pattern, env, k, direction)
        response = llm.chat(build_messages(prompt))
        T = T | json_to_triples(response.text)

    updateEnv(env, T, pattern.s, pattern.o)
    print(f"  {label} → {len(T)} triples")
    return T


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  Experiment — KeyCrank prompt comparison")
    print(f"  Predicate : dbo:spouse  ({len(EXPECTED)} triples)")
    print(f"  Model     : {MODEL}")
    print(f"  Seeds ?x  : {len(SEEDS_X)}  |  Seeds ?y : {len(SEEDS_Y)}")
    print(f"{'='*60}")

    llm = AzureOpenAIClient()
    gt  = {t for t in EXPECTED if t.s in SEEDS_X}
    print(f"\n  GT restricted to seeds ?x : {len(gt)} triples")

    print("\n  --- seed ?x only ---")
    env_a = Environment(); env_a.set("?scientist", SEEDS_X)
    orig_x = run_keycrank(PATTERN, env_a, llm, gen_keycrank_original, "Original")

    env_b = Environment(); env_b.set("?scientist", SEEDS_X)
    impr_x = run_keycrank(PATTERN, env_b, llm, gen_keycrank_improved, "Improved")

    print("\n  --- seed ?x + ?y ---")
    env_c = Environment()
    env_c.set("?scientist", SEEDS_X)
    env_c.set("?spouse",    SEEDS_Y)
    orig_xy = run_keycrank(PATTERN, env_c, llm, gen_keycrank_original, "Original")

    env_d = Environment()
    env_d.set("?scientist", SEEDS_X)
    env_d.set("?spouse",    SEEDS_Y)
    impr_xy = run_keycrank(PATTERN, env_d, llm, gen_keycrank_improved, "Improved")

    report = Report(predicate="dbo:spouse — KeyCrank prompt comparison")
    for label, triples, expected in [
        ("Original (?x only)",  orig_x,  gt),
        ("Improved (?x only)",  impr_x,  gt),
        ("Original (?x + ?y)",  orig_xy, {t for t in gt if t.o in SEEDS_Y}),
        ("Improved (?x + ?y)",  impr_xy, {t for t in gt if t.o in SEEDS_Y}),
    ]:
        report.add(label, Evaluator(triples, expected).evaluate())

    report.print_table()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report.save_json(os.path.join(RESULTS_DIR, "keycrank_prompt.json"))
    report.save_csv(os.path.join(RESULTS_DIR,  "keycrank_prompt.csv"))


if __name__ == "__main__":
    main()