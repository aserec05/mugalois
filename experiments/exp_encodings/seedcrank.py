"""
experiments/exp_seedcrank_prompt/run.py
========================================

Research question
-----------------
Does changing "is one of" to "may be one of" + exit clause
reduce hallucinations in SeedCrank ?

Two variants on dbo:spouse — 43 triples.

Run
---
    python3 -m experiments.exp_seedcrank_prompt.run
"""

import os
import sys
import random

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mugalois.core.types import Triple, TriplePattern, Environment
from mugalois.core.helpers import updateEnv, seedsOf
from mugalois.core.parser import json_to_triples
from mugalois.core.prompts import build_messages, SYSTEM_PROMPT, _build_constraints, genIterativePrompt
from mugalois.llm.llm_client import AzureOpenAIClient

from evaluation.dataset import ScientistDataset
from evaluation.evaluator import Evaluator
from evaluation.report import Report

# ─── Config ───────────────────────────────────────────────────────────────────

PRED_URI = "http://dbpedia.org/ontology/spouse"
MODEL    = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini-2")
MAX_ITER = 3

GT_PATH     = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "ground_truth", "ground_truth.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

ds           = ScientistDataset(GT_PATH)
EXPECTED     = ds.T3(PRED_URI)
ALL_SUBJECTS = ds.subjects(PRED_URI)
ALL_OBJECTS  = ds.objects(PRED_URI)

random.seed(42)
SEEDS_X = set(random.sample(sorted(ALL_SUBJECTS), 10))
SEEDS_Y = set(random.sample(sorted(ALL_OBJECTS),  10))

PATTERN = TriplePattern("?scientist", "dbo:spouse", "?spouse")

# ─── Prompt variants ──────────────────────────────────────────────────────────

def gen_seedcrank_original(pattern, env):
    constraints = _build_constraints(pattern, env)
    return (
        f"Context: The following values are already known:\n"
        f"{constraints}\n\n"
        f"Task: List all triples ({pattern.s}, {pattern.p}, {pattern.o}) that factually hold.\n"
        f"- {pattern.s} must be the subject. {pattern.o} must be the object.\n"
        f"- Use exactly {pattern.p} as predicate. No variation.\n"
        f"- Only use values from the sets above."
    )


def gen_seedcrank_improved(pattern, env):
    constraints = _build_constraints(pattern, env)
    constraints = constraints.replace("is one of", "may be one of")
    return (
        f"Context: The following values are already known:\n"
        f"{constraints}\n\n"
        f"Task: List all triples ({pattern.s}, {pattern.p}, {pattern.o}) that factually hold.\n"
        f"- {pattern.s} must be the subject. {pattern.o} must be the object.\n"
        f"- Use exactly {pattern.p} as predicate. No variation.\n"
        f"- Only use values from the lists above that are factually correct. "
        f"If no triple holds, return an empty list."
    )


# ─── SeedCrank runner ─────────────────────────────────────────────────────────

def run_seedcrank(pattern, env, llm, prompt_fn, label, max_iter=MAX_ITER):
    T   = set()
    ctx = []

    for i in range(max_iter):
        prompt = prompt_fn(pattern, env) if i == 0 else genIterativePrompt(T)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *ctx,
            {"role": "user", "content": prompt},
        ]
        response = llm.chat(messages)
        T_new = json_to_triples(response.text)

        if T_new.issubset(T):
            break

        ctx.append({"role": "user",      "content": prompt})
        ctx.append({"role": "assistant", "content": response.text})
        T = T | T_new

    updateEnv(env, T, pattern.s, pattern.o)
    print(f"  {label} → {len(T)} triples")
    return T


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  Experiment — SeedCrank prompt comparison")
    print(f"  Predicate : dbo:spouse  ({len(EXPECTED)} triples)")
    print(f"  Model     : {MODEL}")
    print(f"  Seeds ?x  : {len(SEEDS_X)}  |  Seeds ?y : {len(SEEDS_Y)}")
    print(f"{'='*60}")

    llm = AzureOpenAIClient()
    gt_x  = {t for t in EXPECTED if t.s in SEEDS_X}
    gt_xy = {t for t in gt_x if t.o in SEEDS_Y}

    print(f"\n  GT ?x only  : {len(gt_x)} triples")
    print(f"  GT ?x + ?y  : {len(gt_xy)} triples")

    print("\n  --- seed ?x only ---")
    env_a = Environment(); env_a.set("?scientist", SEEDS_X)
    orig_x = run_seedcrank(PATTERN, env_a, llm, gen_seedcrank_original, "Original")

    env_b = Environment(); env_b.set("?scientist", SEEDS_X)
    impr_x = run_seedcrank(PATTERN, env_b, llm, gen_seedcrank_improved, "Improved")

    print("\n  --- seed ?x + ?y ---")
    env_c = Environment()
    env_c.set("?scientist", SEEDS_X)
    env_c.set("?spouse",    SEEDS_Y)
    orig_xy = run_seedcrank(PATTERN, env_c, llm, gen_seedcrank_original, "Original")

    env_d = Environment()
    env_d.set("?scientist", SEEDS_X)
    env_d.set("?spouse",    SEEDS_Y)
    impr_xy = run_seedcrank(PATTERN, env_d, llm, gen_seedcrank_improved, "Improved")

    report = Report(predicate="dbo:spouse — SeedCrank prompt comparison")
    for label, triples, gt in [
        ("Original (?x only)",  orig_x,  gt_x),
        ("Improved (?x only)",  impr_x,  gt_x),
        ("Original (?x + ?y)",  orig_xy, gt_xy),
        ("Improved (?x + ?y)",  impr_xy, gt_xy),
    ]:
        report.add(label, Evaluator(triples, gt).evaluate())

    report.print_table()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report.save_json(os.path.join(RESULTS_DIR, "seedcrank_prompt.json"))
    report.save_csv(os.path.join(RESULTS_DIR,  "seedcrank_prompt.csv"))


if __name__ == "__main__":
    main()