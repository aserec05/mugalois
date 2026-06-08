"""
experiments/template1/exp_preliminaries/run.py

Preliminary experiments for Template 1.
Tests all URI encodings x prompt encodings for q1, q4, q5.
N=3 runs per configuration.

Usage
-----
    python run.py                  # Azure (default)
    python run.py --llm ollama --model phi3
    python run.py --dry-run        # MockLLM, no API calls
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import product
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from mugalois.core.types import TriplePattern
from mugalois.core.prompts import (
    genNLPrompt, genSPARQLPrompt,
    genTripleScanPrompt, genTripleScanIterativePrompt,
    genValueScanPrompt, genValueScanIterativePrompt,
    build_messages, build_value_messages,
)
from mugalois.llm.llm_client import OllamaClient, AzureOpenAIClient, MockLLM
from mugalois.core.parser import json_to_triples, json_to_values
from evaluation.metrics import Metrics, AggregatedScores
from evaluation.evaluator import Evaluator
from evaluation.report import Report


# ── config ────────────────────────────────────────────────────────────────────

QUERIES_PATH = Path(__file__).resolve().parents[1] / "queries.json"
RESULTS_DIR  = Path(__file__).resolve().parent / "results"

N_RUNS   = 3
MAX_ITER = 3

QUERIES    = ["q4", "q5"]
TEMPLATES  = ["T1a"]
URI_STYLES = ["local", "prefixed", "full", "nl"]
ENCODINGS  = ["pattern", "constrained", "sparql"]

# URI prefix maps
URI_MAP = {
    "local":    {"dbr": ":",    "dbo": ":"},
    "prefixed": {"dbr": "dbr:", "dbo": "dbo:"},
    "full":     {"dbr": "http://dbpedia.org/resource/",
                 "dbo": "http://dbpedia.org/ontology/"},
    "nl":       {},
}

PRED_NL = {
    "dbo:spouse":          "spouse",
    "dbo:doctoralAdvisor": "doctoral advisor",
    "dbo:award":           "award",
}

RES_NL = {
    "dbr:Albert_Einstein":  "Albert Einstein",
    "dbr:Werner_Heisenberg":"Werner Heisenberg",
    "dbr:Mileva_Marić":     "Mileva Marić",
    "dbr:Marie_Curie":      "Marie Curie",
    "dbr:Turing_Award":     "Turing Award",
}


# ── URI conversion ────────────────────────────────────────────────────────────

def _apply_uri(term: str, style: str) -> str:
    if not isinstance(term, str) or term.startswith("?"):
        return term
    if style == "nl":
        return PRED_NL.get(term) or RES_NL.get(term) or term
    for prefix in ("dbr", "dbo"):
        if term.startswith(prefix + ":"):
            local = term[len(prefix) + 1:]
            if style == "full":
                ns = ("http://dbpedia.org/resource/" if prefix == "dbr"
                      else "http://dbpedia.org/ontology/")
                return f"<{ns}{local}>"
            repl = URI_MAP[style][prefix]
            return f"{repl}{local}"
    return term


def _make_pattern(raw: dict, style: str) -> TriplePattern:
    return TriplePattern(
        s=_apply_uri(raw["s"], style),
        p=_apply_uri(raw["p"], style),
        o=_apply_uri(raw["o"], style),
    )


def _make_sparql(sparql_template: str, style: str) -> str:
    if style == "prefixed":
        return sparql_template
    result = sparql_template
    for prefix in ("dbr", "dbo"):
        def replace_token(m, pfx=prefix):
            local = m.group(1)
            return _apply_uri(f"{pfx}:{local}", style)
        result = re.sub(rf"{prefix}:([\w_À-ÿ]+)", replace_token, result)
    return result


# ── Runners ───────────────────────────────────────────────────────────────────

def run_nl(nl_prompt: str, llm) -> set[str]:
    resp = llm.chat(build_value_messages(genNLPrompt(nl_prompt)))
    return json_to_values(resp.text)


def run_sparql(sparql_query: str, llm) -> set[str]:
    resp = llm.chat(build_value_messages(genSPARQLPrompt(sparql_query)))
    return json_to_values(resp.text)


def run_triple_scan(pattern: TriplePattern, var_side: str,
                    encoding: str, llm) -> set[str]:
    found = set()
    ctx   = []
    sys_msg = build_messages("")[0]
    for i in range(MAX_ITER):
        prompt = (genTripleScanPrompt(pattern, encoding) if i == 0
                  else genTripleScanIterativePrompt(found))
        resp  = llm.chat([sys_msg, *ctx, {"role": "user", "content": prompt}])
        new   = {
            getattr(t, var_side)
            for t in json_to_triples(resp.text)
        }
        if new.issubset(found):
            break
        ctx.append({"role": "user",      "content": prompt})
        ctx.append({"role": "assistant", "content": resp.text})
        found |= new
    return found


def run_value_scan(pattern: TriplePattern, encoding: str, llm) -> set[str]:
    found = set()
    ctx   = []
    sys_msg = build_value_messages("")[0]
    for i in range(MAX_ITER):
        prompt = (genValueScanPrompt(pattern, encoding) if i == 0
                  else genValueScanIterativePrompt(found))
        resp  = llm.chat([sys_msg, *ctx, {"role": "user", "content": prompt}])
        new   = json_to_values(resp.text)
        if new.issubset(found):
            break
        ctx.append({"role": "user",      "content": prompt})
        ctx.append({"role": "assistant", "content": resp.text})
        found |= new
    return found


# ── Aggregation ───────────────────────────────────────────────────────────────

def _aggregate(run_fn, gt: set[str], n: int,
               last_actual: list) -> AggregatedScores:
    """Run n times, aggregate scores, store last actual for FP/FN."""
    scores = []
    for i in range(n):
        actual = run_fn()
        if i == n - 1:
            last_actual.append(actual)
        scores.append(Evaluator(actual, gt, mode="values").evaluate())
    return AggregatedScores.from_runs(scores)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(llm):
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for qid, template in product(QUERIES, TEMPLATES):
        q   = queries[qid]
        tpl = q[template]
        gt  = set(q["ground_truth"])

        raw_pattern = tpl["pattern"]
        # variable side for TripleScan projection
        var_side    = "s" if raw_pattern["s"].startswith("?") else "o"
        nl_prompt   = q["nl_prompt"]
        sparql_base = tpl["sparql_prompt"]

        report = Report(
            query_id=qid, template=template,
            predicate=raw_pattern["p"]
        )

        print(f"\n{'='*60}")
        print(f"  Running {qid} / {template}")
        print(f"{'='*60}")

        # ── NL baseline ──────────────────────────────────────────────
        last = []
        agg  = _aggregate(lambda: run_nl(nl_prompt, llm), gt, N_RUNS, last)
        report.add("NL", agg, actual=last[0], expected=gt)

        # ── SPARQL × URI ─────────────────────────────────────────────
        for uri in URI_STYLES:
            sparql_q = _make_sparql(sparql_base, uri)
            last = []
            agg  = _aggregate(
                lambda sq=sparql_q: run_sparql(sq, llm), gt, N_RUNS, last)
            report.add(f"SPARQL_{uri}", agg, actual=last[0], expected=gt)

        # ── TripleScan × encoding × URI ──────────────────────────────
        for enc, uri in product(ENCODINGS, URI_STYLES):
            pattern = _make_pattern(raw_pattern, uri)
            last = []
            agg  = _aggregate(
                lambda p=pattern, e=enc: run_triple_scan(p, var_side, e, llm),
                gt, N_RUNS, last)
            report.add(f"TripleScan_{enc}_{uri}", agg,
                       actual=last[0], expected=gt)

        # ── ValueScan × encoding × URI ───────────────────────────────
        for enc, uri in product(ENCODINGS, URI_STYLES):
            pattern = _make_pattern(raw_pattern, uri)
            last = []
            agg  = _aggregate(
                lambda p=pattern, e=enc: run_value_scan(p, e, llm),
                gt, N_RUNS, last)
            report.add(f"ValueScan_{enc}_{uri}", agg,
                       actual=last[0], expected=gt)

        # ── Save ─────────────────────────────────────────────────────
        report.print_table()
        stem = f"{qid}_{template}"
        report.save_json(RESULTS_DIR / f"{stem}.json")
        report.save_csv(RESULTS_DIR  / f"{stem}.csv")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm",   default="azure",
                        choices=["azure", "ollama", "mock"])
    parser.add_argument("--model", default="phi3", help="Ollama model name")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run or args.llm == "mock":
        llm = MockLLM()
    elif args.llm == "ollama":
        llm = OllamaClient(model=args.model)
    else:
        llm = AzureOpenAIClient()

    main(llm)