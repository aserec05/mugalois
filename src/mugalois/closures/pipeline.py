# src/mugalois/closures/pipeline.py
from __future__ import annotations

from mugalois.closures.base import Closure
from mugalois.core.types import TriplePattern
from mugalois.core.prompts import genTripleScanPrompt, SYSTEM_PROMPT
from mugalois.core.parser import json_to_triples


class ClosurePipeline:
    def __init__(self, closures: list[Closure], n_break: int = 2, max_iter: int = 15):
        self.closures = closures
        self.n_break  = n_break
        self.max_iter = max_iter

    @property
    def name(self) -> str:
        names = "+".join(c.name for c in self.closures) if self.closures else "Standard"
        return f"Pipeline({names})"

    def run(self, pattern: TriplePattern, tracker) -> tuple[set, int, dict]:
        T             = set()
        ctx           = []
        sys_msg       = {"role": "system", "content": SYSTEM_PROMPT}
        closure_idx   = 0
        empty_streak  = 0
        n_iter        = 0
        stats         = {c.name: 0 for c in self.closures}

        for i in range(self.max_iter):
            n_iter = i + 1

            if i == 0:
                prompt = genTripleScanPrompt(pattern, encoding="pattern")
            elif self.closures:
                idx     = min(closure_idx, len(self.closures) - 1)
                closure = self.closures[idx]
                prompt  = closure.prompt(pattern, T, tracker)
                if empty_streak > 0:
                    stats[closure.name] = stats.get(closure.name, 0) + 1
            else:
                # no closures — standard iterative
                from mugalois.core.prompts import genTripleScanIterativePrompt
                prompt = genTripleScanIterativePrompt(T)

            messages = [sys_msg, *ctx, {"role": "user", "content": prompt}]
            resp     = tracker.chat(messages)
            T_new    = json_to_triples(resp.text)

            if T_new.issubset(T):
                empty_streak += 1
                closure_idx  += 1
                if empty_streak >= self.n_break:
                    break
            else:
                empty_streak = 0
                closure_idx  = 0
                ctx.append({"role": "user",      "content": prompt})
                ctx.append({"role": "assistant", "content": resp.text})
                T |= T_new

        return T, n_iter, stats
