
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from mugalois.core.types import TriplePattern, Environment
from mugalois.scans.ochestror_scan import LLMScan

env = Environment()
env.set("?x", {"Pope Paul VI"})
result = LLMScan(TriplePattern("?x", "successor as Pope", "?y"), env, Environment(), llm)
print(result)