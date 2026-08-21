# µ-Galois

**SPARQL Optimization and Query Planning with Large Language Models**

µ-Galois executes SPARQL queries over RDF knowledge that is not yet represented in any structured knowledge graph, using an LLM as a retrieval source. It combines logical and physical optimization — confidence signals, structural analysis, cardinality estimation — to choose the best execution strategy at runtime.

> *Master's thesis — Thomas Ceresa*
> *Università degli Studi di Padova × Université Claude Bernard Lyon 1, 2025–2026*

---

## Installation

```bash
git clone https://github.com/aserec05/mugalois.git
cd mugalois
pip install -r requirements.txt
```

Copy and fill the environment file:

```bash
cp .env.example .env
```

---

## LLM Configuration

µ-Galois uses **Azure OpenAI (GPT-4o-mini)** by default. Set the following variables in your `.env`:

```env
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_BASE_URL=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-02-01
```

### Quick start with Azure OpenAI

```python
from mugalois.llm.llm_client import AzureOpenAIClient

llm = AzureOpenAIClient()
resp = llm.chat([{"role": "user", "content": "Who were Albert Einstein's spouses?"}])
print(resp.text)
print(f"Tokens used: {resp.usage_tokens}  |  Latency: {resp.latency_s:.2f}s")
```

### Local Ollama (no API key required)

```bash
ollama pull phi3
```

```python
from mugalois.llm.llm_client import OllamaClient

llm = OllamaClient(model="phi3")
resp = llm.chat([{"role": "user", "content": "Who were Albert Einstein's spouses?"}])
print(resp.text)
```

### Dry run (no network)

```python
from mugalois.llm.llm_client import MockLLM

llm = MockLLM(canned='["Mileva Marić", "Elsa Einstein"]')
```

---

## Running Experiments

```bash
# Run all templates, all models (5 runs each)
python3 -m experiments.run_general --all

# Single template
python3 -m experiments.run_general --template T4

# Single query, dry run
python3 -m experiments.run_general --template T1 --query q5 --dry-run

# New calibrated models (C_05, FM_05)
python3 -m experiments.run_new_models --all
```

Results are saved in `experiments/results/`.

---

## Generating Figures

```bash
# RQ1 — Decomposition vs Holistic
python3 -m experiments.plot_rq1

# RQ2 — Confidence vs Structure
python3 -m experiments.plot_rq2

# RQ3 — Closure strategies on plateau queries
python3 -m experiments.plot_rq3

# RQ4 — Quality vs Cost (2×2 figure)
python3 -m experiments.plot_rq4_final
```

Figures are saved as PDF and PNG in `experiments/figures/`.

---

## Project Structure

```
mugalois/
├── src/mugalois/
│   ├── llm/            # LLM clients (Azure, Ollama, Mock, ...)
│   ├── core/           # Types, prompts, parsers, helpers
│   ├── scans/          # LLMScan, KeyScan, SeedScan, TripleScan
│   ├── paths/          # MuGaloisMultiPath (T4)
│   ├── rec/            # MuGaloisRec, fixpoint (T5)
│   ├── choice/         # MuGaloisChoice (T6)
│   ├── hybrid/         # HybridPlanner v4 (T7)
│   └── closures/       # Motivational, Alphabet, Socratic, Contrast, Coach
├── experiments/
│   ├── run_general.py  # Main ablation runner
│   ├── run_new_models.py
│   ├── plot_rq*.py     # Figure scripts
│   ├── results/        # JSON results per query
│   └── figures/        # Generated PDF/PNG
├── evaluation/
│   ├── metrics.py      # Precision / Recall / F1 with fuzzy matching
│   ├── evaluator.py
│   └── report.py
├── requirements.txt
└── .env.example
```

---


---

## Citation

```bibtex
@mastersthesis{ceresa2026mugalois,
  author  = {Thomas Ceresa},
  title   = {You Can Know It: SPARQL Optimization and Query Planning with Large Language Models},
  school  = {Università degli Studi di Padova / Université Claude Bernard Lyon 1},
  year    = {2026}
}
```
