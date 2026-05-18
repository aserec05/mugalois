# µ-Galois

LLM-based RDF triplet extraction — implementation of LLMTripletScan.

## Installation

**Requirements:** Python ≥ 3.11, [Ollama](https://ollama.com) installed and running.

```bash
git clone https://github.com/aserec05/mugalois.git
cd mugalois

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

Pull a model in Ollama:

```bash
ollama pull phi3
```

Copy the config and edit if needed:

```bash
cp .env.example .env
```

Check the connection:

```bash
python tests/test_llm_client.py
```

## Project structure

```
src/
  mugalois/
    llm/
      llm_client.py
tests/
  test_llm_client.py
```

## Note on Ollama URL

- **Native / WSL:** `http://localhost:11434`
- **Docker:** `http://host.docker.internal:11434`

Set `OLLAMA_BASE_URL` in `.env` accordingly.
