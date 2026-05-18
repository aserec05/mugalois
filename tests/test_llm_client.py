import os
from dotenv import load_dotenv
load_dotenv()

from mugalois.llm import OllamaClient

llm = OllamaClient(
    model=os.getenv("OLLAMA_MODEL", "phi3"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
)

resp = llm.chat([{"role": "user", "content": "Say hello in one sentence."}])
print(resp.text)
print(f"tokens={resp.usage_tokens}  latency={resp.latency_s:.2f}s")