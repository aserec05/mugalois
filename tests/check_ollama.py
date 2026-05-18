"""Quick check that Ollama is reachable. Run: python scripts/check_ollama.py"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama

base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
model    = os.getenv("OLLAMA_MODEL", "phi3")

print(f"Connecting to {base_url} — model: {model}")
llm = ChatOllama(base_url=base_url, model=model, temperature=0)
response = llm.invoke("Say hello in one sentence.")
print(response.content)
