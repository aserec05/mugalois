"""TODO : - demain implémenter les algos
- mercredi voir avec gm pour les tockens, les acces et rendre dispo ce client"""

import os, json, time, random
from typing import List, Dict, Any, Optional
import requests

# from openai import OpenAI
# from ibm_watsonx_ai import Credentials
# from ibm_watsonx_ai.foundation_models import ModelInference
# from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

LLM_SEED = "7"

''' clés à remplir mercredi '''
# OPENROUTER_API_KEY=""
# OPENROUTER_MODEL="meta-llama/llama-3.3-70b-instruct"
# AZURE_INFERENCE_ENDPOINT=""
# AZURE_INFERENCE_KEY=""
# AZURE_INFERENCE_MODEL="Llama-3.3-70B-Instruct"
# AZURE_GROK_API_KEY=""
# AZURE_GROK_TARGET_URI=""
# AZURE_OPENAI_API_KEY=""
# AZURE_OPENAI_BASE_URL=""
# AZURE_OPENAI_DEPLOYMENT=""
# WATSONX_API_KEY=""
# WATSONX_URL="https://us-south.ml.cloud.ibm.com"
# WATSONX_PROJECT_ID=""
# WATSONX_MODEL_ID="meta-llama/llama-3-3-70b-instruct"


class LLMResponse:
    def __init__(self, text: str, usage_tokens: int = 0, latency_s: float = 0.0):
        self.text = text
        self.usage_tokens = usage_tokens
        self.latency_s = latency_s


class BaseLLM:
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        raise NotImplementedError


class OllamaClient(BaseLLM):
    '''
    Local Ollama via its OpenAI-compatible endpoint (/v1/chat/completions).
    No API key required.
    '''
    def __init__(
        self,
        model: str = "phi3",
        base_url: str = "http://localhost:11434/v1",
        timeout: int = 18000,
    ):
        self.model = model
        self.base_url = os.getenv("OLLAMA_BASE_URL", base_url).rstrip("/")
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        r = requests.post(
            f"{self.base_url}/chat/completions",
            json={"model": self.model, "messages": messages, "temperature": 0},
            headers={"Content-Type": "application/json", "Authorization": "Bearer ollama"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            usage_tokens=data.get("usage", {}).get("total_tokens"),
        )


# class OpenRouterClient(BaseLLM):
#     '''
#     OpenRouter — many models, pay-per-token, OpenAI-compatible SDK.
#     '''
#     def __init__(self, model=None):
#         if not OPENROUTER_API_KEY:
#             raise RuntimeError("OPENROUTER_API_KEY is required.")
#         self.model = model or OPENROUTER_MODEL
#         self._client = OpenAI(
#             base_url="https://openrouter.ai/api/v1",
#             api_key=OPENROUTER_API_KEY,
#             default_headers={"HTTP-Referer": "https://example.com", "X-Title": "GALOIS"},
#         )
#     def chat(self, messages, **kwargs) -> LLMResponse:
#         t0 = time.time()
#         resp = self._client.chat.completions.create(
#             model=self.model, messages=messages,
#             temperature=0.0, top_p=1.0, seed=int(LLM_SEED), max_tokens=10000, **kwargs,
#         )
#         text = resp.choices[0].message.content or ""
#         usage = getattr(resp, "usage", None)
#         tokens = ((getattr(usage, "prompt_tokens", 0) or 0) + (getattr(usage, "completion_tokens", 0) or 0)) if usage else 0
#         return LLMResponse(text=text, usage_tokens=tokens, latency_s=time.time() - t0)


# class FoundryOpenAIClient(BaseLLM):
#     '''
#     Azure AI Foundry (Serverless) — OpenAI-compatible endpoint.
#     '''
#     def __init__(self):
#         self.model = AZURE_INFERENCE_MODEL
#         self._client = OpenAI(base_url=AZURE_INFERENCE_ENDPOINT.rstrip("/"), api_key=AZURE_INFERENCE_KEY)
#     def chat(self, messages, **kwargs) -> LLMResponse:
#         t0 = time.time()
#         resp = self._client.chat.completions.create(
#             model=self.model, messages=messages,
#             temperature=0.0, top_p=1.0, seed=int(LLM_SEED), max_tokens=10000,
#         )
#         text = resp.choices[0].message.content or ""
#         usage = getattr(resp, "usage", None)
#         tokens = ((getattr(usage, "prompt_tokens", 0) or 0) + (getattr(usage, "completion_tokens", 0) or 0)) if usage else 0
#         return LLMResponse(text=text, usage_tokens=tokens, latency_s=time.time() - t0)


# class AzureGrokClient(BaseLLM):
#     '''
#     Azure Grok — raw HTTP with retry logic (transient errors: 408, 429, 5xx).
#     '''
#     def __init__(self, target_uri=None, api_key=None, timeout=300,
#                  temperature=0.0, top_p=1.0, max_tokens=2000, seed=None, max_retries=5):
#         self.target_uri = (target_uri or AZURE_GROK_TARGET_URI).strip()
#         self.api_key = api_key or AZURE_GROK_API_KEY
#         self.timeout = timeout
#         self.temperature = temperature; self.top_p = top_p; self.max_tokens = max_tokens
#         self.seed = int(seed) if seed is not None else int(LLM_SEED)
#         self.max_retries = max_retries
#     def chat(self, messages, **kwargs) -> LLMResponse:
#         payload = {"messages": messages, "temperature": self.temperature,
#                    "top_p": self.top_p, "max_tokens": self.max_tokens, "seed": self.seed}
#         payload.update(kwargs)
#         headers = {"Content-Type": "application/json", "api-key": self.api_key}
#         last_err = None
#         for attempt in range(1, self.max_retries + 1):
#             t0 = time.time()
#             try:
#                 resp = requests.post(self.target_uri, headers=headers, json=payload, timeout=(30, self.timeout))
#                 resp.raise_for_status()
#                 data = resp.json()
#                 content = data["choices"][0]["message"]["content"]
#                 usage = data.get("usage", {}) or {}
#                 tokens = usage.get("total_tokens") or ((usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0))
#                 return LLMResponse(text=content, usage_tokens=tokens, latency_s=time.time() - t0)
#             except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
#                 last_err = e
#             except requests.exceptions.HTTPError as e:
#                 status = e.response.status_code if e.response is not None else None
#                 if status in {408, 429, 500, 502, 503, 504}: last_err = RuntimeError(f"HTTP {status}")
#                 else: raise
#             if attempt < self.max_retries:
#                 time.sleep(min(20, 2 ** (attempt - 1)) + random.uniform(0, 0.5))
#         raise RuntimeError(f"AzureGrok failed after {self.max_retries} attempts: {last_err}")


# class WatsonxClient(BaseLLM):
#     '''
#     IBM watsonx.ai — official SDK, same message format as OpenAI chat.
#     '''
#     def __init__(self, model_id=None, api_key=None, url=None, project_id=None,
#                  max_tokens=10000, temperature=0.0, top_p=1.0):
#         self._params = TextChatParameters(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
#         self._model = ModelInference(
#             model_id=model_id or WATSONX_MODEL_ID,
#             params=self._params,
#             credentials=Credentials(api_key=api_key or WATSONX_API_KEY, url=url or WATSONX_URL),
#             project_id=project_id or WATSONX_PROJECT_ID,
#         )
#     def chat(self, messages, **kwargs) -> LLMResponse:
#         t0 = time.time()
#         resp = self._model.chat(messages=messages, params=self._params, **kwargs)
#         try: text = resp["choices"][0]["message"]["content"]
#         except Exception: text = str(resp)
#         try:
#             u = resp.get("usage", {})
#             tokens = u.get("total_tokens") or u.get("generated_token_count", 0) + u.get("input_token_count", 0)
#         except Exception: tokens = 0
#         return LLMResponse(text=text, usage_tokens=tokens, latency_s=time.time() - t0)


# class OpenAIClient(BaseLLM):
#     '''
#     OpenAI or Azure OpenAI — uses the Responses API (azure=True for Azure deployment).
#     '''
#     def __init__(self, model=None, azure=False):
#         self.azure = azure
#         if azure:
#             self._client = OpenAI(base_url=AZURE_OPENAI_BASE_URL, api_key=AZURE_OPENAI_API_KEY)
#             self.model = AZURE_OPENAI_DEPLOYMENT
#         else:
#             self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
#             self.model = model or "gpt-4o-mini"
#     def chat(self, messages, **kwargs) -> LLMResponse:
#         t0 = time.time()
#         inp = messages[0]["content"] if (self.azure and len(messages) == 1 and messages[0]["role"] == "user") \
#               else [{"role": m["role"], "content": m["content"]} for m in messages]
#         resp = self._client.responses.create(
#             model=self.model, input=inp,
#             temperature=0.0, top_p=1.0, max_output_tokens=100000, **kwargs,
#         )
#         content = getattr(resp, "output_text", None) or str(resp)
#         try:
#             u = resp.usage
#             tokens = getattr(u, "total_tokens", None) or (getattr(u, "input_tokens", 0) or 0) + (getattr(u, "output_tokens", 0) or 0)
#         except Exception: tokens = 0
#         return LLMResponse(text=content, usage_tokens=tokens, latency_s=time.time() - t0)


class MockLLM(BaseLLM):
    '''Fake client for dry runs — returns minimal JSON without any network call.'''
    def __init__(self, canned: Optional[str] = None):
        self.canned = canned or "[]"
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        last = messages[-1]["content"].lower()
        if "confidence" in last:
            return LLMResponse(text=json.dumps({"confidence": 0.6}), usage_tokens=1, latency_s=0.01)
        if "list more" in last and "empty" in last:
            return LLMResponse(text="[]", usage_tokens=1, latency_s=0.01)
        return LLMResponse(text=self.canned, usage_tokens=1, latency_s=0.01)