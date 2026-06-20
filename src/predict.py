"""Ollama predictor for Qwen3.5 chat completions."""

from __future__ import annotations

import os
import subprocess
import time
from typing import Any, Dict, List, Optional

import requests

LLM_MODEL = os.environ.get("LLM_MODEL", "sorc/qwen3.5-instruct:9b")
DISPLAY_MODEL_NAME = os.environ.get("DISPLAY_MODEL_NAME", LLM_MODEL)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
OLLAMA_MODELS = os.environ.get("OLLAMA_MODELS", "/root/.ollama")
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "-1")
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", "4096"))
MAX_TOKENS_LIMIT = int(os.environ.get("MAX_TOKENS_LIMIT", "16384"))
READY_TIMEOUT_SECONDS = 120
POLL_INTERVAL_SECONDS = 2
MIN_REQUEST_TIMEOUT_SECONDS = 120

_ollama_proc: Optional[subprocess.Popen[bytes]] = None


def _ollama_base_url() -> str:
    host = OLLAMA_HOST.strip()
    if host.startswith("http://") or host.startswith("https://"):
        return host.rstrip("/")
    return f"http://{host}"


def _ollama_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = OLLAMA_HOST
    env["OLLAMA_MODELS"] = OLLAMA_MODELS
    if OLLAMA_KEEP_ALIVE:
        env["OLLAMA_KEEP_ALIVE"] = OLLAMA_KEEP_ALIVE
    return env


def _ollama_is_ready() -> bool:
    try:
        response = requests.get(f"{_ollama_base_url()}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _ensure_ollama_running() -> None:
    global _ollama_proc

    if _ollama_is_ready():
        return

    if _ollama_proc is not None and _ollama_proc.poll() is not None:
        _ollama_proc = None

    if _ollama_proc is None:
        print("Starting ollama serve...", flush=True)
        _ollama_proc = subprocess.Popen(
            ["ollama", "serve"],
            env=_ollama_env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _ollama_proc.poll() is not None:
            raise RuntimeError("ollama serve exited before becoming ready")
        if _ollama_is_ready():
            print("Ollama is ready.", flush=True)
            return
        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(f"Ollama did not become ready within {READY_TIMEOUT_SECONDS}s")


def _request_timeout(max_tokens: int) -> int:
    return max(MIN_REQUEST_TIMEOUT_SECONDS, max_tokens // 4 + 60)


class Predictor:
    """Single-model Ollama chat predictor."""

    def __init__(self) -> None:
        self.model_name = DISPLAY_MODEL_NAME

    def setup(self) -> None:
        _ensure_ollama_running()
        print(f"Warming up {LLM_MODEL}...", flush=True)
        self.predict(
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.0,
            max_tokens=1,
            think=False,
        )
        print(f"Model {self.model_name} ready.", flush=True)

    def predict(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        think: bool = False,
    ) -> Dict[str, Any]:
        _ensure_ollama_running()

        budget = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
        budget = min(max(budget, 1), MAX_TOKENS_LIMIT)

        options: Dict[str, Any] = {
            "temperature": temperature,
            "num_predict": budget,
            "num_ctx": OLLAMA_NUM_CTX,
        }
        if top_p is not None:
            options["top_p"] = top_p
        if top_k is not None:
            options["top_k"] = top_k

        payload: Dict[str, Any] = {
            "model": LLM_MODEL,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": options,
        }

        response = requests.post(
            f"{_ollama_base_url()}/api/chat",
            json=payload,
            timeout=_request_timeout(budget),
        )
        response.raise_for_status()
        data = response.json()

        message = data.get("message") or {}
        content = (message.get("content") or "").strip()

        return {
            "content": content,
            "model": self.model_name,
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count"),
                "completion_tokens": data.get("eval_count"),
            },
        }
