"""Pre-pull Ollama model at image build time."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import requests

LLM_MODEL = os.environ.get("LLM_MODEL", "sorc/qwen3.5-instruct:9b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
OLLAMA_MODELS = os.environ.get("OLLAMA_MODELS", "/root/.ollama")
READY_TIMEOUT_SECONDS = 120
POLL_INTERVAL_SECONDS = 2


def _ollama_base_url() -> str:
    host = OLLAMA_HOST.strip()
    if host.startswith("http://") or host.startswith("https://"):
        return host.rstrip("/")
    return f"http://{host}"


def _wait_for_ollama(proc: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    tags_url = f"{_ollama_base_url()}/api/tags"

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("ollama serve exited before becoming ready")
        try:
            response = requests.get(tags_url, timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(f"Ollama did not become ready within {READY_TIMEOUT_SECONDS}s")


def main() -> None:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = OLLAMA_HOST
    env["OLLAMA_MODELS"] = OLLAMA_MODELS

    print(f"Starting ollama serve (models dir: {OLLAMA_MODELS})...", flush=True)
    proc = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        _wait_for_ollama(proc)
        print(f"Pulling {LLM_MODEL}...", flush=True)
        subprocess.run(
            ["ollama", "pull", LLM_MODEL],
            env=env,
            check=True,
        )
        print(f"Finished pulling {LLM_MODEL}.", flush=True)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Model pull failed: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
