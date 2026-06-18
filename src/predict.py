"""vLLM predictor for Qwen3.5-9B chat completions."""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

from vllm import LLM, SamplingParams

DEFAULT_MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
DEFAULT_MODEL_NAME = os.environ.get("DISPLAY_MODEL_NAME", "Qwen3.5-9B")
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", "4096"))
MAX_TOKENS_LIMIT = int(os.environ.get("MAX_TOKENS_LIMIT", "16384"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))
QUANTIZATION = os.environ.get("QUANTIZATION", "none").strip().lower()
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "false").lower() in (
    "1",
    "true",
    "yes",
)


class Predictor:
    """Single-model vLLM chat predictor."""

    def __init__(self) -> None:
        self.llm: Optional[LLM] = None
        self.model_name = DEFAULT_MODEL_NAME
        self._load_lock = threading.Lock()

    def setup(self) -> None:
        self._load_model()

    def _load_model(self) -> LLM:
        if self.llm is not None:
            return self.llm

        with self._load_lock:
            if self.llm is not None:
                return self.llm

            model_path = DEFAULT_MODEL_DIR
            if not os.path.isdir(model_path):
                model_path = os.environ.get("HF_MODEL", "Qwen/Qwen3.5-9B")

            llm_kwargs: Dict[str, Any] = {
                "model": model_path,
                "max_model_len": MAX_MODEL_LEN,
                "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
                "trust_remote_code": True,
                "language_model_only": True,
            }
            if QUANTIZATION and QUANTIZATION not in ("none", "false", "off"):
                llm_kwargs["quantization"] = QUANTIZATION

            print(
                f"Loading model from {model_path} "
                f"(quantization={QUANTIZATION or 'none'}, max_model_len={MAX_MODEL_LEN})...",
                flush=True,
            )
            self.llm = LLM(**llm_kwargs)
            print(f"Model {self.model_name} loaded.", flush=True)
            return self.llm

    def predict(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        llm = self._load_model()
        budget = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
        budget = min(max(budget, 1), MAX_TOKENS_LIMIT)

        sampling_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": budget,
        }
        if top_p is not None:
            sampling_kwargs["top_p"] = top_p
        if top_k is not None:
            sampling_kwargs["top_k"] = top_k

        sampling_params = SamplingParams(**sampling_kwargs)
        chat_template_kwargs = {"enable_thinking": ENABLE_THINKING}

        outputs = llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            chat_template_kwargs=chat_template_kwargs,
        )

        if not outputs:
            raise RuntimeError("vLLM returned no outputs")

        request_output = outputs[0]
        if not request_output.outputs:
            raise RuntimeError("vLLM returned no completion choices")

        completion = request_output.outputs[0]
        content = (completion.text or "").strip()

        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        if getattr(request_output, "prompt_token_ids", None) is not None:
            prompt_tokens = len(request_output.prompt_token_ids)
        if getattr(completion, "token_ids", None) is not None:
            completion_tokens = len(completion.token_ids)

        return {
            "content": content,
            "model": self.model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }
