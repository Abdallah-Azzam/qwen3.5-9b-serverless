"""RunPod serverless handler for Qwen3.5-9B chat via Ollama."""

import os
from typing import Any, Dict, List, Optional

import runpod
from predict import Predictor
from rp_schema import INPUT_VALIDATIONS, VALID_REASONING_EFFORTS
from runpod.serverless.utils import rp_debugger
from runpod.serverless.utils.rp_validator import validate

MODEL = Predictor()
MODEL.setup()

VALID_ROLES = frozenset({"system", "user", "assistant"})


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes")


def _validate_messages(messages: Any) -> Optional[str]:
    if not messages:
        return "messages is required and must be a non-empty list"
    if not isinstance(messages, list):
        return "messages must be a list"

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return f"messages[{index}] must be an object"
        role = message.get("role")
        if role not in VALID_ROLES:
            return (
                f"messages[{index}].role must be one of: "
                f"{', '.join(sorted(VALID_ROLES))}"
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return f"messages[{index}].content must be a non-empty string"

    return None


def _validate_reasoning_effort(reasoning_effort: Any) -> Optional[str]:
    if reasoning_effort is None:
        return None
    if not isinstance(reasoning_effort, str):
        return "reasoning_effort must be a string"
    if reasoning_effort.lower() not in VALID_REASONING_EFFORTS:
        return (
            "reasoning_effort must be one of: "
            + ", ".join(sorted(VALID_REASONING_EFFORTS))
        )
    return None


def _resolve_think(job_input: dict) -> bool:
    if job_input.get("think") is not None:
        return bool(job_input["think"])

    reasoning_effort = job_input.get("reasoning_effort")
    if reasoning_effort is not None:
        return reasoning_effort.lower() != "none"

    return _env_bool("ENABLE_THINKING", False)


def _apply_env_defaults(job_input: dict) -> dict:
    resolved = dict(job_input)
    if resolved.get("temperature") is None:
        resolved["temperature"] = _env_float("DEFAULT_TEMPERATURE", 0.7)
    if resolved.get("max_tokens") is None:
        resolved["max_tokens"] = _env_int("DEFAULT_MAX_TOKENS", 4096)
    if resolved.get("top_p") is None:
        resolved["top_p"] = _env_float("DEFAULT_TOP_P", 0.8)
    return resolved


@rp_debugger.FunctionTimer
def run_llm_job(job: Dict[str, Any]) -> Dict[str, Any]:
    job_input = job["input"]

    with rp_debugger.LineTimer("validation_step"):
        messages_error = _validate_messages(job_input.get("messages"))
        if messages_error:
            return {"error": messages_error}

        input_validation = validate(job_input, INPUT_VALIDATIONS)
        if "errors" in input_validation:
            return {"error": str(input_validation["errors"])}
        job_input = _apply_env_defaults(input_validation["validated_input"])

        reasoning_error = _validate_reasoning_effort(job_input.get("reasoning_effort"))
        if reasoning_error:
            return {"error": reasoning_error}

    messages: List[Dict[str, str]] = [
        {"role": str(msg["role"]), "content": str(msg["content"])}
        for msg in job_input["messages"]
    ]
    think = _resolve_think(job_input)

    with rp_debugger.LineTimer("prediction_step"):
        try:
            result = MODEL.predict(
                messages=messages,
                temperature=job_input["temperature"],
                max_tokens=job_input["max_tokens"],
                top_p=job_input.get("top_p"),
                top_k=job_input.get("top_k"),
                think=think,
            )
        except Exception as exc:
            return {"error": f"Inference failed: {exc}"}

    return {
        "content": result.get("content") or "",
        "model": result.get("model"),
        "usage": result.get("usage"),
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": run_llm_job})
