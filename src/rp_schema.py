"""Input validation schema for scalar RunPod job fields."""

VALID_REASONING_EFFORTS = frozenset({"none", "low", "medium", "high"})

INPUT_VALIDATIONS = {
    "temperature": {
        "type": float,
        "required": False,
        "default": None,
    },
    "max_tokens": {
        "type": int,
        "required": False,
        "default": None,
    },
    "top_p": {
        "type": float,
        "required": False,
        "default": None,
    },
    "top_k": {
        "type": int,
        "required": False,
        "default": None,
    },
    "think": {
        "type": bool,
        "required": False,
        "default": None,
    },
    "reasoning_effort": {
        "type": str,
        "required": False,
        "default": None,
    },
}
