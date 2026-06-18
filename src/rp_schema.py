"""Input validation schema for scalar RunPod job fields."""

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
    "reasoning_effort": {
        "type": str,
        "required": False,
        "default": None,
    },
}
