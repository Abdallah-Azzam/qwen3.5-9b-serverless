"""Pre-download model weights at image build time."""

import os

from huggingface_hub import snapshot_download

HF_MODEL = os.environ.get("HF_MODEL", "QuantTrio/Qwen3.5-9B-AWQ")
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")


def _resolve_hf_token() -> str | None:
    raw = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or ""
    ).strip()
    return raw or None


HF_TOKEN = _resolve_hf_token()

if HF_TOKEN:
    from huggingface_hub import login

    login(token=HF_TOKEN, add_to_git_credential=False)
    print("Authenticated with Hugging Face Hub.")
else:
    print("No HF_TOKEN set; downloading as anonymous (may be slower).")

print(f"Downloading {HF_MODEL} to {MODEL_DIR}...")
snapshot_download(
    repo_id=HF_MODEL,
    local_dir=MODEL_DIR,
    token=HF_TOKEN,
)
print(f"Finished downloading {HF_MODEL}.")
