# Qwen3.5 9B Chat Endpoint

RunPod Serverless worker for **sorc/qwen3.5-instruct:9b** chat completions via **Ollama** (Q8_0, same model as local MedScribe). Reasoning is **disabled by default**.

## Deploy

1. Push this repo to GitHub or build the Docker image locally.
2. Create a RunPod Serverless endpoint from the image (or publish via RunPod Hub).
3. Choose a **24 GB GPU** and set container disk to **≥ 35 GB**.

### Manual Docker build

```bash
docker build -t qwen35-9b-ollama .
```

The model is pulled from Ollama at build time (~11 GB). No Hugging Face token required.

## VRAM requirements

VRAM = model weights + KV cache (grows with context) + Ollama overhead.

### Weights only

| Model | Quantization | Weight VRAM |
|-------|--------------|-------------|
| `sorc/qwen3.5-instruct:9b` | Q8_0 | ~11 GB |

### KV cache add-on (batch size 1, approximate)

| Context (`OLLAMA_NUM_CTX`) | Extra VRAM |
|----------------------------|------------|
| 8K | ~1–2 GB |
| 32K | ~7–10 GB |

### Recommended GPU

| Profile | Est. total VRAM | GPU |
|---------|-----------------|-----|
| **Default (Q8 + 8K)** | ~12–14 GB | 16–24 GB (A4000, ADA_24, RTX 4090) |
| Long context (Q8 + 32K) | ~18–22 GB | 24 GB (ADA_24, RTX 4090) |

## Features

- Pre-baked `sorc/qwen3.5-instruct:9b` via Ollama at image build time
- Reasoning disabled by default (`ENABLE_THINKING=false`; Ollama auto-enables thinking for Qwen3.5 if omitted)
- Per-request override via `think` or `reasoning_effort`
- OpenAI-style `messages` input via RunPod job payload

## API

`POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync`

**Request:**

```json
{
  "input": {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "temperature": 0.7,
    "max_tokens": 1024
  }
}
```

**Response:**

```json
{
  "status": "COMPLETED",
  "output": {
    "content": "assistant reply text",
    "model": "sorc/qwen3.5-instruct:9b",
    "usage": {
      "prompt_tokens": 120,
      "completion_tokens": 340
    }
  }
}
```

### Input parameters

| Field | Default | Description |
|-------|---------|-------------|
| `messages` | required | `[{role, content}]` — roles: `system`, `user`, `assistant` |
| `temperature` | `0.7` | Sampling temperature |
| `max_tokens` | `4096` | Max completion tokens (capped by `MAX_TOKENS_LIMIT`) |
| `top_p` | `0.8` | Nucleus sampling |
| `top_k` | — | Top-k sampling (optional) |
| `think` | env default | Explicit boolean to enable/disable Qwen3.5 thinking |
| `reasoning_effort` | env default | `"none"` disables thinking; `"low"`, `"medium"`, `"high"` enable it |

When thinking is enabled, only the final answer is returned in `content` (not internal reasoning).

### Endpoint environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_MODEL` | `sorc/qwen3.5-instruct:9b` | Ollama model tag (baked at build) |
| `DISPLAY_MODEL_NAME` | `sorc/qwen3.5-instruct:9b` | Name returned in responses |
| `OLLAMA_HOST` | `127.0.0.1:11434` | In-container Ollama daemon address |
| `OLLAMA_MODELS` | `/root/.ollama` | Model storage path |
| `OLLAMA_NUM_CTX` | `8192` | Context window cap |
| `OLLAMA_KEEP_ALIVE` | `-1` | Keep model loaded in VRAM |
| `DEFAULT_MAX_TOKENS` | `4096` | Default completion length |
| `MAX_TOKENS_LIMIT` | `16384` | Hard cap per request |
| `ENABLE_THINKING` | `false` | Default Qwen3.5 chain-of-thought mode |

## Example

```bash
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"messages":[{"role":"user","content":"Say hello in one sentence."}],"temperature":0.1,"max_tokens":64}}'
```

Disable reasoning explicitly per request:

```bash
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"messages":[{"role":"user","content":"Say hello."}],"think":false,"max_tokens":32}}'
```

For long-running jobs, use async `/run` + `/status/{job_id}` polling.
