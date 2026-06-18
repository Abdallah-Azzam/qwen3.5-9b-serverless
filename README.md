# Qwen3.5 9B Chat Endpoint

RunPod Serverless worker for **Qwen3.5-9B** chat completions via **vLLM** (`Qwen/Qwen3.5-9B` BF16 by default, non-thinking instruct mode).

## Deploy

1. Push this repo to GitHub or build the Docker image locally.
2. Create a RunPod Serverless endpoint from the image (or publish via RunPod Hub).
3. Choose a **24 GB GPU** and set container disk to **≥ 40 GB**.

### Manual Docker build

```bash
docker build \
  --build-arg HF_TOKEN=hf_xxx \
  -t qwen35-9b-endpoint .
```

Optional `HF_TOKEN` build arg speeds up Hugging Face downloads. Leave it unset for public models (anonymous download).

## VRAM requirements

VRAM = model weights + KV cache (grows with context) + ~1–2 GB vLLM overhead.

### Weights only

| Precision | HF model | Weight VRAM |
|-----------|----------|-------------|
| **BF16 / FP16 (default)** | `Qwen/Qwen3.5-9B` | ~18–19 GB |
| AWQ INT4 | `QuantTrio/Qwen3.5-9B-AWQ` | ~6–7 GB |

### KV cache add-on (batch size 1, approximate)

| Context (`MAX_MODEL_LEN`) | Extra VRAM |
|---------------------------|------------|
| 8K | ~1 GB |
| 16K | ~2 GB |
| 32K | ~4 GB |

### Recommended GPU

| Profile | Est. total VRAM | GPU |
|---------|-----------------|-----|
| **Default (BF16 + 8K)** | ~20–22 GB | **24 GB** (ADA_24, RTX 4090, A10) |
| BF16 + 32K | ~24–26 GB | 48 GB (OOM risk on 24 GB) |
| Budget (AWQ + 32K) | ~11–12 GB | 16–24 GB |

Use a **24 GB GPU** with the default BF16 preset. BF16 + 32K context does not fit comfortably on 24 GB — use AWQ or a 48 GB GPU for long context.

### OOM troubleshooting

- Lower `MAX_MODEL_LEN` (e.g. `4096`)
- Lower `GPU_MEMORY_UTILIZATION` (e.g. `0.85`)
- Switch to AWQ (`QUANTIZATION=awq`, `HF_MODEL=QuantTrio/Qwen3.5-9B-AWQ`) for more headroom

## Features

- Pre-baked `Qwen/Qwen3.5-9B` weights at image build time (BF16, closest vLLM match to local Q8 quality)
- Non-thinking instruct mode (`ENABLE_THINKING=false`)
- Text-only inference (`language_model_only`) — skips vision encoder
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
    "model": "Qwen3.5-9B",
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
| `reasoning_effort` | — | Accepted but ignored |

### Endpoint environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `HF_MODEL` | `Qwen/Qwen3.5-9B` | HuggingFace model ID (baked at build) |
| `MODEL_DIR` | `/models` | Local path to baked weights |
| `DISPLAY_MODEL_NAME` | `Qwen3.5-9B` | Name returned in responses |
| `QUANTIZATION` | `none` | vLLM quant method (`awq` for AWQ variant) |
| `MAX_MODEL_LEN` | `8192` | Context window cap |
| `DEFAULT_MAX_TOKENS` | `4096` | Default completion length |
| `MAX_TOKENS_LIMIT` | `16384` | Hard cap per request |
| `GPU_MEMORY_UTILIZATION` | `0.90` | vLLM VRAM fraction |
| `ENABLE_THINKING` | `false` | Qwen3.5 chain-of-thought mode |

## Example

```bash
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"messages":[{"role":"user","content":"Say hello in one sentence."}],"temperature":0.1,"max_tokens":64}}'
```

For long-running jobs, use async `/run` + `/status/{job_id}` polling.
