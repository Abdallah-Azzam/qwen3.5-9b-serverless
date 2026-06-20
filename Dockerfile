FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN rm -f /etc/apt/sources.list.d/*.list

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

ENV LLM_MODEL=sorc/qwen3.5-instruct:9b
ENV DISPLAY_MODEL_NAME=sorc/qwen3.5-instruct:9b
ENV OLLAMA_HOST=127.0.0.1:11434
ENV OLLAMA_MODELS=/root/.ollama
ENV OLLAMA_NUM_CTX=8192
ENV OLLAMA_KEEP_ALIVE=-1
ENV DEFAULT_MAX_TOKENS=4096
ENV MAX_TOKENS_LIMIT=16384
ENV DEFAULT_TEMPERATURE=0.7
ENV DEFAULT_TOP_P=0.8
ENV ENABLE_THINKING=false

WORKDIR /

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends \
        sudo ca-certificates git wget curl bash libgl1 libx11-6 \
        software-properties-common build-essential zstd -y && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

RUN apt-get update -y && \
    apt-get install python3.11 python3.11-dev python3.11-venv python3-pip -y --no-install-recommends && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    rm -f /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r /requirements.txt --no-cache-dir

COPY builder/fetch_models.py /fetch_models.py
ARG LLM_MODEL=sorc/qwen3.5-instruct:9b
RUN LLM_MODEL="${LLM_MODEL}" python /fetch_models.py && rm /fetch_models.py

COPY src .
COPY handler.py .
COPY test_input.json .

CMD python -u /rp_handler.py
