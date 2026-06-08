#!/usr/bin/env bash
set -euo pipefail

# Create required directories
mkdir -p /app/model_local/whisper
mkdir -p /app/model_local/wav2vec2
mkdir -p /app/temp_audio

# If HF_TOKEN is set and models don't exist, download them
if [ -n "${HF_TOKEN:-}" ]; then
    WHISPER_EXISTS=$(ls /app/model_local/whisper/pytorch_model.bin /app/model_local/whisper/model.safetensors 2>/dev/null || true)
    WAV2VEC2_EXISTS=$(ls /app/model_local/wav2vec2/pytorch_model.bin /app/model_local/wav2vec2/model.safetensors 2>/dev/null || true)

    if [ -z "$WHISPER_EXISTS" ] || [ -z "$WAV2VEC2_EXISTS" ]; then
        echo "Downloading models from Hugging Face..."
        cd /app && python model_downloader.py
    else
        echo "Models already present, skipping download."
    fi
fi

echo "Starting server on 0.0.0.0:8000..."
exec uvicorn colab_server:app --host 0.0.0.0 --port 8000 --log-level info
