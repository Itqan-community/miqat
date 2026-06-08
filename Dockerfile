# ── Build & Run ────────────────────────────────────────────────────
#   docker build -t colabwis .
#
#   GPU: docker run --gpus all -p 8000:8000 \
#          -v model_data:/app/model_local \
#          -e HF_TOKEN=hf_your_token \
#          colabwis
#
#   CPU: docker run -p 8000:8000 \
#          -v model_data:/app/model_local \
#          colabwis
#
#   First-run model download:
#     docker run --rm -e HF_TOKEN=hf_xxx \
#       -v model_data:/app/model_local colabwis python model_downloader.py
# ───────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# gcc/g++ — compile ctc_segmentation (C extension)
# patchelf — strip executable-stack flag from ctranslate2 .so (glibc compat)
# ffmpeg/libsndfile — audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ patchelf \
        ffmpeg libsndfile1 libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Layer 1: copy requirements first so pip install is cached ─────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python3 -c "import glob,subprocess; [subprocess.run(['patchelf','--clear-execstack',s],check=True) for s in glob.glob('/usr/local/lib/python3.11/site-packages/ctranslate2*/**/*.so*',recursive=True)]; print('Done')"

# ── Layer 2: copy the rest of the project ─────────────────────────
COPY . .
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Required directories
RUN mkdir -p /app/model_local/whisper \
             /app/model_local/wav2vec2 \
             /app/temp_audio

ENV HF_TOKEN=""
EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
