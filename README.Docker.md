# Docker Setup

## Build

```bash
docker build -t colabwis .
```

## Run

### GPU (recommended — models are large and benefit from CUDA)

```bash
docker run --gpus all -p 8000:8000 \
  -v model_data:/app/model_local \
  -e HF_TOKEN=hf_your_token \
  colabwis
```

### CPU

```bash
docker run -p 8000:8000 \
  -v model_data:/app/model_local \
  colabwis
```

### With docker-compose

```bash
# GPU (default)
docker compose up --build

# CPU only
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
```

## First-run model download

Models are downloaded automatically on first start if `HF_TOKEN` is set.
Alternatively, run the helper explicitly:

```bash
docker compose run --rm downloader
```

Or manually:

```bash
docker run --rm -e HF_TOKEN=hf_your_token \
  -v model_data:/app/model_local \
  colabwis python model_downloader.py
```

## API

| Method | Path                  | Description                          |
|--------|-----------------------|--------------------------------------|
| POST   | `/align/cloud`        | Upload audio + reference text        |
| GET    | `/align/status/<id>`  | Poll job status                      |
| GET    | `/health`             | Health check                         |

## Volumes

- `model_data` — persists Whisper & Wav2Vec2 models across rebuilds (~2–3 GB total).

## Notes

- The server uses an **in-memory** job store. Results are lost on restart.
- `temp_audio/` is cleaned up automatically after each job.
- PyTorch auto-detects CUDA; `--gpus all` is the only requirement for GPU acceleration.
- `ctranslate2` (a `whisperx` dependency) is patched at build time with `patchelf` to strip the executable-stack flag, avoiding glibc incompatibility.
