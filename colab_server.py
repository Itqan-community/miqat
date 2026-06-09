import os
import uuid
import torch
import gc
import shutil
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import soundfile as sf
from alignment_engine import AlignmentEngine

app = FastAPI(title="Miqat Colab Alignment Server")

# Allow browser direct uploads (bypass local proxy for large audio files)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

WHISPER_PATH = "model_local/whisper"
WAV2VEC2_PATH = "model_local/wav2vec2"

# Device check at module load time
import torch
print(f"[Server] torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"[Server] torch.version.cuda: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"[Server] GPU: {torch.cuda.get_device_name(0)}, CUDA devices: {torch.cuda.device_count()}")

if not os.path.exists(WHISPER_PATH) or not os.path.exists(WAV2VEC2_PATH):
    print("Warning: Models not found locally. Please run model_downloader.py first.")

alignment_engine = AlignmentEngine(WHISPER_PATH, WAV2VEC2_PATH)
print(f"[Server] AlignmentEngine device: {alignment_engine.device}")
os.makedirs("temp_audio", exist_ok=True)

# ─── In-memory job store ───────────────────────────────────────────────────────
# Structure: { job_id: { "status": queued|processing|done|error,
#                        "alignments": [...] | None,
#                        "error": str | None } }
jobs: dict = {}

# Single-thread executor so GPU tasks never run in parallel
_executor = ThreadPoolExecutor(max_workers=1)


# ─── Background processing ────────────────────────────────────────────────────
def _run_job(job_id: str, file_path: str, reference_text: str, method: str):
    """Runs in a background thread. Updates jobs[job_id] when done."""
    try:
        jobs[job_id]["status"] = "processing"

        # Decide strategy based on method and audio duration
        speech, sr = sf.read(file_path, dtype='float32')
        if len(speech.shape) > 1:
            speech = speech.mean(axis=1)
        duration = len(speech) / sr
        del speech  # free RAM immediately

        print(f"[Job {job_id[:8]}] method={method}, duration={duration:.1f}s")

        if method == "hybrid":
            alignments = alignment_engine.align_hybrid(file_path, reference_text)
        elif method == "whisperx":
            alignments = alignment_engine.align_whisperx(file_path, reference_text)
        elif method == "whisper":
            alignments = alignment_engine.align_whisper(file_path, reference_text)
        elif duration > 35:
            # Long audio: robust ctc-segmentation
            alignments = alignment_engine.align_smart(file_path, reference_text)
        else:
            # Short audio: direct ctc
            alignments = alignment_engine.align(file_path, reference_text)

        jobs[job_id] = {
            "status": "done",
            "method": method,
            "alignments": alignments,
            "error": None,
        }
        print(f"[Job {job_id[:8]}] ✓ done — {len(alignments)} words")

    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs[job_id] = {
            "status": "error",
            "alignments": None,
            "error": str(e),
        }
        print(f"[Job {job_id[:8]}] ✗ error: {e}")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/align/cloud")
async def align_cloud(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    reference_text: str = Form(...),
    method: str = Form(...),
):
    """
    Accepts an audio file and returns a job_id immediately.
    The client must poll GET /align/status/{job_id} for the result.
    """
    if not reference_text.strip():
        return JSONResponse(
            {"status": "error", "message": "No reference text provided"},
            status_code=400,
        )

    job_id = str(uuid.uuid4())
    # Use job_id in filename to avoid collisions
    file_path = f"temp_audio/{job_id}_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    jobs[job_id] = {"status": "queued", "alignments": None, "error": None}

    # Submit to background thread (non-blocking — no timeout risk)
    background_tasks.add_task(
        lambda: _executor.submit(_run_job, job_id, file_path, reference_text, method)
    )

    return JSONResponse({
        "status": "queued",
        "job_id": job_id,
        "message": "Job started. Poll /align/status/{job_id} for results.",
    })


@app.get("/align/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Returns the current status of an alignment job.
      - queued / processing → still running
      - done → 'alignments' list is present
      - error → 'message' explains the failure
    """
    job = jobs.get(job_id)
    if not job:
        return JSONResponse(
            {"status": "error", "message": "Job not found"},
            status_code=404,
        )

    if job["status"] == "done":
        return JSONResponse({
            "status": "success",
            "alignments": job["alignments"],
        })
    elif job["status"] == "error":
        return JSONResponse(
            {"status": "error", "message": job["error"]},
            status_code=500,
        )
    else:
        return JSONResponse({
            "status": job["status"],
            "message": "Still processing — try again in a few seconds.",
        })


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)