import os
import shutil
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

def clean_lock_files(directory):
    """Deeply removes .lock and .incomplete files from a directory."""
    if not os.path.exists(directory):
        return
    
    print(f"Cleaning existing lock files in {directory}...")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".lock") or file.endswith(".incomplete"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed: {file}")
                except Exception as e:
                    print(f"Could not remove {file}: {e}")

def verify_model(local_dir):
    """Simple check for essential model file."""
    essential_files = ["pytorch_model.bin", "model.safetensors"]
    found = any(os.path.exists(os.path.join(local_dir, f)) for f in essential_files)
    return found

def download_models():
    models = [
        {"id": "tarteel-ai/whisper-base-ar-quran", "dir": "model_local/whisper"},
        {"id": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic", "dir": "model_local/wav2vec2"},
        {"id": "Systran/faster-whisper-large-v2", "dir": "model_local/whisperx"},
    ]

    for model in models:
        model_id = model["id"]
        local_dir = model["dir"]

        print(f"\n--- Processing model {model_id} ---")

        # Clean up interrupted downloads
        clean_lock_files(local_dir)

        if not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)

        print(f"Downloading model {model_id} to {local_dir}...")

        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                token=hf_token
            )

            # Check for CTranslate2 files (whisperx) or PyTorch files (custom align)
            has_ct2 = os.path.exists(os.path.join(local_dir, "model.safetensors")) or \
                      os.path.exists(os.path.join(local_dir, "pytorch_model.bin")) or \
                      os.path.exists(os.path.join(local_dir, "model.bin"))
            if has_ct2:
                print(f"SUCCESS: Model {model_id} verified.")
            else:
                print(f"WARNING: Model {model_id} downloaded but weights not found in {local_dir}")

        except Exception as e:
            print(f"ERROR downloading model {model_id}: {e}")

if __name__ == "__main__":
    download_models()
