import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError


def parse_model_ref(model_ref: str) -> tuple[str, str]:
    """Parse 'repo_id:filename' into (repo_id, filename)."""
    if ":" not in model_ref:
        print(f"Error: Invalid model reference '{model_ref}'")
        print("Expected format: 'owner/repo:filename.gguf'")
        print("Example: 'bartowski/Qwen2.5-7B-Instruct-GGUF:Qwen2.5-7B-Instruct-Q4_K_M.gguf'")
        sys.exit(1)
    repo_id, filename = model_ref.split(":", 1)
    if not repo_id or not filename:
        print(f"Error: Invalid model reference '{model_ref}' — both repo and filename are required")
        sys.exit(1)
    return repo_id, filename


def ensure_model(repo_id: str, filename: str, models_dir: Path) -> Path:
    """Download model from HuggingFace if not already present. Returns path to GGUF file."""
    local_path = models_dir / repo_id / filename
    if local_path.exists():
        print(f"Model already downloaded: {local_path}")
        return local_path

    print(f"Downloading {repo_id}/{filename} to {local_path.parent}...")
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_path.parent,
        )
        print(f"Downloaded: {downloaded}")
        return Path(downloaded)
    except HfHubHTTPError as e:
        print(f"Error downloading model: {e}")
        print("Check that the repo_id and filename are correct.")
        sys.exit(1)
