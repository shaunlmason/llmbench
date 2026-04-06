import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError


# Matches URLs like:
#   https://huggingface.co/{owner}/{repo}/resolve/{branch}/{filename}
#   https://huggingface.co/{owner}/{repo}/blob/{branch}/{filename}
_HF_URL_PATH_RE = re.compile(
    r"^/(?P<owner>[^/]+)/(?P<repo>[^/]+)/(?:resolve|blob)/[^/]+/(?P<filename>.+)$"
)


def parse_model_ref(model_ref: str) -> tuple[str, str]:
    """Parse a model reference into (repo_id, filename).

    Accepts:
    - 'owner/repo:filename.gguf'
    - 'https://huggingface.co/owner/repo/resolve/main/filename.gguf[?download=true]'
    - 'https://huggingface.co/owner/repo/blob/main/filename.gguf'
    """
    # HuggingFace URL form
    if model_ref.startswith(("http://", "https://")):
        parsed = urlparse(model_ref)
        if parsed.netloc not in ("huggingface.co", "www.huggingface.co"):
            print(f"Error: Only huggingface.co URLs are supported, got '{parsed.netloc}'")
            sys.exit(1)
        match = _HF_URL_PATH_RE.match(parsed.path)
        if not match:
            print(f"Error: Could not parse HuggingFace URL '{model_ref}'")
            print("Expected format: https://huggingface.co/owner/repo/resolve/main/filename.gguf")
            sys.exit(1)
        return f"{match['owner']}/{match['repo']}", match["filename"]

    # owner/repo:filename form
    if ":" not in model_ref:
        print(f"Error: Invalid model reference '{model_ref}'")
        print("Expected one of:")
        print("  owner/repo:filename.gguf")
        print("  https://huggingface.co/owner/repo/resolve/main/filename.gguf")
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
