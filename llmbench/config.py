from pathlib import Path

DEFAULT_PORT = 8080
DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_LIMIT = 200
DEFAULT_TASKS = "hellaswag,humaneval"
DEFAULT_MODELS_DIR = Path.home() / "models"
HISTORY_DIR = Path.home() / ".llmbench"
HISTORY_FILE = HISTORY_DIR / "history.json"
HEALTH_CHECK_TIMEOUT = 120
HEALTH_CHECK_INTERVAL = 2

SERVICE_NAMES = {
    "gpu0": "llama-gpu0",
    "gpu1": "llama-gpu1",
    "both": ["llama-gpu0", "llama-gpu1"],
}
