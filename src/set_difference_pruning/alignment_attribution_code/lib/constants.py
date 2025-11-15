import os
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
MODEL_CACHE_DIR = os.environ.get(
    "MODEL_CACHE_DIR",
    os.path.join(_REPO_ROOT, "model_cache"),
)
RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    os.path.join(_REPO_ROOT, "results"),
)
EVAL_CACHE_PATH = os.environ.get(
    "EVAL_CACHE_PATH",
    os.path.join(_REPO_ROOT, "eval_cache"),
)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EVAL_CACHE_PATH, exist_ok=True)