import os
COLAB_ROOT = "/content"
MODEL_CACHE_DIR = os.path.join(COLAB_ROOT, "model_cache")
RESULTS_DIR = os.path.join(COLAB_ROOT, "results")
EVAL_CACHE_PATH = os.path.join(COLAB_ROOT, "eval_cache")
SAVE_PATH = RESULTS_DIR
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EVAL_CACHE_PATH, exist_ok=True)