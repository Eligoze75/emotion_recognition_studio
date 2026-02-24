"""Offline evaluation and sanity checks."""

# Do not import run_offline_eval here: it pulls in torch/cv2 and causes
# "found in sys.modules before execution" when running python -m src.evaluation.offline_eval
__all__ = ["run_offline_eval"]
