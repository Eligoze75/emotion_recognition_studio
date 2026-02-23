"""Seed setting for reproducible runs."""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int]) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    When seed is None, no seeding is performed (non-deterministic behavior).

    Args:
        seed: Integer seed value, or None to leave RNGs unchanged.
    """
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

