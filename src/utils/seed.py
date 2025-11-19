"""
Utility to fix random seeds.

In simple words: if we set the seed, then every time we run the code,
we will get (almost) the same results. This is very important for research.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seeds for Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # These two lines make cuDNN deterministic. Training may be a bit slower,
    # but results become more reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False