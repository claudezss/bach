import logging
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).parent

logging.basicConfig(level=logging.WARNING)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
