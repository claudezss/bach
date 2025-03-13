from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from bach import ROOT_DIR


class RNNDataset(Dataset):
    def __init__(self, data_path: Path = ROOT_DIR.parent / "data_cache" / "dataset.npy", sequence_length=150):

        self.sequence_length = sequence_length
        self.data = np.load(data_path)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        notes_tensor = torch.tensor(self.data[idx], dtype=torch.float32)
        return notes_tensor  # Shape: (sequence_length, 3)
