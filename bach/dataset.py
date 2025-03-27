import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from bach import ROOT_DIR


class RNNDataset(Dataset):
    def __init__(
        self,
        data_path: Path = ROOT_DIR.parent / "data_cache" / "dataset.npy",
        sequence_length=150,
        max_data_size: int | None = None,
    ):

        self.sequence_length = sequence_length

        if data_path.suffix == ".npy":
            self.data = np.load(data_path)
        elif data_path.suffix == ".pkl":
            with open(data_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            raise ValueError("Unsupported file type. Please provide a .npy or .pkl file.")

        self.data = torch.tensor(self.data, dtype=torch.float32)

        if max_data_size is not None:
            self.data = self.data[:max_data_size, :, :]

        self.mean = torch.mean(self.data)
        self.std = torch.std(self.data)

        # Standardize the tensor
        self.data = (self.data - self.mean) / self.std

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        notes_tensor = torch.tensor(self.data[idx], dtype=torch.float32)
        return notes_tensor  # Shape: (sequence_length, 3)


# Dataset
class MIDIDataset(Dataset):
    def __init__(
        self,
        tokens_file_path=ROOT_DIR.parent / "data_cache" / "transformer_dataset.pkl",
        seq_length=512,
        max_data_size: int | None = None,
    ):

        with open(tokens_file_path, "rb") as f:
            self.tokens = pickle.load(f)

        self.seq_length = seq_length
        self.token_sequences = []

        for token in tqdm(self.tokens):

            # Create sequences with overlap
            if len(token) > 0:
                for i in range(0, len(token) - seq_length, seq_length // 2):
                    seq = token[i : i + seq_length]
                    if len(seq) == seq_length:
                        self.token_sequences.append(seq)

                # Add last sequence if it's not too short
                if len(token) % seq_length > seq_length // 2:
                    last_seq = token[-seq_length:]
                    if len(last_seq) == seq_length:
                        self.token_sequences.append(last_seq)

        if max_data_size is not None:
            self.token_sequences = self.token_sequences[:max_data_size]

        print(f"Created {len(self.token_sequences)} sequences")

    def __len__(self):
        return len(self.token_sequences)

    def __getitem__(self, idx):
        tokens = self.token_sequences[idx]

        # Input and target sequences (shifted by 1 for next token prediction)
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)

        return x, y
