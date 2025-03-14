from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bach import ROOT_DIR
from bach.dataset import RNNDataset
from bach.model import MusicRNN
from bach.train import ARTIFACT_PATH


def train_rnn(data_path: Path = ROOT_DIR.parent / "data_cache" / "dataset.npy", epochs: int = 200) -> None:
    artifact_path = ARTIFACT_PATH / "rnn"

    artifact_path.mkdir(exist_ok=True, parents=True)

    dataset = RNNDataset(data_path)

    loader = DataLoader(dataset[:80000], batch_size=328, shuffle=True)

    test_loader = DataLoader(dataset[80000:100000], batch_size=328, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    model = MusicRNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    criterion = torch.nn.MSELoss()

    epoch_loop = tqdm(range(epochs), leave=True)

    best_loss = np.inf

    for epoch in epoch_loop:

        is_best = False

        model.train()

        total_loss = 0.0

        total_test_loss = 0.0

        for batch in loader:
            batch = batch.to(device)

            with torch.amp.autocast("cuda"):
                out = model(batch)
                loss = criterion(out, batch)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        scheduler.step()

        model.eval()

        for batch in test_loader:
            batch = batch.to(device)

            with torch.no_grad():
                out = model(batch)
                loss = criterion(out, batch)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)

        if avg_test_loss < best_loss:
            is_best = True
            best_loss = avg_test_loss

        if is_best:
            torch.save(model.state_dict(), artifact_path / "rnn_model.pt")

        epoch_loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
        epoch_loop.set_postfix(train_loss=avg_loss, test_loss=avg_test_loss, is_best=is_best)


train_rnn()
