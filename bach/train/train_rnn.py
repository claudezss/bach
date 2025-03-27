from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from bach import ROOT_DIR, get_device
from bach.dataset import RNNDataset
from bach.model import MusicRNN
from bach.train import ARTIFACT_PATH

app = typer.Typer()


@app.command()
def train(
    data_path: Path = ROOT_DIR.parent / "data_cache" / "dataset.pkl", epochs: int = 200, max_data_size: int = 10_000
) -> None:

    artifact_path = ARTIFACT_PATH / "rnn"

    artifact_path.mkdir(exist_ok=True, parents=True)

    dataset = RNNDataset(data_path, max_data_size=max_data_size)

    train_size = int(len(dataset) * 0.85)

    val_size = int(len(dataset) - train_size)

    train_set, val_set = random_split(dataset, [train_size, val_size])

    loader = DataLoader(train_set, batch_size=328, shuffle=True)

    val_loader = DataLoader(val_set, batch_size=328, shuffle=False)

    device = get_device()

    print(device)

    model = MusicRNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    criterion = torch.nn.MSELoss()

    epoch_loop = tqdm(range(epochs), leave=True)

    best_loss = np.inf

    train_losses = []

    val_losses = []

    for epoch in epoch_loop:

        is_best = False

        model.train()

        total_loss = 0.0

        total_test_loss = 0.0

        for batch in loader:

            batch = batch.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                out = model(batch)
                loss = criterion(out, batch)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        train_losses.append(avg_loss)

        scheduler.step()

        model.eval()

        for batch in val_loader:
            batch = batch.to(device)

            with torch.no_grad():
                out = model(batch)
                loss = criterion(out, batch)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(val_loader)

        val_losses.append(avg_test_loss)

        if avg_test_loss < best_loss:
            is_best = True
            best_loss = avg_test_loss

        if is_best:
            torch.save(model.state_dict(), artifact_path / "rnn_model.pt")

        epoch_loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
        epoch_loop.set_postfix(train_loss=avg_loss, test_loss=avg_test_loss, is_best=is_best)

    torch.save({"train_losses": train_losses, "val_losses": val_losses}, artifact_path / "losses.pt")

    torch.save({"mean": dataset.mean, "std": dataset.std}, artifact_path / "scaler.pt")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss", marker="o")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(artifact_path / "loss_plot.png")
    plt.show()


if __name__ == "__main__":
    app()
