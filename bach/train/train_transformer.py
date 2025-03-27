import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import typer
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from bach import get_device
from bach.data import MIDIProcessor
from bach.dataset import MIDIDataset
from bach.model import MusicTransformer
from bach.train import ARTIFACT_PATH

app = typer.Typer()


@app.command()
def train(epochs: int = 20, max_data_size: int = 10_000):

    artifacts_path = ARTIFACT_PATH / "transformer"

    artifacts_path.mkdir(exist_ok=True, parents=True)

    dataset = MIDIDataset(max_data_size=max_data_size)

    processor = MIDIProcessor()

    train_size = int(len(dataset) * 0.85)

    val_size = int(len(dataset) - train_size)

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    model = MusicTransformer(vocab_size=processor.vocab_size)

    device = get_device()

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print(device)

    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=model.processor.PAD_TOKEN if hasattr(model, "processor") else -100)

    train_losses = []

    val_losses = []

    for epoch in range(epochs):

        total_loss = 0

        total_val_loss = 0

        model.train()

        loop = tqdm(train_loader, leave=True)

        for batch_idx, (x, y) in enumerate(loop):

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # Create mask for causal attention
            src_mask = model._generate_square_subsequent_mask(x.size(1)).to(x.device)

            output = model(x, src_mask)

            # Reshape for loss computation
            output = output.reshape(-1, model.vocab_size)
            y = y.reshape(-1)

            loss = criterion(output, y)
            loss.backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                loop.set_postfix(
                    Epoch=f"{epoch+1}/{epochs}", Batch=f"{batch_idx}/{len(train_loader)}", Loss=f"{loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)

        train_losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{epochs} complete, Avg Loss: {avg_loss:.4f}")

        model.eval()

        loop = tqdm(val_loader, leave=True)

        for batch_idx, (x, y) in enumerate(loop):

            x, y = x.to(device), y.to(device)

            with torch.no_grad():

                # Create mask for causal attention
                src_mask = model._generate_square_subsequent_mask(x.size(1)).to(x.device)

                output = model(x, src_mask)

                # Reshape for loss computation
                output = output.reshape(-1, model.vocab_size)
                y = y.reshape(-1)

                loss = criterion(output, y)

                total_val_loss += loss.item()

                if batch_idx % 10 == 0:
                    loop.set_postfix(
                        Epoch=f"{epoch+1}/{epochs}",
                        Batch=f"{batch_idx}/{len(val_loader)}",
                        ValLoss=f"{loss.item():.4f}",
                    )

        avg_val_loss = total_val_loss / len(val_loader)

        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs} complete, Avg Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
            },
            artifacts_path / f"epoch_{epoch+1}.pt",
        )

    # Plot the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss", marker="o")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(artifacts_path / "loss_plot.png")
    plt.show()


if __name__ == "__main__":
    app()
