import torch
import torch.nn as nn
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm

from bach.data import MIDIProcessor
from bach.dataset import MIDIDataset
from bach.model import MusicTransformer
from bach.train import ARTIFACT_PATH

app = typer.Typer()


# Training utilities
@app.command()
def train(epochs=10):

    artifacts_path = ARTIFACT_PATH / "transformer"

    artifacts_path.mkdir(exist_ok=True, parents=True)

    dataset = MIDIDataset()

    processor = MIDIProcessor()

    train_loader = DataLoader(dataset, batch_size=328, shuffle=True)

    model = MusicTransformer(vocab_size=processor.vocab_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print(device)

    model.to(device)

    model.train()

    criterion = nn.CrossEntropyLoss(ignore_index=model.processor.PAD_TOKEN if hasattr(model, "processor") else -100)

    for epoch in range(epochs):

        total_loss = 0

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
        print(f"Epoch {epoch+1}/{epochs} complete, Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            },
            artifacts_path / f"epoch_{epoch+1}.pt",
        )


if __name__ == "__main__":
    app()
