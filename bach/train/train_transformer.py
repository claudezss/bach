import torch
import torch.nn as nn
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm

from bach import get_device
from bach.data import MIDIProcessor
from bach.dataset import MIDIDataset, MusicDataset
from bach.model import MusicTransformer
from bach.train import ARTIFACT_PATH

app = typer.Typer()

# Constants
MAX_SEQ_LEN = 512  # Maximum sequence length
VOCAB_SIZE = 512  # Size of vocabulary (pitch + duration + instrument + special tokens)
D_MODEL = 256  # Embedding dimension
N_HEADS = 8  # Number of attention heads
N_LAYERS = 6  # Number of transformer layers
D_FF = 1024  # Feedforward dimension
DROPOUT = 0.1  # Dropout rate

# Special tokens
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2


def train_model(model, train_dataloader, val_dataloader=None, epochs=10, lr=0.0001, device="cuda"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (src, tgt) in enumerate(train_dataloader):
            src, tgt = src.to(device), tgt.to(device)

            # Forward pass
            src = src.transpose(0, 1)  # Change to (seq_len, batch_size)
            tgt_input = tgt.transpose(0, 1)[:-1, :]  # Exclude the last token
            tgt_output = tgt.transpose(0, 1)[1:, :]  # Exclude the first token

            # Create masks
            src_padding_mask = (src == PAD_TOKEN).transpose(0, 1)
            tgt_padding_mask = (tgt_input == PAD_TOKEN).transpose(0, 1)

            output = model(src, tgt_input)

            # Reshape for loss calculation
            output = output.view(-1, output.shape[-1])
            tgt_output = tgt_output.contiguous().view(-1)

            # Calculate loss
            loss = criterion(output, tgt_output)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Average loss: {avg_loss:.4f}")

        # Validation
        if val_dataloader is not None:
            val_loss = evaluate(model, val_dataloader, criterion, device)
            print(f"Validation loss: {val_loss:.4f}")

    return model


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            # Forward pass
            src = src.transpose(0, 1)
            tgt_input = tgt.transpose(0, 1)[:-1, :]
            tgt_output = tgt.transpose(0, 1)[1:, :]

            output = model(src, tgt_input)

            # Reshape for loss calculation
            output = output.view(-1, output.shape[-1])
            tgt_output = tgt_output.contiguous().view(-1)

            # Calculate loss
            loss = criterion(output, tgt_output)
            total_loss += loss.item()

    return total_loss / len(dataloader)


@app.command()
def train(epochs: int = 10):

    artifacts_path = ARTIFACT_PATH / "transformer"

    artifacts_path.mkdir(exist_ok=True, parents=True)

    dataset = MusicDataset()

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128)

    model = MusicTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
    )

    device = get_device()

    print(device)

    model.to(device)

    model.train()

    # Train model
    model = train_model(
        model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=10, device=device
    )

    # Save model
    torch.save(model.state_dict(), artifacts_path / "music_transformer.pth")


if __name__ == "__main__":
    app()
