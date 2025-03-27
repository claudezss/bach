import torch
import pandas as pd
from pathlib import Path


rnn_path = Path("data_cache/artifacts/rnn/losses.pt")
transformer_path = Path("data_cache/artifacts/transformer/epoch_20.pt")  # Adjust if different


rnn_data = torch.load(rnn_path)
transformer_data = torch.load(transformer_path)


rnn_df = pd.DataFrame({
    "Model": "RNN",
    "Epoch": list(range(1, len(rnn_data["train_losses"]) + 1)),
    "Train Loss": rnn_data["train_losses"],
    "Val Loss": rnn_data["val_losses"]
})

transformer_df = pd.DataFrame({
    "Model": "Transformer",
    "Epoch": list(range(1, len(transformer_data["train_losses"]) + 1)),
    "Train Loss": transformer_data["train_losses"],
    "Val Loss": transformer_data["val_losses"]
})

# Combine into one table
metric_table = pd.concat([rnn_df, transformer_df], ignore_index=True)

# Save as CSV
metric_table.to_csv("metric_table.csv", index=False)

print(metric_table.head(10))
