import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# RNN Model
class MusicRNN(nn.Module):

    def __init__(self, input_size=4, hidden_size=256, num_layers=4, output_size=4):
        super(MusicRNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out


# Transformer Model

# Constants
MAX_SEQ_LENGTH = 512
EMBEDDING_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.1


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MusicTransformer(
    nn.Module,
):
    def __init__(self, vocab_size=390, d_model=EMBEDDING_DIM, nhead=NUM_HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super(MusicTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, vocab_size)

        self.d_model = d_model
        self.vocab_size = vocab_size

    def forward(self, src, src_mask=None):
        # Create embedding and apply positional encoding
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)

        # Create causal mask if not provided
        if src_mask is None:
            src_mask = self._generate_square_subsequent_mask(src.size(1)).to(src.device)

        # Apply transformer encoder
        output = self.transformer_encoder(src, src_mask)
        output = self.output_layer(output)

        return output

    def _generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def generate(self, primer_tokens, max_length=1000, temperature=1.0):
        """Generate new music from a primer sequence"""
        self.eval()
        with torch.no_grad():
            generated = primer_tokens.copy()

            # Convert to tensor
            current_input = torch.tensor([generated[-min(len(generated), MAX_SEQ_LENGTH - 1) :]], dtype=torch.long).to(
                next(self.parameters()).device
            )

            for _ in range(max_length):
                # Get logits from the model
                logits = self(current_input)

                # Get the next token prediction (last position)
                next_token_logits = logits[0, -1, :] / temperature

                # Apply softmax to get probabilities
                probs = F.softmax(next_token_logits, dim=-1)

                # Sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1).item()

                # Add the token to our generated sequence
                generated.append(next_token)

                # Stop if we generate an EOS token
                if next_token == self.processor.EOS_TOKEN:
                    break

                # Update the input sequence
                current_input = torch.cat(
                    (current_input, torch.tensor([[next_token]], dtype=torch.long).to(current_input.device)), dim=1
                )

                # Keep sequence length manageable for next iteration
                if current_input.size(1) >= MAX_SEQ_LENGTH:
                    current_input = current_input[:, -MAX_SEQ_LENGTH + 1 :]

            return generated
