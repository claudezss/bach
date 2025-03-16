import math
from pathlib import Path

import numpy as np
import pretty_midi
import torch
import torch.nn.functional as F
import typer
from huggingface_hub import hf_hub_download

from bach import get_device
from bach.data import (
    D_FF,
    D_MODEL,
    DROPOUT,
    EOS_TOKEN,
    MAX_SEQ_LEN,
    N_HEADS,
    N_LAYERS,
    SOS_TOKEN,
    VOCAB_SIZE,
    MIDIProcessor,
)
from bach.model import MusicRNN, MusicTransformer

app = typer.Typer()


@app.command()
def generate_random_midi(output_path="random_music.mid", num_notes=127, instrument_name="Acoustic Grand Piano"):
    """
    Generates a random MIDI file with a specified number of notes.

    Args:
        output_path (str): Path to save the generated MIDI file.
        num_notes (int): Number of notes to generate.
        instrument_name (str): Instrument name for playback.
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))

    start_time = 0.0

    for _ in range(num_notes):
        pitch = np.random.randint(50, 80)  # Random pitch (MIDI range 50-80)
        velocity = np.random.randint(60, 120)  # Random velocity (60-120)
        duration = np.random.uniform(0.2, 0.8)  # Random duration
        end_time = start_time + duration

        note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
        instrument.notes.append(note)

        start_time += np.random.uniform(0.1, 0.5)  # Increment start time

    midi.instruments.append(instrument)
    midi.write(output_path)
    print(f"Random MIDI file saved at: {output_path}")


def generate_music() -> pretty_midi.PrettyMIDI:
    from bach.data import midi_to_notes

    device = "cuda"
    model = MusicRNN().to(device)
    model.load_state_dict(torch.load("D:\Dev\\repo\\bach\\artifacts\\rnn_model.pt"))

    model.eval()

    input_midi = midi_to_notes("D:\Dev\\repo\\bach\data_cache\data\\albeniz-aragon_fantasia_op47_part_6.mid")

    input_midi_ts = torch.tensor(input_midi, dtype=torch.float32).to(device)
    input_midi_ts = input_midi_ts[0, :, :]

    generated_sequence = []

    with torch.inference_mode():
        for _ in range(3):
            output = model(input_midi_ts)
            next_note = output[:15, :]
            input_midi_ts = torch.cat([input_midi_ts[15:, :], next_note])
            generated_sequence.append(next_note.cpu().numpy())

    music = np.array(generated_sequence).reshape(-1, 4)

    midi = pretty_midi.PrettyMIDI()

    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano"))

    start_time = 0
    for note_data in music:
        _, duration, pitch, velocity = note_data

        pitch = int(np.clip(pitch, 0, 127))  # Ensure valid pitch range
        velocity = int(np.clip(velocity, 0, 127))  # Ensure valid velocity

        note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=start_time + duration)
        start_time = start_time + duration
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write("test.mid")


@app.command()
def generate_music_by_transformer(
    input_midi_path: str,
    output_midi_path: str,
    temperature: float = 0.8,
    input_seq_length: int = 100,
    max_output_seq_length: int = 200,
):
    cache_dir = Path(__file__).parent.parent / "data_cache" / "model"
    cache_dir.mkdir(exist_ok=True, parents=True)

    hf_hub_download(
        repo_id="claudezss/bach",
        filename="music_transformer.pt",
        repo_type="model",
        cache_dir=cache_dir,
        local_dir=cache_dir,
    )

    model = MusicTransformer()
    model.load_state_dict(torch.load(cache_dir / "music_transformer.pt", map_location="cpu")["model_state_dict"])
    processor = MIDIProcessor()
    model.processor = processor
    device = get_device()
    model.to(device)
    model.eval()

    token = processor.midi_to_tokens(input_midi_path)

    sequence = []

    seq_length = 512

    if len(token) > 0:

        for i in range(0, len(token) - seq_length, seq_length // 2):
            seq = token[i : i + seq_length]
            if len(seq) == seq_length:
                sequence.append(seq)

        # Add last sequence if it's not too short
        if len(token) % seq_length > seq_length // 2:
            last_seq = token[-seq_length:]
            if len(last_seq) == seq_length:
                sequence.append(last_seq)

    input = sequence[0][:input_seq_length]
    generated_tokens = model.generate(input, max_length=max_output_seq_length, temperature=temperature)

    output_midi_path = Path(output_midi_path)
    processor.tokens_to_midi(input, str((output_midi_path.parent / "input.mid").absolute()))
    processor.tokens_to_midi(generated_tokens, str(output_midi_path.absolute()))


def generate_music2(model, seed_midi, processor, max_length=1024, temperature=0.5, device="cuda"):
    model.eval()

    # Process seed
    seed_tokens = processor.midi_to_sequence(seed_midi)
    if len(seed_tokens) > max_length // 2:
        seed_tokens = seed_tokens[: max_length // 2]

    # Convert to tensor and ensure it fits within model's capacity
    seed_tensor = torch.tensor(seed_tokens).unsqueeze(1).to(device)  # (seq_len, 1)

    # Initialize target with SOS token
    tgt = torch.tensor([[SOS_TOKEN]]).to(device)  # (1, 1)

    # Generate sequence
    generated_tokens = [SOS_TOKEN]

    for _ in range(max_length):
        try:
            # Make sure seed_tensor and tgt are compatible for the model
            if seed_tensor.size(0) > MAX_SEQ_LEN:
                seed_tensor = seed_tensor[-MAX_SEQ_LEN:]

            # Create masks
            tgt_mask = model.generate_square_subsequent_mask(tgt.size(0)).to(device)

            # Forward pass - use memory instead of direct encoder-decoder pass
            memory = model.transformer_encoder(
                model.pos_encoder(model.embedding(seed_tensor) * math.sqrt(model.d_model))
            )
            tgt_embedded = model.pos_encoder(model.embedding(tgt) * math.sqrt(model.d_model))
            output = model.transformer_decoder(tgt_embedded, memory, tgt_mask=tgt_mask)
            output = model.fc_out(output)

            next_token_logits = output[-1, 0] / temperature
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1).item()

            # Add to sequence
            generated_tokens.append(next_token)
            next_token_tensor = torch.tensor([[next_token]]).to(device)
            tgt = torch.cat([tgt, next_token_tensor], dim=0)

            # Stop if EOS token is generated or if sequence is too long
            if next_token == EOS_TOKEN or len(generated_tokens) >= max_length:
                break

        except RuntimeError as e:
            print(f"Error during generation, stopping early: {e}")
            break

    # Convert tokens back to MIDI
    midi_data = processor.sequence_to_midi(generated_tokens)
    return midi_data


def inference_example(seed_midi_path, output_path):
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
    )
    model.load_state_dict(
        torch.load("D:\Dev\\repo\\bach\\data_cache\\artifacts\\transformer\\music_transformer.pth", map_location=device)
    )
    model.to(device)

    # Initialize processor
    processor = MIDIProcessor()

    # Load seed MIDI
    seed_midi = pretty_midi.PrettyMIDI(seed_midi_path)

    # Generate music
    generated_midi = generate_music2(
        model=model, seed_midi=seed_midi, processor=processor, max_length=1024, temperature=1.0, device=device
    )

    # Save generated music
    generated_midi.write(output_path)
    print(f"Generated music saved to {output_path}")


# if __name__ == "__main__":
#     app()

inference_example("D:\\Dev\\repo\\bach\\data_cache\\data\\joplin-a_picture_of_her_face.mid", "t.mid")
