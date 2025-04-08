from pathlib import Path

import numpy as np
import pretty_midi
import torch
import typer
from huggingface_hub import hf_hub_download

from bach import get_device
from bach.data import MIDIProcessor
from bach.model import MusicTransformer

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


def transformer_inference(
    input_midi_path: str,
    temperatures: list[float] = (0.1, 0.5, 1),
    input_seq_length: int = 100,
    max_output_seq_length: int = 300,
):
    cache_dir = Path(__file__).parent.parent / "data_cache" / "model"

    cache_dir.mkdir(exist_ok=True, parents=True)

    typer.secho(
        f"Downloading model from Hugging Face Hub (https://huggingface.co/claudezss/bach) " f"to {cache_dir}",
        fg=typer.colors.GREEN,
    )
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

    typer.secho(f"Running inference on {device}", fg=typer.colors.GREEN)
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

    input_midi = processor.tokens_to_midi(input, return_data=True)

    output_midi_data = {"input": input_midi}

    for temperature in temperatures:

        generated_tokens = model.generate(input, max_length=max_output_seq_length, temperature=temperature)

        generated_midi = processor.tokens_to_midi(generated_tokens, return_data=True)
        output_midi_data.update({f"temperature-{temperature}": generated_midi})
    return output_midi_data


@app.command()
def generate_music_by_transformer(
    input_midi_path: str,
    output_midi_path: str,
    temperatures: list[float] = (0.1, 0.5, 1),
    input_seq_length: int = 100,
    max_output_seq_length: int = 300,
):
    generated_midi_data = transformer_inference(
        input_midi_path=input_midi_path,
        temperatures=temperatures,
        input_seq_length=input_seq_length,
        max_output_seq_length=max_output_seq_length,
    )
    output_midi_path = Path(output_midi_path)

    for label, midi_data in generated_midi_data.items():
        file_path = output_midi_path.parent / f"{label}.mid"
        midi_data.write(str(file_path.absolute()))
        typer.secho(f"Generated MIDI file was saved at: {file_path}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
