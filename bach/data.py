import json
import logging
import math
import multiprocessing
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np
import pretty_midi
import typer
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from rich.progress import track

from bach import ROOT_DIR

app = typer.Typer()

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=Warning)


def get_workers(n_workers: int | None) -> int:
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    return n_workers


def download_single_file(data, repo_id: str, cache_dir: Path) -> Optional[str]:
    """Download a single MIDI file and return its path."""
    try:
        filename = data["file_name"]
        midi_file_path = hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type="dataset", cache_dir=cache_dir, local_dir=cache_dir
        )
        logger.debug(f"MIDI file downloaded to: {midi_file_path}")
        return midi_file_path
    except Exception as e:
        logger.debug(f"Failed to download {data.get('file_name', 'unknown')}: {str(e)}")
        return None


@app.command()
def load(cache_dir: str = ROOT_DIR.parent / "data_cache", n_workers: int = None) -> None:
    """
    Load and download MIDI files using parallel processing.

    Args:
        cache_dir: Directory to store downloaded files
        n_workers: Number of worker processes (defaults to CPU count if None)

    Returns:
        List of paths to the downloaded MIDI files
    """
    # Create cache directory if it doesn't exist
    if not isinstance(cache_dir, Path):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    repo_id = "drengskapur/midi-classical-music"

    # Load dataset (contains file names only)
    ds = load_dataset(repo_id, cache_dir=str(cache_dir.absolute()), split="train")

    # Use ProcessPoolExecutor instead of multiprocessing.Pool
    midi_file_paths = []

    n_workers = get_workers(n_workers)

    # Using ThreadPoolExecutor can also work and avoids multiprocessing issues
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Create a list of futures
        futures = [executor.submit(download_single_file, data, repo_id, cache_dir) for data in ds]

        # Process results as they complete
        for future in track(
            as_completed(futures), total=len(futures), description=f"Downloading MIDI files using {n_workers} workers"
        ):
            result = future.result()
            if result:
                midi_file_paths.append(result)

    logger.debug(f"Downloaded {len(midi_file_paths)} MIDI files successfully")

    print(f"Downloaded {len(midi_file_paths)} MIDI files successfully")

    file_path_cache = cache_dir / "midi_files.json"

    with open(file_path_cache, "+w") as f:
        json.dump(midi_file_paths, f)

    print(f"File paths were saved to {file_path_cache.absolute()}")


def midi_to_notes(midi_path, sequence_length=30, shift=15) -> np.ndarray:
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        return None

    notes = []
    for instrument in midi.instruments:
        if not instrument.is_drum:  # Exclude drum tracks
            for note in instrument.notes:
                notes.append([note.start, note.duration, note.pitch, note.velocity])

    notes = sorted(notes, key=lambda x: x[0])  # Sort by start time
    notes = np.array(notes)  # Convert to NumPy array

    data = []

    idx_start = 0
    idx_end = sequence_length

    for i in range(len(notes) // shift):
        if idx_end < len(notes):
            note_data = notes[idx_start:idx_end]
        else:
            note_data = notes[idx_start:]
            padding = np.zeros((sequence_length - len(note_data), 4))  # Pad if needed
            note_data = np.vstack([note_data, padding])
        idx_start += shift
        idx_end += shift
        data.append(note_data)

    return np.array(data)


@app.command()
def generate_rnn_dataset(cache_dir: str = ROOT_DIR.parent / "data_cache", n_workers: int = None) -> None:
    if not isinstance(cache_dir, Path):
        cache_dir = Path(cache_dir)
    file_paths = json.load(open(cache_dir / "midi_files.json"))

    n_workers = get_workers(n_workers)
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Create a list of futures
        futures = [executor.submit(midi_to_notes, file) for file in file_paths]

        # Process results as they complete
        for future in track(
            as_completed(futures), total=len(futures), description=f"Processing MIDI files using {n_workers} workers"
        ):
            result = future.result()
            if result is not None:
                results.append(result)

    # D = # of midi data X # sequence X # of features
    results = np.vstack(results)

    with open(cache_dir / "dataset.pkl", "wb") as f:
        pickle.dump(results, f)


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


class MIDIProcessor:
    def __init__(self, vocab_size=VOCAB_SIZE):
        # Reserved tokens
        self.pad_token = PAD_TOKEN
        self.sos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN

        # Token ranges
        self.start_pitch = 10
        self.num_pitches = 128
        self.start_duration = self.start_pitch + self.num_pitches
        self.num_durations = 100  # Quantized durations
        self.start_instrument = self.start_duration + self.num_durations
        self.num_instruments = 16  # General MIDI has 16 instrument families

        assert self.start_instrument + self.num_instruments < vocab_size, "Vocabulary size too small"

    def encode_note(self, pitch, duration_bin, instrument):
        pitch_token = self.start_pitch + pitch
        duration_token = self.start_duration + duration_bin
        instrument_token = self.start_instrument + instrument
        return [instrument_token, pitch_token, duration_token]

    def decode_token(self, token):
        if token < self.start_pitch:
            return {"type": "special", "value": token}
        elif token < self.start_duration:
            return {"type": "pitch", "value": token - self.start_pitch}
        elif token < self.start_instrument:
            return {"type": "duration", "value": token - self.start_duration}
        else:
            return {"type": "instrument", "value": token - self.start_instrument}

    def quantize_duration(self, duration):
        # Quantize duration to one of num_durations bins
        # Using log scale to better represent shorter durations
        max_duration = 4.0  # Maximum duration in seconds
        if duration > max_duration:
            duration = max_duration

        # Log scale quantization
        bin_idx = int(self.num_durations * math.log(1 + duration * 10) / math.log(1 + max_duration * 10))
        return min(bin_idx, self.num_durations - 1)

    def dequantize_duration(self, bin_idx):
        # Convert bin back to duration
        max_duration = 4.0
        return (math.exp(bin_idx * math.log(1 + max_duration * 10) / self.num_durations) - 1) / 10

    def midi_to_sequence(self, midi_file):
        """Convert MIDI file to token sequence"""
        if isinstance(midi_file, str):
            try:
                midi_data = pretty_midi.PrettyMIDI(midi_file)
            except Exception as e:
                os.remove(midi_file)
                return None
        else:
            midi_data = midi_file

        # Sort all notes by their start time
        all_notes = []
        for i, instrument in enumerate(midi_data.instruments):
            instrument_id = min(i, self.num_instruments - 1)  # Limit to available instrument tokens
            for note in instrument.notes:
                all_notes.append(
                    {"start": note.start, "end": note.end, "pitch": note.pitch, "instrument": instrument_id}
                )

        all_notes.sort(key=lambda x: x["start"])

        # Convert to token sequence
        tokens = [self.sos_token]
        for note in all_notes:
            duration = note["end"] - note["start"]
            duration_bin = self.quantize_duration(duration)
            note_tokens = self.encode_note(note["pitch"], duration_bin, note["instrument"])
            tokens.extend(note_tokens)

        tokens.append(self.eos_token)

        return tokens

    def sequence_to_midi(self, tokens, tempo=120):
        """Convert token sequence back to MIDI"""
        midi_data = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instruments = [pretty_midi.Instrument(program=i) for i in range(self.num_instruments)]

        current_time = 0.0
        current_instrument = 0
        current_pitch = 60

        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == self.eos_token:
                break

            token_info = self.decode_token(token)

            if token_info["type"] == "instrument":
                current_instrument = token_info["value"]
                i += 1
            elif token_info["type"] == "pitch":
                current_pitch = token_info["value"]

                # Look ahead for duration
                if i + 1 < len(tokens):
                    next_token = tokens[i + 1]
                    next_info = self.decode_token(next_token)
                    if next_info["type"] == "duration":
                        duration = self.dequantize_duration(next_info["value"])

                        # Create a note
                        note = pretty_midi.Note(
                            velocity=100, pitch=current_pitch, start=current_time, end=current_time + duration
                        )
                        try:
                            instruments[current_instrument].notes.append(note)
                        except IndexError:
                            pass
                        finally:
                            current_time += duration
                            i += 2
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1

        # Add instruments to MIDI data
        for instrument in instruments:
            if len(instrument.notes) > 0:
                midi_data.instruments.append(instrument)

        return midi_data


@app.command()
def generate_transformer_dataset(cache_dir: str = ROOT_DIR.parent / "data_cache", n_workers: int = None) -> None:
    if not isinstance(cache_dir, Path):
        cache_dir = Path(cache_dir)
    with open(cache_dir / "midi_files.json", "r") as f:
        file_paths = json.load(f)
    with open(cache_dir / "midi_files.json", "r") as f:
        file_paths = json.load(f)

    n_workers = get_workers(n_workers)
    results = []
    processor = MIDIProcessor()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Create a list of futures
        futures = [executor.submit(processor.midi_to_sequence, file) for file in file_paths]

        # Process results as they complete
        for future in track(
            as_completed(futures), total=len(futures), description=f"Processing MIDI files using {n_workers} workers"
        ):
            tokens = future.result()

            if tokens is not None and len(tokens) > 0:
                results.append(tokens)

    with open(cache_dir / "transformer_dataset.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    app()
