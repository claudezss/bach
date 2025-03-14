import json
import logging
import multiprocessing
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


class MIDIProcessor:
    def __init__(self, resolution=100):
        self.resolution = resolution  # Time steps per quarter note
        # Define vocabulary size based on MIDI event types
        # Notes (128), velocities (32 bins), time shifts (100 bins), and special tokens
        self.note_range = 128  # MIDI notes 0-127
        self.velocity_bins = 32  # Quantized velocity levels
        self.time_bins = 100  # Quantized time shift bins

        # Token indices:
        # 0-127: Note-on events (pitch)
        # 128-159: Velocity bins
        # 160-259: Time-shift bins
        # 260-387: Note-off events (pitch)
        # 388-389: Special tokens (PAD, EOS)
        self.vocab_size = 390

        # Special tokens
        self.PAD_TOKEN = 388
        self.EOS_TOKEN = 389

    def quantize_velocity(self, velocity):
        """Quantize velocity (0-127) into fewer bins"""
        return 128 + min(int(velocity * self.velocity_bins / 128), self.velocity_bins - 1)

    def quantize_time(self, time_delta):
        """Quantize time delta into time bins"""
        # Convert time in seconds to time bins
        time_bin = min(int(time_delta * self.resolution), self.time_bins - 1)
        return 160 + time_bin

    def midi_to_tokens(self, midi_file: str) -> List[int]:
        """Convert a MIDI file to a sequence of tokens"""
        try:
            # Load MIDI file
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            tokens = []

            # Process each instrument
            for instrument in midi_data.instruments:
                # Skip drum tracks
                if instrument.is_drum:
                    continue

                # Sort notes by start time
                notes = sorted(instrument.notes, key=lambda note: note.start)

                # Process notes
                last_time = 0
                for note in notes:
                    # Add time shift token
                    time_delta = note.start - last_time
                    if time_delta > 0:
                        time_token = self.quantize_time(time_delta)
                        tokens.append(time_token)

                    # Add note-on token
                    tokens.append(note.pitch)  # Note on: 0-127

                    # Add velocity token
                    velocity_token = self.quantize_velocity(note.velocity)
                    tokens.append(velocity_token)

                    # Update last time for the next note
                    last_time = note.start

                    # Add time shift to note-off
                    note_duration = note.end - note.start
                    time_token = self.quantize_time(note_duration)
                    tokens.append(time_token)

                    # Add note-off token
                    tokens.append(260 + note.pitch)  # Note off: 260-387

            tokens.append(self.EOS_TOKEN)  # End of sequence
            return tokens
        except Exception as e:
            print(f"Error processing MIDI file {midi_file}: {e}")
            return []

    def tokens_to_midi(self, tokens: List[int], output_file: str):
        """Convert a sequence of tokens back to a MIDI file"""
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Piano by default

        current_time = 0.0
        active_notes = {}  # pitch -> (start_time, velocity)

        i = 0
        while i < len(tokens):
            token = tokens[i]

            # Note-on event
            if 0 <= token < 128:
                pitch = token

                # Get velocity (should follow note-on)
                if i + 1 < len(tokens) and 128 <= tokens[i + 1] < 160:
                    velocity_token = tokens[i + 1]
                    velocity = int((velocity_token - 128) * 128 / self.velocity_bins)
                    i += 1
                else:
                    velocity = 64  # Default velocity

                # Store note start info
                active_notes[pitch] = (current_time, velocity)

            # Time-shift event
            elif 160 <= token < 260:
                time_bin = token - 160
                time_delta = time_bin / self.resolution
                current_time += time_delta

            # Note-off event
            elif 260 <= token < 388:
                pitch = token - 260
                if pitch in active_notes:
                    start_time, velocity = active_notes[pitch]
                    # Create note
                    note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=current_time)
                    instrument.notes.append(note)
                    del active_notes[pitch]

            # Handle special tokens or unexpected values
            elif token == self.EOS_TOKEN:
                break

            i += 1

        # Add any still-active notes with a small duration
        for pitch, (start_time, velocity) in active_notes.items():
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start_time,
                end=current_time + 0.1,  # Small duration to end the note
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)
        midi.write(output_file)


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
        futures = [executor.submit(processor.midi_to_tokens, file) for file in file_paths]

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
