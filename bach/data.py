import json
import logging
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pretty_midi
import torch
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


def midi_to_notes(midi_path, sequence_length=150) -> np.ndarray:
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        return None

    notes = []
    for instrument in midi.instruments:
        if not instrument.is_drum:  # Exclude drum tracks
            for note in instrument.notes:
                notes.append([note.start, note.pitch, note.velocity])

    notes = sorted(notes, key=lambda x: x[0])  # Sort by start time
    notes = np.array(notes)  # Convert to NumPy array

    if len(notes) < sequence_length:
        padding = np.zeros((sequence_length - len(notes), 3))  # Pad if needed
        notes = np.vstack([notes, padding])

    return notes[:sequence_length]


@app.command()
def generate(cache_dir: str = ROOT_DIR.parent / "data_cache", n_workers: int = None) -> None:
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
    results = np.array(results)
    torch.save(results, cache_dir / "dataset.pt")


if __name__ == "__main__":
    app()
