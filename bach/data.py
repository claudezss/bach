import logging
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from bach import ROOT_DIR

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)


def download_single_file(data, repo_id: str, cache_dir: Path) -> Optional[str]:
    """Download a single MIDI file and return its path."""
    try:
        filename = data["file_name"]
        midi_file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", cache_dir=cache_dir)
        logger.debug(f"MIDI file downloaded to: {midi_file_path}")
        return midi_file_path
    except Exception as e:
        logger.debug(f"Failed to download {data.get('file_name', 'unknown')}: {str(e)}")
        return None


def load(cache_dir: Path = ROOT_DIR.parent / "data_cache", n_workers: int = None) -> List[str]:
    """
    Load and download MIDI files using parallel processing.

    Args:
        cache_dir: Directory to store downloaded files
        n_workers: Number of worker processes (defaults to CPU count if None)

    Returns:
        List of paths to the downloaded MIDI files
    """
    # Create cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Set number of workers (default to CPU count if not specified)
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free

    repo_id = "drengskapur/midi-classical-music"

    # Load dataset (contains file names only)
    ds = load_dataset(repo_id, cache_dir=str(cache_dir.absolute()), split="train")

    # Use ProcessPoolExecutor instead of multiprocessing.Pool
    midi_file_paths = []

    # Using ThreadPoolExecutor can also work and avoids multiprocessing issues
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Create a list of futures
        futures = [executor.submit(download_single_file, data, repo_id, cache_dir) for data in ds]

        # Process results as they complete
        for future in tqdm(
            as_completed(futures), total=len(futures), desc=f"Downloading MIDI files using {n_workers} workers"
        ):
            result = future.result()
            if result:
                midi_file_paths.append(result)

    logger.debug(f"Downloaded {len(midi_file_paths)} MIDI files successfully")
    return midi_file_paths


if __name__ == "__main__":
    midi_paths = load()
