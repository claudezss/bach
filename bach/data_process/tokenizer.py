import logging
import os

from midi_neural_processor.processor import decode_midi, encode_midi

logger = logging.getLogger(__name__)


def tokenize_midi(midi_file) -> list[int]:
    try:
        return encode_midi(midi_file)
    except Exception as e:
        logger.error(f"Error tokenizing MIDI file {midi_file}: {e}")
        os.remove(midi_file)
        return []


def detokenize_midi(tokens, output_file) -> None:
    try:
        decode_midi(tokens, output_file)
    except Exception as e:
        logger.error(f"Error decoding tokens to MIDI file {output_file}: {e}")
        raise e
