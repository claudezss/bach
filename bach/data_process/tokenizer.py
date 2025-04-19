import logging

from midi_neural_processor.processor import decode_midi, encode_midi

logger = logging.getLogger(__name__)


def tokenize_midi(midi_file) -> list[int]:
    try:
        return encode_midi(midi_file)
    except Exception as e:
        logger.error(f"Error tokenizing MIDI file {midi_file}: {e}")
        return []
