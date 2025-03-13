import numpy as np
import pretty_midi
import torch

from bach.model import MusicRNN


def generate_random_midi(output_path="random_music.mid", num_notes=30, instrument_name="Acoustic Grand Piano"):
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


# generate_random_midi("input.mid")
generate_music()
