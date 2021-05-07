import numpy as np
import midiwrap
from preprocessing import positional_encoding as pe

max_time_ms = 10_000  # sec
max_pitch = 127
max_duration = 5  # sec


def compute_base(emb_dim, max_value):
    return max_value ** (1 / emb_dim)


def encode_notes(midi: midiwrap.MidiFile, emb_dim):
    codes = []
    for i, note_record in midi.notes.iterrows():
        start, duration, pitch, _, _, _ = note_record.values
        pitch_code = pe.encode(pitch, dim=emb_dim, base=compute_base(emb_dim, max_pitch))
        start_code = pe.encode(start, dim=emb_dim, base=compute_base(emb_dim, max_time_ms))
        duration_code = pe.encode(duration, dim=emb_dim, base=compute_base(emb_dim, max_duration))
        codes.append(np.hstack((pitch_code, start_code, duration_code)))
    X = np.asarray(codes)
    return X


def decode_X(emb_dim):
    pass


if __name__ == "__main__":
    midi = midiwrap.MidiFile('../data/fur_elise.mid')
    X = encode_notes(midi, emb_dim=32)

    import matplotlib.pyplot as plt

    plt.imshow(X, aspect='auto')
    plt.show()
