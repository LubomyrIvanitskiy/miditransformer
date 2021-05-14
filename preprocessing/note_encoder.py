import numpy as np
import midiwrap
from preprocessing import pos_encoding as pe

max_time = 10_000  # sec
ms_in_sec= 1000
max_pitch = 127
max_duration = 5  # sec


def compute_base(emb_dim, max_value):
    return max_value ** (1 / emb_dim)


def get_notes(midi: midiwrap.MidiFile, track_name=None):
    notes = midi.notes[midi.notes.Name == track_name] if track_name else midi.notes
    melody = []
    for i, note_record in notes.iterrows():
        start, duration, pitch, _, _, _ = note_record.values
        melody.append((pitch, start, duration))
    return melody


def encode_notes(midi: midiwrap.MidiFile, emb_dim, track_name=None):
    codes = []
    notes = midi.notes[midi.notes.Name == track_name] if track_name else midi.notes
    for i, note_record in notes.iterrows():
        start, duration, pitch, _, _, _ = note_record.values
        # print(f"pitch {i}", pitch)
        pitch_code = pe.encode(
            t=pitch,
            P=pe.get_int_periods(base=compute_base(emb_dim, max_pitch), count=emb_dim)
        )
        start_code = pe.encode(
            t=np.round(start*ms_in_sec).astype(int),
            P=pe.get_int_periods(base=compute_base(emb_dim, max_time), count=emb_dim)
        )
        duration_code = pe.encode(
            t=np.round(duration*ms_in_sec).astype(int),
            P=pe.get_int_periods(base=compute_base(emb_dim, max_duration), count=emb_dim)
        )
        codes.append(np.hstack((pitch_code, start_code, duration_code)))
    X = np.asarray(codes)
    return X


def decode_X(codes, emb_dim):
    pitch_P = pe.get_int_periods(base=compute_base(emb_dim, max_pitch), count=emb_dim)
    time_P = pe.get_int_periods(base=compute_base(emb_dim, max_time), count=emb_dim)
    duration_P = pe.get_int_periods(base=compute_base(emb_dim, max_duration), count=emb_dim)
    splitted = codes.reshape(X.shape[0], 3, codes.shape[-1] // 3)
    melody = []
    for pitch_code, time_code, duration_code in splitted:
        decoded_pitch = pe.decode(pitch_code, pitch_P)
        decoded_time = pe.decode(time_code, time_P)
        decoded_duration = pe.decode(duration_code, duration_P)
        melody.append((decoded_pitch, decoded_time/ms_in_sec, decoded_duration/ms_in_sec))
    return melody


if __name__ == "__main__":
    midi = midiwrap.MidiFile('../data/fur_elise.mid')
    dim = 32
    track_names = midi.track_names()

    X = encode_notes(midi, emb_dim=dim, track_name=track_names[0])
    decoded = decode_X(X, emb_dim=dim)
    real_notes = get_notes(midi, track_name=track_names[0])

    melody = midiwrap.MelodyBuilder()
    instrument = 'piano'
    for pitch, time, duration in decoded:
        melody.add_note(pitch, time, duration, instrument)
    melody.write_to_file('decoded_melody4.mid')
