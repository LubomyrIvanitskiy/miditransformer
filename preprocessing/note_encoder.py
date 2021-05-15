import numpy as np
import midiwrap
from preprocessing import pos_encoding as pe

max_time = 10  # sec
ms_in_sec = 1000
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


def encode_notes(midi: midiwrap.MidiFile, emb_dim=None, P=None, track_name=None):
    if P is not None:
        P = np.asarray(P)
    codes = []
    notes = midi.notes[midi.notes.Name == track_name] if track_name else midi.notes
    for i, note_record in notes.iterrows():
        start, duration, pitch, _, _, _ = note_record.values
        # print(f"pitch {i}", pitch)
        pitch_moduli = P if P is not None else pe.get_int_periods(base=compute_base(emb_dim, max_pitch), count=emb_dim)
        start_moduli = P if P is not None else pe.get_int_periods(base=compute_base(emb_dim, max_time), count=emb_dim)
        duration_moduli = P if P is not None else pe.get_int_periods(base=compute_base(emb_dim, max_duration),
                                                                     count=emb_dim)
        pitch_code = pe.encode(
            t=pitch,
            P=pitch_moduli
        )
        start_code = pe.encode(
            t=np.round(start * ms_in_sec).astype(int),
            P=start_moduli
        )
        duration_code = pe.encode(
            t=np.round(duration * ms_in_sec).astype(int),
            P=duration_moduli
        )
        codes.append(np.hstack((pitch_code, start_code, duration_code)))
    X = np.asarray(codes)
    return X


def decode_X(codes, emb_dim=None, P=None):
    pitch_P = P if P is not None else pe.get_int_periods(base=compute_base(emb_dim, max_pitch), count=emb_dim)
    time_P = P if P is not None else pe.get_int_periods(base=compute_base(emb_dim, max_time), count=emb_dim)
    duration_P = P if P is not None else pe.get_int_periods(base=compute_base(emb_dim, max_duration),
                                                                 count=emb_dim)
    splitted = codes.reshape(codes.shape[0], 3, codes.shape[-1] // 3)
    melody = []
    for pitch_code, time_code, duration_code in splitted:
        decoded_pitch = pe.decode(pitch_code, pitch_P)
        decoded_time = pe.decode(time_code, time_P)
        decoded_duration = pe.decode(duration_code, duration_P)
        melody.append((decoded_pitch, decoded_time / ms_in_sec, decoded_duration / ms_in_sec))
    return melody


if __name__ == "__main__":
    midi = midiwrap.MidiFile('../data/fur_elise.mid')
    dim = 12
    track_names = midi.track_names()

    coprime_P = np.asarray([47, 43, 41, 37, 31, 29, 23, 19, 17, 13, 11, 7])

    X = encode_notes(midi, P=coprime_P, track_name=track_names[0])
    # X[0][0]=0.5
    decoded = decode_X(X, P=coprime_P)
    real_notes = get_notes(midi, track_name=track_names[0])

    melody = midiwrap.MelodyBuilder()
    instrument = 'piano'
    for pitch, time, duration in decoded:
        melody.add_note(pitch%127, time%max_time, duration%max_duration, instrument)
    melody.write_to_file('decoded_melody4.mid')
