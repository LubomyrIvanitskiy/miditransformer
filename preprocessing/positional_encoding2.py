import numpy as np
import rns


def get_moduli(base, count, multiplier=100):
    return np.asarray([multiplier * base ** i for i in range(count)]).astype(int)[::-1]


def encode(t, P):
    code = np.zeros(len(P) * 2)
    code[np.arange(0, 2*len(P), 2)] = np.sin(t / P)
    code[np.arange(1, 2*len(P), 2)] = np.cos(t / P)
    return code


def _get_angles(pos_enc):
    sines = pos_enc[np.arange(0, len(pos_enc), 2)]
    cosines = pos_enc[np.arange(1, len(pos_enc), 2)]

    acosines = np.arccos(cosines)
    acosines[sines < 0] = 2 * np.pi - acosines[sines < 0]
    angles = acosines
    return angles


def decode(pos_enc, dim, base):
    angles = _get_angles(pos_enc)

    return angles


if __name__ == '__main__':
    P = get_moduli(1.1, 8)
    print('P', P)
    t = 28
    rns_encoded = rns.encode(t, P)
    print('rns encoded', rns_encoded)

    encoded = encode(t, P)
    print('sin encoded', encoded)

    angles= _get_angles(encoded)
    print('angles', angles)
    code = np.round(angles*P).astype(int)
    print('angles*P', code)

    rns_decoded = rns.decode(code, P)
    print('rns decoded', rns_decoded)



