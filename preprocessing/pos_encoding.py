import numpy as np
import rns

"""
Encode position by sines and cosines and decode it back
Periods should be integers
"""


def get_int_periods(base, count, multiplier=100):
    """
    Get periods in form base**i for i in count converted to integers by multiplying by multiplier and then rounding
    Example:
        for base 1.1 and count=10 it should be round((1.1^0, 1.1^1, ... , 1.1^n)*multiplier)
        >>> get_int_periods(1.1, 10, multiplier=100)
        [194 177 161 146 133 121 110 100]
        >>> get_int_periods(1.1, 10, multiplier=10)
        [19 17 16 14 13 12 11 10]
    """
    return np.asarray([multiplier * base ** i for i in range(count)]).astype(int)[::-1]


def encode(t, P):
    """
    Get sinusoidal encoded vector
    code[2i] = np.sin(2PI*t/P[i]) for odd i
    code[2i+1] = np.cos(2PI*t/P[i]) for even i
    :param t: position
    :param P: periods
    :return: a code
    """
    code = np.zeros(len(P) * 2)
    # for P=1 period of sin will be exactly 1, not the 2pi
    code[np.arange(0, 2 * len(P), 2)] = np.sin(2 * np.pi * t / P)
    code[np.arange(1, 2 * len(P), 2)] = np.cos(2 * np.pi * t / P)
    return code


def decode(code, P):
    """
    Decode sinusoidal code back into position
    :param code: code obtained from the encode() method
    :param P: periods
    :return: position t
    """
    sines = code[np.arange(0, len(code), 2)]
    cosines = code[np.arange(1, len(code), 2)]

    acosines = np.arccos(cosines)
    acosines[sines < 0] = 2 * np.pi - acosines[sines < 0]
    angles = acosines / (2 * np.pi)
    rns_code = np.round(angles * P).astype(int)
    return rns.decode(rns_code, P)


if __name__ == '__main__':
    P = get_int_periods(1.1, 8, multiplier=100)
    print('P', P)
    print('Dynamic range', rns.get_dynamic_range(P))
    t = 655345507
    print('t', t)

    encoded = encode(t, P)
    print('positional encoded', encoded)

    decoded = decode(encoded, P)
    print('decoded', decoded)

    assert t == decoded
