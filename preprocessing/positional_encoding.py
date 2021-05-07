import numpy as np


# https://colab.research.google.com/drive/1ulRc8-WPv1QQ8qsfeVFviuiigMbbHofY?usp=sharing

def printer(i, a, m, k, m_a, k_a, z, x):
    print(f"{a}x%{m}={k}")
    print(f"a_{i}={a}")
    print(f"m_{i}={m}")
    print(f"k_{i}={k}")
    print(f"m_{i}_a={m_a}")
    print(f"k_{i}_a={k_a}")
    print(f"z_{i}={z}")
    print(f"x_{i}={x}")
    print(f"({a}*{x})%{m}={k}")
    print()


def solver(a, m, k, i=0, prnt=False):
    """
    Custom method to solve equations like (a*x)%m=k. Find x
    :return: x
    """
    assert a > 0

    k_a = k % a
    m_a = m % a
    z = a - k_a

    if a == 1:
        x = k % m
        if prnt:
            printer(i, a, m, k, m_a, k_a, z, x)
        return x

    if m % a == 0:
        if k % m % a == 0:
            return k // a
        else:
            raise ValueError('No solutions')

    new_x = solver(
        a=m_a,
        m=a,
        k=z,
        i=i + 1
    )
    x = (k + new_x * m) // a
    if prnt:
        printer(i, a, m, k, m_a, k_a, z, x)
    return x


def get_freqs(base, dim):
    return np.asarray([base ** -i for i in range(dim)])


def encode(t, dim, base, return_freqs=False):
    assert dim % 2 == 0
    code = np.zeros(dim)
    freqs = get_freqs(base, dim // 2)
    code[np.arange(0, dim, 2)] = np.sin(freqs * t)
    code[np.arange(1, dim, 2)] = np.cos(freqs * t)
    if return_freqs:
        return code, freqs
    else:
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
    return np.sum([(base ** i) * angles for i in range(dim - 1, -1, -1)])


def to_1_1(n, periods):
    delta = periods - periods[0]
    code = n % periods[0] - (n // periods[0] * delta) % periods
    code[code < 0] = periods[code < 0] + code[code < 0]
    return code[::-1]


def test_sound_hyperparams():
    import matplotlib.pyplot as plt

    max_time_ms = 10_000  # max, melody duration, sec
    max_pitch = 127  # 0 - 127
    max_duration = 5  # note duration, sec

    emb_dim = 16

    base_time = max_time_ms ** (1 / emb_dim)
    print("base_time", base_time)

    base_pitch = max_pitch ** (1 / emb_dim)
    print("base_pitch", base_pitch)

    base_duration = max_duration ** (1 / emb_dim)
    print("base_duration", base_duration)

    test_value = 26
    print("test value", test_value)

    code = encode(test_value, dim=emb_dim, base=base_time)
    print("encoded", code)

    decoded = decode(code, emb_dim, base_time)
    print("decoded", decoded)
    # plt.imshow(code[None])
    # plt.show()


if __name__ == "__main__":
    test_value = 8
    base = 1.1
    dim = 6

    freqs = get_freqs(base=1.1, dim=dim // 2)
    periods = 2 * np.pi / freqs

    integer_periods = periods  ## 100, 110, 121
    print("integer_periods", integer_periods)
    enco_11 = to_1_1(test_value, integer_periods)
    print(f"{test_value} to enco_11=", enco_11)
    print()

    code = encode(test_value, dim=6, base=1.1)
    angles = _get_angles(code)
    print("angles", angles)
    period_percentage = angles / (2 * np.pi)
    print("period_percentage", period_percentage)
    scores = period_percentage * periods
    print("scores", scores[::-1])
