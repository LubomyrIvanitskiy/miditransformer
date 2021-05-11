import numpy as np
from itertools import accumulate
import matplotlib.pyplot as plt


# https://colab.research.google.com/drive/1Kr35XRlv1U0JDMluWRbotNMx5CthQ5Ee?usp=sharing

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


def solver(a, m, k, i=0, prnt=False, allow_zero=False):
    """
    if allow_zero 2x%3=0, x=0
    if not 2x%3=0, x=3
    """

    if m > k:
        if k == a == 0:
            return k
        elif k % a == 0:
            return k // a

    assert a > 0

    k_a = k % a
    m_a = m % a
    m_a = m_a if m_a > 0 else a
    z = a - k_a
    z = z if z > 0 else a

    if a == 1:
        x = k % m
        if not allow_zero:
            x = x if x > 0 else k
        if prnt:
            print('case 1')
            printer(i, a, m, k, m_a, k_a, z, x)
        return x

    if m % a == 0:
        if k % m % a == 0:
            x = k % m // a
            if not allow_zero:
                x = x if x > 0 else k // a
            if prnt:
                print('case 2')
                printer(i, a, m, k, m_a, k_a, z, x)
            return x
        else:
            print(f"{a}x%{m}={k} - No SOLUTIONS")
            raise ValueError('No solutions')

    new_x = solver(
        a=m_a,
        m=a,
        k=z,
        i=i + 1,
        prnt=prnt,
        allow_zero=allow_zero
    )
    x = (k + new_x * m) // a
    if prnt:
        print('case 3')
        printer(i, a, m, k, m_a, k_a, z, x)
    return x


def get_code_capacity(P):
    """
    Different P can encode differen amount of unique numbers. The maximum count is least common multiplier of P
    If P is co-prime - they have the maximum coding capacity
    for example P=[3,4,6] can encode only 12 numbers, while
    P=[3,4,5] - 20 numbers
    :return: max number can be encoded in such mixed radix system
    """
    return np.lcm.reduce(P)


def encode_modulo(n, P):
    """
    P is like a pseudo mixed radix system where each position has different radix P
    pseudo is becose encoding is not rely on shifting left each time when moving up between levels
    P should be in descending order
    :param n: number you want to encode
    :param P: modulo bases (periods)
    :return: array of code in this pseudo mix radix system
    """
    code = [n % P[-i] for i in range(1, len(P) + 1)][::-1]
    return code


def decode_modulo(code, P):
    lcms = np.fromiter(accumulate(P[::-1], np.lcm), int)[::-1]

    n = code[-1] % P[-1]

    for i in range(1, len(P)):

        bottom_p = lcms[-i]

        per_diff = bottom_p % P[-i - 1]  # rev

        current_next = n % P[-i - 1]
        wanted_next = code[-i - 1] % P[-i - 1]
        if wanted_next < current_next:
            wanted_next = wanted_next + P[-i - 1]

        distance = wanted_next - current_next

        distance = distance % P[-i - 1]
        if distance > 0:
            bottomp_scroll_count = solver(a=per_diff, m=P[-i - 1], k=distance, allow_zero=True)
            n = n + bottomp_scroll_count * bottom_p

    return n


def show_teeth(periods, size):
    """
    Show plot for modular positional system with different periods
    """
    x = np.arange(size)
    plt.vlines(x=x, ymin=0, ymax=2, linestyle=':')

    ys = [(x % p) / p for p in periods]

    for i in range(len(ys)):
        plt.plot(x, i + ys[i])
        plt.scatter(x, i + ys[i])

    plt.vlines(x=x, ymin=0, ymax=len(ys), linestyle=':')
    plt.show()


def get_freqs(base, dim):
    return np.asarray([base ** -i for i in range(dim)])


def list_encodings(periods):
    A = '0123456789ABCDEFGHJKLMNOPQRSTXYZ'
    max_value = get_code_capacity(periods)
    D = [A[:periods[i]] * (max_value // periods[i]) for i in range(len(periods))]

    codes = []
    for code in zip(*D):
        codes.append(''.join(code))
    return codes


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


def test_sinusoidal():
    test_value = 8
    base = 1.1
    dim = 6

    freqs = get_freqs(base=base, dim=dim // 2)
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


def test_modular():
    P = [11, 7, 4, 3]
    n = 333
    print("n=", n)
    code = encode_modulo(n, P)
    print("code=", code)
    decoded = decode_modulo(code=code, P=P)
    print("decoded=", decoded)
    if n == decoded:
        print("n==decoded")

    all_encodings = list_encodings(P)
    for n, code in enumerate(all_encodings):
        print(n, ':', code)


if __name__ == "__main__":
    # test_sinusoidal()
    #TODO pred_n only matches for hight discretisation and small value. Need to fix
    test_value = 456
    print("test_value", test_value)
    base = 1.5
    discretisation = 1000
    code = encode(test_value, dim=6, base=base)
    periods = 1/get_freqs(dim=3, base=base)
    int_periods = (periods*discretisation).astype(int)
    print("int_periods", int_periods)
    print("encoding capacity", get_code_capacity(int_periods[::-1]))
    angles = _get_angles(code)
    print("angles", angles)
    period_percentage = angles / (2 * np.pi)
    print("period_percentage", period_percentage)
    scores = (period_percentage * int_periods).astype(int)
    print("scores", scores)

    print(scores[::-1], int_periods[::-1])
    decoded = decode_modulo(code=scores[::-1], P=int_periods[::-1])
    print("decoded", decoded)

    pred_n = int(np.round((2*np.pi*decoded)/discretisation))
    print("pred_n", pred_n)

