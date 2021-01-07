import itertools

import numpy as np
import torch


def torch_flat(x):
    return torch.flatten(x, start_dim=1)


def class_name(o):
    return o.__class__.__name__


def get_bits(x: np.ndarray):
    return np.where(x <= 0, 0, 1)


def count_nonzero(t: np.ndarray):
    return np.count_nonzero(t)


def count_zero(t: np.ndarray):
    return t.size - count_nonzero(t)


def count_errors(b: np.ndarray, b_est: np.ndarray):
    return len(np.argwhere(b != b_est))


def tree_level(psv: np.ndarray):
    return count_nonzero(psv)


def is_root(psv: np.ndarray):
    return tree_level(psv) == 0


def is_goal(psv):
    return tree_level(psv) == psv.size


def tree_b(y: np.ndarray, R: np.ndarray, psv: np.ndarray):
    lv = tree_level(psv)
    if lv == 0:
        return 0
    else:
        b2 = np.square(y - R @ psv)
        return b2[-lv]


def tree_g(y: np.ndarray, R: np.ndarray, psv: np.ndarray):
    lv = tree_level(psv)
    if lv == 0:
        return 0
    else:
        b2 = np.square(y - R @ psv)
        return np.sum(b2[-lv:])


def partial_vector(sv: np.ndarray, lv: int):
    psv = np.zeros_like(sv)
    psv[-lv:] = sv[-lv:]
    return psv


def complex_channel(n_ant):
    H = np.random.randn(2 * n_ant, 2 * n_ant)
    H[n_ant:, n_ant:] = H[:n_ant, :n_ant]
    H[:n_ant, n_ant:] = -H[n_ant:, :n_ant]
    return H


def random_bits(shape):
    return np.where(np.random.uniform(0, 1, shape) < 0.5, 0, 1)


def qpsk(bits: np.ndarray):
    return 2 * bits - 1


def mmse_estimate(y, H):
    eye = np.eye(H.shape[1])
    H_pinv = np.linalg.inv(H.T @ H + eye) @ H.T
    z = H_pinv @ y
    return np.where(z < 0, -1, 1)


def average_vector_power(x: np.ndarray):
    if x.ndim == 3:
        squared_l2norm = np.sum(np.square(x), axis=1)
        return np.mean(squared_l2norm)
    else:
        raise ValueError("Unsupported ndim={}".format(x.ndim))


def _test_():
    M = 16  # 16-QAM
    n = int(np.log2(M))  # Number of bits per symbol
    k = 3  # 6 antennas

    test_snr_db = 21
    p = 10 ** (test_snr_db / 10)

    n_ant = 1

    constellation = np.array([x for x in itertools.product([1, -1, 3, -3], repeat=2 * n_ant)]).T
    s = np.zeros([10000, 2 * n_ant, 1])
    for i in range(10000):
        t = np.random.randint(0, constellation.shape[1])
        s[i, :, :] = constellation[:, t:t + 1]

    H = np.random.randn(10000, 2 * n_ant, 2 * n_ant)
    H[:, n_ant:, n_ant:] = H[:, :n_ant, :n_ant]
    H[:, :n_ant, n_ant:] = -H[:, n_ant:, :n_ant]

    z = np.sqrt(p / n_ant) / np.sqrt(10) * H @ s
    w = np.random.normal(0, 1, [10000, 2 * n_ant, 1])

    p_hs = average_vector_power(z)
    p_w = average_vector_power(w)
    snr = p_hs / p_w
    snr_db = 10 * np.log10(snr)

    print("p_hs={} p_w={} snr={} snr_db={}".format(p_hs, p_w, snr, snr_db))


if __name__ == "__main__":
    _test_()
