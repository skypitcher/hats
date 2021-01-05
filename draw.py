import matplotlib.pyplot as plt
import itertools
import numpy as np

from prelude import *

omega = [-3, -1, 1, 3]
constellation = np.array([x for x in itertools.product(omega, repeat=2)])
constellation = np.expand_dims(constellation, 2)

plt.figure()
plt.subplot(221)
plt.xticks(omega)
plt.yticks(omega)
plt.scatter(constellation[:, 0, :], constellation[:, 1, :])


for i in range(3):
    fig_loc = 222 + i

    snr = i * 5
    p = 10 ** (snr / 10)
    H = np.sqrt(p / 1) / np.sqrt(10) * np.expand_dims(complex_channel(1), 0)
    skewed = H @ constellation
    plt.subplot(fig_loc)
    plt.scatter(skewed[:, 0, :], skewed[:, 1, :])

    idx = 0
    x = constellation[idx:idx + 1, :, :]
    w = np.random.randn(1, 2, 1)
    z = H @ x
    y = z + w
    plt.scatter(y[:, 0, :], y[:, 1, :], s=96, marker='+')
    plt.scatter(z[:, 0, :], z[:, 1, :], s=96, marker='x')

plt.show()
