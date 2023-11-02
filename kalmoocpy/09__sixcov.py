from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
import numpy as np
θ = arange(0, 2 * pi + 0.1, 0.1)
x = cos(θ)
y = sin(θ)
X = array([x, y])
# plot2D(X, "black", 1)


# A = array([[4, 1], [1, 3]])
# B = sqrtm(A)

A1 = np.array([[1, 0], [0, 3]])
A2 = np.array(
    [
        [np.cos(np.pi / 4), -np.sin(np.pi / 4)],
        [np.sin(np.pi / 4), np.cos(np.pi / 4)],
    ]
)

cov = []
cov_1 = np.eye(2)
cov_2 = 3 * cov_1
cov_3 = A1 @ cov_2 @ A.T + cov_1
cov_4 = A2 @ cov_3 @ A2.T
cov_5 = cov_4 + cov_3
cov_6 = A2 @ cov_5 @ A2.T
cov = [cov_1, cov_2, cov_3, cov_4, cov_5, cov_6]


def plot_conf_ell(cov, name):
    y = sqrtm(cov) @ X
    plt.plot(y[0, :], y[1, :], label=name)


for i in range(6):
    plot_conf_ell(cov[i], "covariance_{}".format(i + 1))

plt.legend()
ax = plt.gca()
ax.set_aspect("equal", adjustable="box")
