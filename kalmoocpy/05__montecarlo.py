from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py

import numpy as np

P = 2 * np.random.rand(100, 2)

x = np.array([[0], [0]])

y = np.array([0, 1, 2.5, 4.1, 5.8, 7.5])
ym = np.array([0, 0, 0, 0, 0, 0])
m = y.shape[0]
x = np.zeros((m, 2))


def get_ym(p):
    A = np.array([[1, 0], [p[0], 0.3]])
    B = np.array([p[1], 1 - p[1]])
    C = np.array([1, 1])

    for i in range(1, m):
        x[i, :] = A @ x[i - 1, :] + B
        ym[i] = C @ x[i, :]
    return ym


def cost(y, ym):
    return np.abs(y - ym).max()


epsilon = 1
pm = []
for p in P:
    ym_p = get_ym(p)
    if cost(y, ym) < epsilon:
        pm.append(p)
        scatter(p[0], p[1], color="red")
    else:
        scatter(p[0], p[1], color="black")



a = np.linspace(0, 2, 100)
b = 1.2 / (0.7 + a)

plot(a, b, )
plt.show()
