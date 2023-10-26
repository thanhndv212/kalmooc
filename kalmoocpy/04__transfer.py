# https://www.ensta-bretagne.fr/kalmooc/

import numpy as np
import matplotlib.pyplot as plt


# Ex4:
def ex4():
    # y(k) = -a1*y(k-1) - a0*y(k-2) + b1*u(k-1) + b0*u(k-2)

    k = np.arange(8)
    u = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    y = np.array([0, -1, -2, 3, 7, 11, 16, 36])
    
    # measures
    y_meas = y[2 : len(k)]

    # regressor
    M = np.zeros((len(k) - 2, 4))
    M[:, 0] = -y[1 : (len(k) - 1)]
    M[:, 1] = -y[0 : (len(k) - 2)]
    M[:, 2] = u[1 : (len(k) - 1)]
    M[:, 3] = u[0 : (len(k) - 2)]

    # estimate
    (a1, a0, b1, b0) = (np.linalg.pinv(M.T @ M)) @ (M.T @ y_meas)
    print(f"a1 = {a1}, a0 = {a0}, b1 = {b1}, b0 = {b0}")


if __name__ == "__main__":
    ex4()
