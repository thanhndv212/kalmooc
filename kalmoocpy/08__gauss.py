from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
import numpy as np
from numpy.linalg import det
from numpy import exp

x, y = meshgrid(arange(-5, 5, 0.1), arange(-5, 5, 0.1))

# z = exp(-((x - 1) ** 2 + y**2 + (x - 1) * y))

x_exp = np.array([1, 2])
cov_x = np.array([[1, 0], [0, 1]])


def gauss(cov_x, x, y, x_exp):
    E = (
        cov_x[0, 0] * (x - x_exp[0]) ** 2
        + 2 * cov_x[0, 1] * (x - x_exp[0]) * (y - x_exp[1])
        + cov_x[1, 1] * (y - x_exp[1]) ** 2
    )

    gauss_x = (1 / 2 * np.pi * (np.sqrt(det(cov_x)))) * exp(-0.5 * E)
    return gauss_x


gauss_x = gauss(cov_x, x, y, x_exp)


def plot_gauss(x, y, gauss_x):
    plot_contour(x, y, gauss_x)
    plot_surface(x, y, gauss_x)

def plot_surface(x, y, gauss_x):
    ax = figure3D()
    ax.plot_surface(x, y, gauss_x)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
def plot_contour(x, y, gauss_x):
    fig = figure()
    contour(x, y, gauss_x)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    pause(1)


A = np.array(
    [
        [np.cos(np.pi / 6), -np.sin(np.pi / 6)],
        [np.sin(np.pi / 6), np.cos(np.pi / 6)],
    ]
) @ (np.array([[1, 0], [0, 3]]))
b = np.array([2, -5])

y_exp = A @ x_exp + b
cov_y = A @ cov_x @ A.T
x, y = meshgrid(arange(-5, 5, 0.1), arange(-5, 5, 0.1))

gauss_y = gauss(cov_y, x, y, y_exp)
plot_gauss(x, y, gauss_y)



plot_contour(x, y, gauss(np.eye(2), x, y, np.zeros(2)))

plot_contour(x, y, gauss(3*np.eye(2), x, y, np.zeros(2)))