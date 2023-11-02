from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
import numpy as np
%matplotlib

N = 1000
X0 = randn(2, N)
x_bar = array([1, 2])
cov_x = array([[4, 3], [3, 3]])
X = np.full((N, 2), x_bar).T + sqrtm(cov_x) @ X0


def plot_cloud(ax, X, x_bar):
    ax.scatter(X[0], X[1], marker="+", color="blue")
    ax.scatter(x_bar[0], x_bar[1], color="red")


ax1 = init_figure(-10, 10, -10, 10)
for conf_n in [0.9, 0.99, 0.999]:
    draw_ellipse_cov(ax1, x_bar, cov_x, conf_n, [1, 0.8, 0.8])
plot_cloud(ax1, X, x_bar)

#########################################################
dt = 0.01
ts = [0, 1, 2, 3, 4, 5]
alpha = dt * randn(2, N)
alpha_bar = np.array([0, 0])
cov_alpha = dt * np.eye(2)
A = np.array([[0, 1], [-1, 0]])
b = np.array([2, 3])


def x_dot(x, t, N):
    return A @ x + np.full((N, 2), b * np.sin(t)).T


# initial
X0 = randn(2, N)

x_bar = array([1, 2])
cov_x = array([[4, 3], [3, 3]])
X = np.full((N, 2), x_bar).T + sqrtm(cov_x) @ X0

X_bar = []
X_bar.append(x_bar)
COV_x = []
COV_x.append(cov_x)
T = np.arange(0, 5, dt)
Xs = []
Xs.append(X)
for t in T:
    x_bar = x_bar + dt * (A @ x_bar +  b * np.sin(t)) + alpha_bar
    X_bar.append(x_bar)
    cov_x = (np.eye(2) + dt * A) @ cov_x @ (np.eye(2) + dt * A).T + cov_alpha
    COV_x.append(cov_x)
    X = X + dt * x_dot(X, t, N) + alpha
    Xs.append(X)

ax2 = init_figure(-10, 10, -10, 10)
for i in range(500):
    
    draw_ellipse_cov(
        ax2, X_bar[i], COV_x[i], 0.9, col=[1, 0.8, 0.8], coledge="green"
    )
    draw_ellipse_cov(
        ax2, X_bar[i], COV_x[i], 0.99, col=[1, 0.8, 0.8], coledge="orange"
    )
    ax2.scatter(Xs[i][0], Xs[i][1], marker="+", color="blue")
    # ax2.scatter(X_bar[i][0], X_bar[i][1], color="red")
    pause(0.001)
    ax2.clear()
    ax2.xmin = -10
    ax2.xmax = 10
    ax2.ymin = -10
    ax2.ymax = 10
