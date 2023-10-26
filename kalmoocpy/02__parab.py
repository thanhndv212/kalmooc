# https://www.ensta-bretagne.fr/kalmooc/

# from roblib import *
import numpy as np
import matplotlib.pyplot as plt

%matplotlib
# Ex1
def ex1():
    # %matplotlib
    x, y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
    f = x * y
    u = y
    v = x
    # Vector field of gradient of f = xy
    fi1 = plt.figure(1)
    plt.quiver(x, y, u, v, scale=15)

    # Contour lines of f
    plt.contour(x, y, f, 50, cmap="jet", lw=0.2)
    plt.title("Vector field of gradient of f = xy")

    x, y = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(-1, 1, 30))
    f_1 = 2 * x**2 + x * y + 4 * y**2 + y - x + 3
    u_1 = 4 * x + y - 1
    v_1 = x + 8 * y + 1

    # Vector field of gradient of f_1
    fi2 = plt.figure(2)
    plt.quiver(x, y, u_1, v_1, scale=15)
    # Contour lines of f_1
    plt.contour(x, y, f_1, 50, cmap="jet", lw=0.2)
    plt.title("Vector field of gradient of f = 2x**2 + xy + 4y**2 + y − x + 3.")


# Ex2
def ex2():
    # %matplotlib
    t = np.array([-3, -1, 0, 2, 3, 6])
    y = np.array([17, 3, 1, 5, 11, 46])
    M = np.vstack((t**2, t, (np.ones(len(t))))).T
    p_est = (np.linalg.pinv(M.T @ M)) @ (M.T @ y)

    # filtered measure
    y_hat = M @ p_est

    # vector of residuals
    r = y - y_hat

    # plot
    plt.plot(t, y, label="measures")
    plt.plot(t, y_hat, label="estimated")
    plt.title("Measures and estimated values of y(t) = a*t**2 + b*t + c")
    plt.legend()
    plt.show()


# Ex3
def ex3():
    # %matplotlib
    U = np.array([4, 10, 10, 13, 15])
    T_r = np.array([0, 1, 5, 5, 3])
    Ohm = np.array([5, 10, 8, 14, 17])
    M = np.vstack((U, T_r)).T
    p_est = (np.linalg.pinv(M.T @ M)) @ (M.T @ Ohm)
    y_hat = M @ p_est
    r = Ohm - y_hat
    print("filtered measured values: ", y_hat)
    print("vector of residuals: ", r)

    y_0 = np.array([20, 10]) @ p_est
    print("deduced value: ", y_0)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    U_ = np.linspace(U.min(), 20, 10)
    Tr_ = np.linspace(T_r.min(), 10, 10)
    M_ = np.array([U_, Tr_]).T
    Ohm_ = M_ @ p_est

    xx, yy = np.meshgrid(U_, Tr_)
    z = p_est[0] * xx + p_est[1] * yy

    ax.plot_surface(xx, yy, z, alpha=0.2)
    ax.scatter(U, T_r, Ohm, c="red", marker="o", label="measures")
    ax.scatter(U, T_r, y_hat, c="green", marker="o", label="estimated")
    ax.scatter(20, 10, y_0, c="green", marker="*", label="deduced")
    ax.legend()
    ax.set_xlabel("U")
    ax.set_ylabel("T_r")
    ax.set_zlabel("Ohm")
    ax.set_title("Measures and estimated values of Ohm(U, T_r)")


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
