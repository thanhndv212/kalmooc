from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
from mpl_toolkits import mplot3d
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
%matplotlib

# ex5

xmin, xmax, ymin, ymax = -6, 8, -6, 8
# ax = init_figure(xmin, xmax, ymin, ymax)

M = array([[-1, 1, 3, 4], [-1, 2, 2, 5]])
m = size(M, 1)
fig = plt.figure(1)

for i in range(0, m):
    plot(M[0, i], M[1, i], "ro")
    text(M[0, i], M[1, i], "$M_{}$".format(i + 1))
grid(True)

def fi(p, Mi):
    """Compute distance vector from P to 4 markers of M
    Input: P is a 2x1 vector.
    """
    return norm(p - Mi)

def f(p):
    """Compute distance vector from P to 4 markers of M
    Input: P is a 2x1 vector.
    """
    return array([fi(p, M[:,i]) for i in range(m)])


# ex5.1
P_0 = array([[2, -3]])
plot(P_0[0, 0], P_0[0, 1], "bo")
print("Distance at [2, -3]: ", f(P_0))

# ex5.2
y = array([4, 5, 5, 8])


def j(P):
    """Compute deviation vector from P to 4 markers of M"""
    P1 = P[0]
    P2 = P[1]
    error = 0
    for i in range(y.shape[0]):
        error += (sqrt((P1 - M[0, i]) ** 2 + (P2 - M[1, i]) ** 2) - y[i]) ** 2
    return error


P1, P2 = meshgrid(arange(xmin, xmax, 0.1), arange(ymin, ymax, 0.1))
F = j([P1, P2])
F_min = F.min()
xmin, ymin = np.unravel_index(np.argmin(F), F.shape)
print("Fmin at: ", P1[xmin, ymin], P2[xmin, ymin], F_min)
fig2 = plt.figure(2)
ax = plt.axes(projection='3d')
ax.contour3D(P1, P2, F, levels=50)
ax.contour(P1, P2, F, zdir='z', offset=F_min, levels=50)
ax.plot(P1[xmin, ymin], P2[xmin, ymin], F_min, marker='o', markersize=5, color="red")

def newton(p0, epsilon, max_iter=100):
    """Newton's method to find a root of f
    Input: f is a function of one variable
    x0 is the initial guess
    epsilon is the precision
    Output: the root
    """
    p = p0
    count = 0
    while norm(f(p))> epsilon:
        Mat = array([[(p[0] - M[0, i])/fi(p, M[:,i]),(p[1] - M[1, i])/fi(p, M[:,i])] for i in range(m)])
        p = p + (np.linalg.pinv(Mat.T @ Mat)) @ Mat.T @ (y-f(p))
        count += 1
        if count > max_iter:
            print("Max iteration reached")
            break
    return p

P = newton(p0=[4,3], epsilon=0.01)
print("Solution by Newton method: ",P, f(P))