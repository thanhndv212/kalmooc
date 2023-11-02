# https://www.ensta-bretagne.fr/kalmooc/
from roblib import *
import numpy as np
from numpy.linalg import det


def draw_room():
    for j in range(A.shape[1]):
        plot(
            array([A[0, j], B[0, j]]),
            array([A[1, j], B[1, j]]),
            label="wall_{}".format(j + 1),
        )


def draw(p, y, col):
    draw_tank(p, "darkblue", 0.1)
    p = p.flatten()
    y = y.flatten()
    for i in arange(0, 8):
        plot(
            p[0] + array([0, y[i] * cos(p[2] + i * pi / 4)]),
            p[1] + array([0, y[i] * sin(p[2] + i * pi / 4)]),
            color=col,
        )


def f(p, A, B):
    m = np.array([p[0], p[1]])
    U = np.zeros((2, 8))
    for jj in range(8):
        U[:, jj] = np.array(
            [np.cos(p[2] + jj * np.pi / 4), np.sin(p[2] + jj * np.pi / 4)]
        )
    D = []
    n_wall = A.shape[1]
    fresh_rays = list(range(8))
    used_rays = []
    for u_i in fresh_rays:
        if u_i not in used_rays and len(used_rays) < len(fresh_rays):
            # print("ray {} looking for ...".format(u_i + 1))
            u = U[:, u_i]
            D_buffer = []
            count = 0
            for i in range(n_wall):
                a = A[:, i]
                b = B[:, i]
                cond_1 = det(np.array([a - m, u]).T) * det(
                    np.array([b - m, u]).T
                )
                cond_2 = det(np.array([a - m, b - a]).T) * det(
                    np.array([u, b - a]).T
                )
                if cond_1 <= 0 and cond_2 >= 0:
                    count += 1
                    D_buffer.append(
                        det(np.array([a - m, b - a]))
                        / det(np.array([u, b - a]))
                    )
            if count > 0:
                D.append(np.array(D_buffer).min())
                print("ray {} came cross {} walls. ".format(u_i + 1, count))
                used_rays.append(u_i)
            else:
                print("ray {} hits no wall.".format(u_i + 1))

    return np.array(D)


def j(p, A, B, y):
    if f(p, A, B).shape[0] == 8:
        return np.sum((f(p, A, B) - y) ** 2)
    else:
        return 1e9


def simulated_annealing(p0, p, A, B, y, rate, init_temp, min_temp, count):
    curr_sol = p0
    curr_cost = j(p0, A, B, y)
    temp = init_temp
    while temp > min_temp:
        count += 1
        print("Iteration ", count + 1)
        new_sol = np.zeros_like(curr_sol)
        new_sol[0] = p[np.random.randint(0, 1000), 0]
        new_sol[1] = p[np.random.randint(0, 1000), 1]
        new_sol[2] = p[np.random.randint(0, 1000), 2]
        new_cost = j(new_sol, A, B, y)
        if new_cost < curr_cost:
            curr_sol = new_sol
            curr_cost = new_cost
        temp *= rate
         
    return curr_sol, curr_cost, count


A = np.array(
    [
        [0, 7, 7, 9, 9, 7, 7, 4, 2, 0, 5, 6, 6, 5],
        [0, 0, 2, 2, 4, 4, 7, 7, 5, 5, 2, 2, 3, 3],
    ]
)
B = np.array(
    [
        [7, 7, 9, 9, 7, 7, 4, 2, 0, 0, 6, 6, 5, 5],
        [0, 2, 2, 4, 4, 7, 7, 5, 5, 0, 2, 3, 3, 2],
    ]
)
y = np.array([6.4, 3.6, 2.3, 2.1, 1.7, 1.6, 3.0, 3.1])


px_min = 0
px_max = 9
py_min = 0
py_max = 7
ptheta_max = 2 * np.pi

p = np.random.rand(1000, 3)
p[:, 0] = px_max * p[:, 0]
p[:, 1] = py_max * p[:, 1]
p[:, 2] = ptheta_max * p[:, 2]

p0 = np.array([2, 1.5, np.pi/4])  # initial guess
rate = 0.99
init_temp = 10
min_temp = 0.01
count = 0
draw_room()
draw(p0, y, "red")
plt.show()
min_cost = 1e5
while min_cost > 0.5:
    p0, min_cost, count = simulated_annealing(p0, p, A, B, y, rate, init_temp, min_temp, count)
print("Min cost: ", min_cost)
plt.figure()
draw_room()
draw(p0, y, 'blue')
scatter(p[:, 0], p[:, 1], color="red")
plt.show()
