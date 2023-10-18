from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py

def f(x,u):
    return (array([[x[3,0]*cos(x[4,0])*cos(x[2,0])],
                       [x[3,0]*cos(x[4,0])*sin(x[2,0])],
                       [x[3,0]*sin(x[4,0])/L],
                       [u[0,0]], [u[1,0]]]))

L,dt = 3,0.1
u = array([[0,0]]).T
x = array([[0,0,pi/3,4,0.2]]).T

Gα = array([[0,0,0,0,0],
                [0,0,0,0,0],
                [0,0,0.01*dt,0,0],
                [0,0,0,0.01*dt,0],
                [0,0,0,0,0.01*dt]])

ax=init_figure(-50,50,-50,50)

for t in arange(0,2,dt) :
    clear(ax)
    draw_car(x,'blue',L)  	
    uz = array([[0,0,dt*u[0,0]]]).T
    x = x + dt*f(x,u) + mvnrnd1(Gα)
pause(1)
