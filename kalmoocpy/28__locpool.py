from roblib import *

def f(x,u):
    mx,my,θ,v,δ=list(x[0:5,0])
    u1,u2,u3=list(u[0:3,0])
    return array([[v*cos(θ)],[v*sin(θ)],[u1],[u2],[u3]])

def draw_robot(mx,my,θ,δ,d,col):
    draw_tank(array([[mx],[my],[θ]]),col,2)
    plot([mx,mx+cos(θ+δ)*d],[my,my+sin(θ+δ)*d],"red")

def draw_simu(x,d):
    clear(ax)
    mx,my,θ,v,δ=list(x[0:5,0])
    draw_polygon(ax,P,"cyan")
    draw_robot(mx,my,θ,δ,d,'darkblue')
    pause(0.01)

ax=init_figure(-60,60,-60,60)
Rx,Ry=40,50
dt=0.05
P=array([[Rx,-Ry],[Rx,Ry],[-Rx,Ry],[-Rx,-Ry],[Rx,-Ry]])
x=array([[10],[-10],[1],[3],[0]]) # x=[x,y,θ,v,δ]
for t in arange(0,2,dt):
    u=array([[0.2],[0],[2]])
    draw_simu(x,10)
    x=x+f(x,u)*dt






