from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py

def j(p1,p2):
    return p1**2+p1*p2

xmin,xmax,ymin,ymax=-6,8,-6,8
ax=init_figure(xmin,xmax,ymin,ymax)

M=array([[-1,1,3,4],[-1,2,2,5]])
m=size(M,1)
for i in range(0,m): plot(M[0,i],M[1,i],'ro')

P1,P2 = meshgrid(arange(xmin,xmax,0.1), arange(ymin,ymax,0.1))
F=j(P1,P2)
contour(P1,P2,F,50)

pause(10)
