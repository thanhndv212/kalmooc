
from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
xhat = array([[0,1]]).T
Γx = array([[0.7,0.3],[0.3,0.2]])
ax=init_figure(-10,10,-10,10)
clear(ax)
draw_ellipse_cov(ax,xhat, Γx, 0.99,'red')
pause(1)   