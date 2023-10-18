from roblib import *

D=loadcsv("lidar_data.csv")

m=len(D) # = 512
n = 10
X = D[:, 0]
Y = D[:, 1]

ax=init_figure(-0.5,3.5,-1.5,2.5)
plot(X,Y,'black')
pause(1)
