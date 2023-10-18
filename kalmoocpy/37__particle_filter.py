from roblib import *

def θ(t):  return 0.2*t

def g(x):
    y=[norm(x-M[:,j].reshape((2,1))) for j in range(0,m)]
    return array(y).reshape((m,1))


def draw_robot(x,col='black'):
    draw_tank(array([[x[0,0]],[x[1,0]],[θ(t)]]),col,0.5)
    plot(M[0,:],M[1,:],'r.',markersize=10)

def draw_particles_weight(P,W,col):
    for i in range(N):
        plot(P[0,i]+0.2*randn(),P[1,i]+0.2*randn(),col,markersize=1+1*N*(W[i]))
    pause(0.1)


N=2000
t=0
dt=0.1
Rx=15
ax=init_figure(-Rx,Rx,-Rx,Rx)
M=array([[3,2,4],[8,6,11]])
m=M.shape[1]
Γα = 0.1*diag([dt,dt])
Γβ = eye(m)
x0=array([[0],[0]])

P=uniform(-Rx,Rx,2*N).reshape(2,N)
W = array([1.0/N]*N)
draw_robot(x0)
print("y= g(x0) = ",g(x0))
draw_particles_weight(P,W,'g.')


pause(1)
