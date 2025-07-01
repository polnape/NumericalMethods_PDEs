import numpy as np

def exactSol(X):
    x = X[:,0]
    y = X[:,1]
    h1 = (x**2)*((1-x)**2)*np.exp(10*x)  
    h2 = (y**2)*((1-y)**2)*np.exp(10*y) 
    u = (1/2000)*h1*h2
    dh1 = (4*x**2+10*x**4-16*x**3+2*x)*np.exp(10*x)
    dh2 = (4*y**2+10*y**4-16*y**3+2*y)*np.exp(10*y)
    u_x = (1/2000)*h2*dh1
    u_y = (1/2000)*h1*dh2
    return u,u_x,u_y

def sourceTerm(X):
    x = X[0]
    y = X[1]
    ddh1 = (2+28*x-8*x**2-120*x**3+100*x**4)*np.exp(10*x)
    ddh2 = (2+28*y-8*y**2-120*y**3+100*y**4)*np.exp(10*y)
    h1 = (x**2)*((1-x)**2)*np.exp(10*x)
    h2 = (y**2)*((1-y)**2)*np.exp(10*y)
    f = -(1/2000)*(ddh1*h2+h1*ddh2)
    return f
