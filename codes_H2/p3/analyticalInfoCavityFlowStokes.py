import numpy as np

def exactSolStokes(X):
    x = X[:,0]
    y = X[:,1]
    ux=x*0
    uy=x*0
    p=x*0
    i=np.where(abs(y-1)<1.e-6)[0]
    ux[i]=1
    return ux,uy,p

def bodyForceStokes(X):
    x = X[0]
    y = X[1]
    f = np.array([0,0])
    return f
