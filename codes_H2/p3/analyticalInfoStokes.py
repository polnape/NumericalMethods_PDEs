import numpy as np

def exactSolStokes(X):
    x = X[:,0]
    y = X[:,1]
    ux=y*np.sin(x)
    uy=-0.5*np.cos(x)*(y**2)
    p=np.sin(x)
    return ux,uy,p

def bodyForceStokes(X):
    x = X[0]
    y = X[1]
    f = np.array([2*y*np.sin(x)+np.cos(x),-np.cos(x)*(y**2)+2*np.cos(x)])
    return f
