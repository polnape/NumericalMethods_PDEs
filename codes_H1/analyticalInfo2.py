import numpy as np

def sourceTerm(pt):
    pi = np.pi
    x = pt[0]
    y = pt[1]
    s = np.sin(pi*x)
    c = np.cos(pi*y)
    return 2*pi**2*s*c

def exactSol(pt):
    pi = np.pi
    x = pt[:,0]
    y = pt[:,1]
    sx = np.sin(pi*x)
    sy = np.sin(pi*y)
    cx = np.cos(pi*x)
    cy = np.cos(pi*y)
    u = sx*cy
    u_x = pi*cx*cy
    u_y = -pi*sx*sy
    return u, u_x, u_y
