import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as mtri

def plotMesh(X,T,refElement):
    elementType = refElement.type
    # degree = refElement.degree
    if elementType == 0:
        plotOrder = [0,1,2,3,0]
    elif elementType == 1:
        plotOrder = [0,1,2,0]   
    for e in np.arange(T.shape[0]):
        Te = T[e,:]
        Xe = X[Te,:]
        plt.plot(Xe[plotOrder,0], Xe[plotOrder,1], 'k')
    plt.plot(X[:,0], X[:,1],'*')    

def plotSlopes(x, y):
    for i in np.arange(len(x) - 1):
        dx = (x[i+1] - x[i])
        dy = (y[i+1] - y[i])
        slope = (dy / dx ).item() # dx and dy are scalars, so slope will be a scalar
        xm = (x[i+1] + x[i]) / 2
        ym = (y[i+1] + y[i]) / 2
        plt.text(xm + dx * 0, ym + dy * 0.2, round(slope, 2))


def contourPlot(u,X,T):
    eltType = np.shape(T)[1]
    nnodes = np.shape(X)[0]
    nElts  = np.shape(T)[0]
    # Solution plot
    x = X[:,0] 
    y = X[:,1]
    u = u[:,0]
    # Contour plot
    if eltType == 4:
        Ttriangles = np.concatenate((T[:,[3,0,1]],T[:,[3,1,2]]))
    elif eltType == 9:
         Ttriangles = np.concatenate((T[:,[6,7,8]],T[:,[6,8,5]],T[:,[5,8,3]],T[:,[5,3,4]],T[:,[7,0,1]],T[:,[7,1,8]],T[:,[8,1,2]],T[:,[8,2,3]]))
    else:
        Ttriangles = T
    plt.tricontourf(x,y,u,30,triangles=Ttriangles,cmap = cm.coolwarm, vmin=u.min(), vmax=u.max(), origin='lower',
          extent=[x.min(), x.max(), y.min(), y.max()])
    plt.xlabel("x")
    plt.ylabel("y")        
    plt.colorbar()

def surfPlot(u,X,T):
    eltType = np.shape(T)[1]
    nnodes = np.shape(X)[0]
    nElts  = np.shape(T)[0]
    # Solution plot
    x = X[:,0]
    y = X[:,1]
    u = u[:,0]
    if eltType==4:
        Ttriangles = np.concatenate((T[:,[3,0,1]],T[:,[3,1,2]]))
    elif eltType == 9:
         Ttriangles = np.concatenate((T[:,[6,7,8]],T[:,[6,8,5]],T[:,[5,8,3]],T[:,[5,3,4]],T[:,[7,0,1]],T[:,[7,1,8]],T[:,[8,1,2]],T[:,[8,2,3]]))
    else:
        Ttriangles = T
    # Surface plot
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    surf = ax.plot_trisurf(x,y,u,triangles = Ttriangles,cmap=cm.coolwarm)
    fig.colorbar(surf, shrink=0.5, aspect=5) # Add a color bar which maps values to colors.