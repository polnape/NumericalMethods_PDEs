import numpy as np
from scipy.sparse import diags
from analyticalInfoStokes import * 
from PlottingFunctions import *
from ReferenceElement import *
from ErrorComputation import *
from MeshingFunctions import *
from SystemComputation import *
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
# This library enhances some plot options
from mpl_toolkits.mplot3d import Axes3D
# Plots are shown on a new window, where we can move and zoom
#matplotlib auto  

do_plot = 1

# Problem definition 
domain = np.array([0,1,0,1])
viscosity=0.0004
rey = 1/viscosity
print(rey)

# Stokes reference element and computational mesh
referenceElementStokes=defineRefElementStokesQ2Q1()
nx = 50; ny = nx; print('Number of elements',np.array([nx,ny]))
X,T,Xp,Tp = UniformRectangleMeshStokesQ1Q2(domain,nx,ny)
if do_plot == 1:
    plotMeshStokesQ2Q1(X,T,Xp,Tp,referenceElementStokes); plt.show()

# FE system assembly
[K,f,Gt] = computeSystemStokes(X,T,Xp,Tp,referenceElementStokes)
K=viscosity*K
nOfNodesP=Xp.shape[0]
A=np.block([[K,Gt],[np.transpose(Gt),np.zeros((nOfNodesP,nOfNodesP))]])
b=np.vstack((f,np.zeros((nOfNodesP,1))))
plt.spy(A); plt.show()

# Dirichlet boundary conditions
x1 = domain[0]; x2 = domain[1]; y1 = domain[2]; y2 = domain[3]
tol = 1e-8
nodes_Left = np.where(abs(X[:,0]-x1)<tol)[0]
nodes_Right = np.where(abs(X[:,0]-x2)<tol)[0]
nodes_Bottom = np.where(abs(X[:,1]-y1)<tol)[0]
nodes_Top = np.where(abs(X[:,1]-y2)<tol)[0]
nodesDir = np.unique(np.block([nodes_Left,nodes_Right,nodes_Bottom,nodes_Top]))
nOfNodes=X.shape[0]
dofDir=np.hstack((nodesDir,nodesDir+nOfNodes,nodesDir[0]+2*nOfNodes)) #velocity at all boundary, pressure at 1st node of the boundary
uDx,uDy,pD = exactSolStokes(X[nodesDir,:])
valDir=np.hstack((uDx,uDy,pD[0])); valDir.shape = (len(valDir),1)

# System reduction and solution 
sol = findSolution_SystemReduction(A,b,dofDir,valDir)
ux=sol[:nOfNodes]; uy=sol[nOfNodes:2*nOfNodes]; p=sol[2*nOfNodes:]
    
#Plots
if do_plot==1:
    uxex,uyex,pex = exactSolStokes(X)
    contourfPlot(ux,X,T)
    plt.title('x-velocity - FEM')
    plt.show()
    contourfPlot(uxex.reshape(nOfNodes,1),X,T)
    plt.title('x-velocity - analytical')
    plt.show()
    contourfPlot(uy,X,T)
    plt.title('y-velocity - FEM')
    plt.show()
    contourfPlot(uyex.reshape(nOfNodes,1),X,T)
    plt.title('y-velocity - analytical')
    plt.show()
    contourfPlot(p,Xp,Tp)
    plt.title('pressure - FEM')
    plt.show()
    contourfPlot(pex.reshape(nOfNodes,1),X,T)
    plt.title('pressure - analytical')
    plt.show()    
#%%
# L2 error computation
L2ErrorU,L2ErrorP = computeL2ErrorStokes(sol,X,T,Xp,Tp,referenceElementStokes)
print('L2 error velocity: ', L2ErrorU)
print('L2 error pressure: ', L2ErrorP)

#Computation of the stream function and plot of stream lines
Tboundary=boundaryConnectivityQ2Q1(X,T,Xp,Tp)
phi = computeStreamFunction(ux,uy,X,T,Tboundary,referenceElementStokes)

if do_plot==1:
    plt.quiver(X[:,0],X[:,1],ux,uy)
    plt.title('velocity - FEM')
    plt.show()
    contourPlot(phi,X,T)
    plt.title('Stream lines')
    plt.show()
#%%
import numpy as np

