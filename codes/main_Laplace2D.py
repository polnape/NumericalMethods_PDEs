import numpy as np
from scipy.sparse import diags
from analyticalInfo import * 
from  PlottingFunctions import *
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

# Reference element and computational mesh
elementType = 0# 1 = Triangles, 0 = Quadrilaterals
degree =2
referenceElement = defineRefElement(elementType, degree)

# Creation of the mesh
h = 1 / 2

nx = int((domain[1] - domain[0]) / h)  # Number of elements in x-direction
ny = int((domain[3] - domain[2]) / h)  # Number of elements in y-direction
print('Number of elements:', np.array([nx, ny]))

X,T = rectangleMesh(domain,nx,ny,referenceElement)
if do_plot == 1:
    plotMesh(X,T,referenceElement); plt.show()
print(T)
# FE system assembly
# MODIFIED FOR THE SECOND PROBLEM, WITH A MATRIX K
[K,f] = computeSystemLaplace(X,T,referenceElement)
plt.spy(K); plt.show()

# Dirichlet boundary conditions
x1 = domain[0]; x2 = domain[1]
y1 = domain[2]; y2 = domain[3]
tol = 1e-8
nodes_Left = np.where(abs(X[:,0]-x1)<tol)[0]
nodes_Right = np.where(abs(X[:,0]-x2)<tol)[0]
nodes_Bottom = np.where(abs(X[:,1]-y1)<tol)[0]
nodes_Top = np.where(abs(X[:,1]-y2)<tol)[0]
nodesDir = np.unique(np.block([nodes_Left,nodes_Right,nodes_Bottom,nodes_Top]))
valDir = exactSol(X[nodesDir,:])[0];valDir.shape = (len(nodesDir),1)

# System reduction and solution 
u = findSolution_SystemReduction(X,T,K,f,nodesDir,valDir)

# L2 error computation
L2Error = computeL2Error(u,X,T,referenceElement)
print('L2 error: ', L2Error)
H1Error = computeH1Error(u, X, T, referenceElement)
print("H1 error:", H1Error)
    

nOfNodes = np.shape(X)[0]
if do_plot==1:
    # Contour plot
    contourPlot(u,X,T) 
    plt.show()
    u_analytic = exactSol(X)[0].reshape(nOfNodes,1)
    contourPlot(u_analytic,X,T)
    plt.show()
    # Surface plot
    surfPlot(u,X,T)
    plt.show()
    surfPlot(u_analytic,X,T)
    plt.show()
#%%
import numpy as np
erroresL2 = []
erroresH1 = []
lnh_list = []
elementType = 0# 1 = Triangles, 0 = Quadrilaterals
degree =1
referenceElement = defineRefElement(elementType, degree)

for n in range(40, 80, 2):
    h = 1/n
    lnh = np.log10(h)

    lnh_list.append(lnh)




    nx = int((domain[1] - domain[0]) / h)  # Number of elements in x-direction
    ny = int((domain[3] - domain[2]) / h)  # Number of elements in y-direction
    # print('Number of elements:', np.array([nx, ny]))

    X,T = rectangleMesh(domain,nx,ny,referenceElement)
    # if do_plot == 1:
    #     plotMesh(X,T,referenceElement); plt.show()
   
    # FE system assembly
    [K,f] = computeSystemLaplace(X,T,referenceElement)
    # plt.spy(K); plt.show()

    # Dirichlet boundary conditions
    x1 = domain[0]; x2 = domain[1]
    y1 = domain[2]; y2 = domain[3]
    tol = 1e-8
    nodes_Left = np.where(abs(X[:,0]-x1)<tol)[0]
    nodes_Right = np.where(abs(X[:,0]-x2)<tol)[0]
    nodes_Bottom = np.where(abs(X[:,1]-y1)<tol)[0]
    nodes_Top = np.where(abs(X[:,1]-y2)<tol)[0]
    nodesDir = np.unique(np.block([nodes_Left,nodes_Right,nodes_Bottom,nodes_Top]))
    valDir = exactSol(X[nodesDir,:])[0];valDir.shape = (len(nodesDir),1)

    # System reduction and solution 
    u = findSolution_SystemReduction(X,T,K,f,nodesDir,valDir)

    # L2 error computation
    L2Error = computeL2Error(u,X,T,referenceElement)
    erroresL2.append(np.log10(L2Error))
    # print('L2 error: ', L2Error)
    H1Error = computeH1Error(u, X, T, referenceElement)
    erroresH1.append(np.log10(H1Error))
    # print("H1 error:", H1Error)
plt.plot(lnh_list, erroresL2, color="red", marker='o', label='log10(L2) Error')
plt.plot(lnh_list, erroresH1, marker='x', label='log10(H1) Error')
plt.xlabel('log10(h)')
plt.ylabel('log10(Error)')
plt.legend()
# plotSlopes(lnh_list, erroresL2)
# plotSlopes(lnh_list, erroresH1)
plt.grid(True)
plt.show()
#%%
import numpy as np
from sklearn.linear_model import LinearRegression
X_lnh = np.array(lnh_list).reshape(-1, 1)

# Regresión lineal para erroresL2
regresion_L2 = LinearRegression().fit(X_lnh, erroresL2)
pendiente_L2 = regresion_L2.coef_[0]

# Regresión lineal para erroresH1
regresion_H1 = LinearRegression().fit(X_lnh, erroresH1)
pendiente_H1 = regresion_H1.coef_[0]

# Imprimir las pendientes
print(f"La pendiente de la regresión para erroresL2 es: {pendiente_L2}")
print(f"La pendiente de la regresión para erroresH1 es: {pendiente_H1}")
    

