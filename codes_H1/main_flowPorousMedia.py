import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from MeshingFunctions import *
from PlottingFunctions import *
from ReferenceElement import * 
from SystemComputation import *

do_plot = 1

# Reference element
degree = 2
eltType = 0 # 0: quadrilaterals, 1 = triangles
refElement = defineRefElement(eltType,degree)

# Computational mesh
fineness = 2 #from 0 to 3
wallDistance = 10 # 5 or 10
X,T = ExcavationMeshes(degree,eltType,fineness,wallDistance)
if do_plot == 1:
    plotMesh(X,T,refElement); plt.show()
    
#%%

K1 = 1e-6
K2 = 1e-7
Kmat = np.diag([K1, K2])

# Ensamblaje del sistema
[K, f] = computeSystemLaplace2(X, T, refElement, Kmat)

# Visualización de la matriz K
plt.spy(K)
plt.show()

# Tolerancia para identificar los nodos en las fronteras
tol = 1e-8

x1 = 0
x2 = 1.5
y = 1.5

# Frontera superior (nivel piezométrico fijo u(x,y) = y)
nodes_Top = np.where(abs(X[:,1] - np.max(X[:,1])) < tol)[0]  # Parte superior
# Frontera inferior entre x1 y x2, con y fijo en 1.5
nodes_Bottom = np.where((abs(X[:, 1] - y) < tol) & (X[:, 0] >= x1) & (X[:, 0] <= x2))[0]

# Las condiciones de Dirichlet se aplican solo en las fronteras superior e inferior
nodesDir = np.unique(np.block([nodes_Top, nodes_Bottom]))

valDir = X[nodesDir, 1]  # Usamos la coordenada y como valor de Dirichlet

# Aseguramos que valDir tenga la forma correcta de vector columna
valDir = valDir.reshape(-1, 1)
f = f*0
# Reducción del sistema y solución
u = findSolution_SystemReduction(X, T, K, f, nodesDir, valDir)

# Visualización de los resultados
nOfNodes = np.shape(X)[0]
if do_plot == 1:
    # Contour plot
    contourPlot(u, X, T)
    plt.show()

    # Surface plot
    surfPlot(u, X, T)
    plt.show()
    
#%%
# LAGRANGE MULTPLIERS

import numpy as np
import matplotlib.pyplot as plt
from MeshingFunctions import *
from PlottingFunctions import *
from ReferenceElement import * 
from SystemComputation import * 

do_plot = 1

# Reference element
degree = 2
eltType = 0  # 0: quadrilaterals, 1 = triangles
refElement = defineRefElement(eltType, degree)

# Computational mesh
fineness = 2  # from 0 to 3
wallDistance = 10  # 5 or 10
X, T = ExcavationMeshes(degree, eltType, fineness, wallDistance)
if do_plot == 1:
    plotMesh(X, T, refElement)
    plt.show()

# Material properties
K1 = 1e-6
K2 = 1e-7
Kmat = np.diag([K1, K2])

# System assembly
K, f = computeSystemLaplace2(X, T, refElement, Kmat)

# Visualize the matrix K
plt.spy(K)
plt.show()

# Tolerance to identify boundary nodes
tol = 1e-8

x1 = 0
x2 = 1.5
y = 1.5

# Top boundary (fixed piezometric level u(x,y) = y)
nodes_Top = np.where(abs(X[:, 1] - np.max(X[:, 1])) < tol)[0]  # Top part
# Bottom boundary between x1 and x2, with y fixed at 1.5
nodes_Bottom = np.where((abs(X[:, 1] - y) < tol) & (X[:, 0] >= x1) & (X[:, 0] <= x2))[0]

# Dirichlet boundary nodes
nodesDir = np.unique(np.block([nodes_Top, nodes_Bottom]))

# Dirichlet boundary values (y-coordinates)
valDir = X[nodesDir, 1].reshape(-1, 1)

# Assemble matrix A for Lagrange multipliers
A = np.zeros((len(nodesDir), K.shape[0]))
for i, node in enumerate(nodesDir):
    # Nodos valen 1 en las Dirichlet boundary conditions
    A[i, node] = 1.0  

# Build the augmented system
# k_aug nos sirve luego para resolver nuestro sistema
zero_block = np.zeros((len(nodesDir), len(nodesDir)))
K_aug = np.block([[K, A.T],
                  [A, zero_block]])


b = valDir

f_aug = np.vstack([f, b])


sol_aug = np.linalg.solve(K_aug, f_aug)

# Solutins 
u = sol_aug[:K.shape[0]]
lambda_vals = sol_aug[K.shape[0]:]

# Visualization of the results
if do_plot == 1:
    # Contour plot
    contourPlot(u, X, T)
    plt.show()

    # Surface plot
    surfPlot(u, X, T)
    plt.show()
#%%
# Definir el vector lambda_vals y los nodos que pertenecen a nodes_Bottom
lambda_vals = sol_aug[K.shape[0]:]

# Filtrar los valores de lambda_vals que corresponden a los nodes_Bottom
lambda_bottom_vals = lambda_vals[np.isin(nodesDir, nodes_Bottom)]

# Calcular el sumatorio de los valores de Lagrange en los nodos de la frontera inferior
sumatorio_bottom = np.sum(lambda_bottom_vals)

# Imprimir el resultado
print("Sumatorio de los valores de Lagrange en los nodos de la frontera inferior:", sumatorio_bottom)


#%%
import numpy as np
import matplotlib.pyplot as plt

# Define an array of mesh sizes (fineness levels)
mesh_sizes = [0, 1, 2, 3]  # Example mesh sizes (adjust as needed)
flows= []
flow_values = []

for fineness in mesh_sizes:
    wallDistance = 10  # 5 or 10
    X, T = ExcavationMeshes(degree, eltType, fineness, wallDistance)
    if do_plot == 1:
        plotMesh(X, T, refElement)
        plt.show()
    
    # Material properties
    K1 = 1e-6
    K2 = 1e-7
    Kmat = np.diag([K1, K2])
    
    # System assembly
    K, f = computeSystemLaplace2(X, T, refElement, Kmat)
    
    # Visualize the matrix K
    plt.spy(K)
    plt.show()
    
    # Tolerance to identify boundary nodes
    tol = 1e-8
    
    x1 = 0
    x2 = 1.5
    y = 1.5
    
    # Top boundary (fixed piezometric level u(x,y) = y)
    nodes_Top = np.where(abs(X[:, 1] - np.max(X[:, 1])) < tol)[0]  # Top part
    # Bottom boundary between x1 and x2, with y fixed at 1.5
    nodes_Bottom = np.where((abs(X[:, 1] - y) < tol) & (X[:, 0] >= x1) & (X[:, 0] <= x2))[0]
    
    # Dirichlet boundary nodes
    nodesDir = np.unique(np.block([nodes_Top, nodes_Bottom]))
    
    # Dirichlet boundary values (y-coordinates)
    valDir = X[nodesDir, 1].reshape(-1, 1)
    
    # Assemble matrix A for Lagrange multipliers
    A = np.zeros((len(nodesDir), K.shape[0]))
    for i, node in enumerate(nodesDir):
        # Nodos valen 1 en las Dirichlet boundary conditions
        A[i, node] = 1.0  
    
    # Build the augmented system
    # k_aug nos sirve luego para resolver nuestro sistema
    zero_block = np.zeros((len(nodesDir), len(nodesDir)))
    K_aug = np.block([[K, A.T],
                      [A, zero_block]])
    
    
    b = valDir
    
    f_aug = np.vstack([f, b])
    
    
    sol_aug = np.linalg.solve(K_aug, f_aug)
    
    # Solutions 
    u = sol_aug[:K.shape[0]]
    lambda_vals = sol_aug[K.shape[0]:]
    # Filtrar los valores de lambda_vals que corresponden a los nodes_Bottom
    lambda_bottom_vals = lambda_vals[np.isin(nodesDir, nodes_Bottom)]

    # Calcular el sumatorio de los valores de Lagrange en los nodos de la frontera inferior
    sumatorio_bottom = np.sum(lambda_bottom_vals)
    flows.append(sumatorio_bottom)

# Plot the flow as a function of mesh size
plt.figure()
plt.plot(mesh_sizes, flows, marker='o')
plt.xlabel('Mesh Size (Fineness Level)')
plt.ylabel('Flow Value')
plt.title('Influence of Mesh Size on Flow Value')
plt.grid(True)
plt.show()



