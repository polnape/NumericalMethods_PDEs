# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:04:39 2024

@author: polop
"""

import numpy as np
from scipy.sparse import diags
from AnalyticallInfo import * 
from  PlottingFunctions import *
from ReferenceElement import *

from MeshingFunctions import *
from SystemComputation import *
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
# This library enhances some plot options
from mpl_toolkits.mplot3d import Axes3D

from scipy.sparse.linalg import eigs
# Plots are shown on a new window, where we can move and zoom
#matplotlib auto  


#%%
import numpy as np
import matplotlib.pyplot as plt

# Define the function g(t)
def g(t):
    return -300 / (t + 3) + 116
def dg(t):
    return +300/(t+3)**2

# Define the range of t values
t_values = np.linspace(0, 50, 500)
g_values = g(t_values)

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(t_values, g_values, label=r"$g(t) = -\frac{300}{t+3} + 116$", linewidth=2)
plt.title("Wall Temperature Over Time", fontsize=14)
plt.xlabel("Time (t)", fontsize=12)
plt.ylabel("Temperature (°C)", fontsize=12)
plt.axhline(116, color="red", linestyle="--", label="Initial Wall Temperature (116°C)")
plt.grid(alpha=0.5)
plt.legend(fontsize=12)
plt.show()


#%%
do_plot = 1
nu = 1
# Problem definition 
domain = np.array([0,5,0,5])

# Reference element and computational mesh
elementType = 0# 1 = Triangles, 0 = Quadrilaterals
degree =1
referenceElement = defineRefElement(elementType, degree)

# Creation of the mesh
h = 1 / 2

nx = int((domain[1] - domain[0]) / h)  # Number of elements in x-direction
ny = int((domain[3] - domain[2]) / h)  # Number of elements in y-direction
print('Number of elements:', np.array([nx, ny]))
nOfElements = nx*ny

X,T = rectangleMesh(domain,nx,ny,referenceElement)
if do_plot == 1:
    plotMesh(X,T,referenceElement); plt.show()
# print(f"T:{T}")
# print(f"X:{X}")
nOfNodes = X.shape[0]


# Paso 1: Identificar los nodos en la frontera para aplicar condiciones de Dirichlet
def setBoundaryConditions(X):
    nOfNodes = X.shape[0]
    nodesDir = []
    for i in range(nOfNodes):
        x, y = X[i, :]
        # Condiciones de frontera (x=0, x=5, y=0, y=5)
        if x == 0 or x == 5 or y == 0 or y == 5:
            nodesDir.append(i)
    nodesDir = np.array(nodesDir)
    nodesUnk = np.setdiff1d(np.arange(nOfNodes), nodesDir)

    # Inicializar valores de Dirichlet
    valDir = np.zeros((len(nodesDir), 1))
    dtvalDir = np.zeros((len(nodesDir), 1))
    return nodesDir, nodesUnk, valDir, dtvalDir

# Identificar los nodos de Dirichlet y desconocidos
nodesDir, nodesUnk, valDir, dtvalDir = setBoundaryConditions(X)



# Paso 2: Aplicar condiciones de contorno en la simulación
def applyBoundaryConditions(U, nodesDir, valDir, t):
    # Actualizar los valores de Dirichlet según la función g(t)
    valDir[:] = g(t)
    U[nodesDir] = valDir
    return U



K, M, MLumped, f = computeSystemHeatEquation(X, T, referenceElement)



Lumping = 0
K_red = K[nodesUnk,:][:,nodesUnk]
if Lumping==1:
    M_red = MLumped[nodesUnk,:][:,nodesUnk]
    f_red = f[nodesUnk,:]-K[nodesUnk,:][:,nodesDir]@valDir-MLumped[nodesUnk,:][:,nodesDir]@dtvalDir
elif Lumping == 0:
    M_red = M[nodesUnk,:][:,nodesUnk]
    f_red = f[nodesUnk,:]-K[nodesUnk,:][:,nodesDir]@valDir-M[nodesUnk,:][:,nodesDir]@dtvalDir

## Euler explicit time intergration
# Time steps
Tfinal = 3
nOfElements = T.shape[0]
if Lumping == 0:
    timeStep = 1/(6*nu*(nOfElements**2+2))
elif Lumping == 1:
    timeStep = 1/(2*nu*(nOfElements**2+2))
timeIterationsExp = round(Tfinal/timeStep+1); timeStep = Tfinal/timeIterationsExp
timeVec = np.linspace(0,Tfinal,timeIterationsExp); 
print('Time steps',timeIterationsExp)
if Lumping==1:
    print('Stability condition', nu*timeStep*nOfElements**2,'<=',1/2)
elif Lumping==0:
    print('Stability condition', nu*timeStep*nOfElements**2,'<=',1/6)
    

    

MLumped_red = MLumped[nodesUnk, :][:, nodesUnk]



# Calcular los valores propios de -M^-1 K para las matrices consistente y lumped
eig_vals_consistent, _ = eigs(-np.linalg.inv(M_red) @ K_red, k=6, which='LM')
eig_vals_lumped, _ = eigs(-np.linalg.inv(MLumped_red) @ K_red, k=6, which='LM')

# Extraer el valor propio máximo (en magnitud)
lambda_max_consistent = max(np.abs(eig_vals_consistent))
lambda_max_lumped = max(np.abs(eig_vals_lumped))

# Calcular el umbral de \Delta T
delta_t_consistent = 2 / lambda_max_consistent
delta_t_lumped = 2 / lambda_max_lumped

print("Máximo valor propio (Consistente):", lambda_max_consistent)
print("Máximo valor propio (Lumped):", lambda_max_lumped)
print("Delta T (Consistente):", delta_t_consistent)
print("Delta T (Lumped):", delta_t_lumped)


# EULER EXPLICIT TIME INTEGRATION
timeStep = 0.001

Tfinal = 10
timeIterationsExp = round(Tfinal/timeStep+1);
timeVec = np.linspace(0,Tfinal, timeIterationsExp);
print('Time steps', timeIterationsExp)

# def initial_Cond(X):
#     # Devuelve un array   lleno de 16 (temperatura inicial) para cada nodo
#     return np.full(X.shape[0], 16.0)
# # Condición inicial
# inCond = initial_Cond(X); inCond.shape = (nOfNodes)
# MINIMIZO LA DIFERENCIA ENTRE CADA NODO Y EL PUNTO A BUSCAR
aux = X - [2.5, 2.5]
idx = (np.sum(np.square(aux), axis=1)).argmin()
u_SolExp = np.zeros((nOfNodes,timeIterationsExp))
u_SolExp[:,0] = 16

Lumping = 1
for k in np.arange(timeIterationsExp - 1):
    # System reduction
    # print(k)
    valDir = np.zeros((len(nodesDir), 1)) + g(timeVec[k])
    dtvalDir = np.zeros((len(nodesDir), 1)) + dg(timeVec[k])
    K_red = K[nodesUnk, :][:, nodesUnk]
    if Lumping == 1:
        # dtvalDir is used when our dirichlet boundary conditions change over time
        M_red = MLumped[nodesUnk, :][:, nodesUnk]
        # it is like if we had to f, the one of the dirichlet, with some wind of source, and the one on the inside
        f_red = f[nodesUnk, :] - K[nodesUnk, :][:, nodesDir] @ valDir - MLumped[nodesUnk, :][:, nodesDir] @ dtvalDir
    elif Lumping == 0:
        M_red = M[nodesUnk, :][:, nodesUnk]
        f_red = f[nodesUnk, :] - K[nodesUnk, :][:, nodesDir] @ valDir - M[nodesUnk, :][:, nodesDir] @ dtvalDir

    # Compute right-hand side
    rhs = M_red @ u_SolExp[nodesUnk, k] + timeStep * (f_red.flatten() - K_red @ u_SolExp[nodesUnk, k])

    # Solve for internal nodes
    u_SolExp[nodesUnk, k + 1] = np.linalg.solve(M_red, rhs)

    # Update boundary nodes with Dirichlet condition
    u_SolExp[nodesDir, k + 1] = valDir.flatten()

    # print(k, timeIterationsExp, timeVec[k], u_SolExp[idx,k])



boundary_node_index = nodesDir[0]
aux = X - [2.5, 2.5]
idx_interior = np.sum(np.square(aux), axis=1).argmin()
plt.figure(figsize=(10, 6))
plt.plot(timeVec, u_SolExp[idx_interior, :], label=f"Interior Node (2.5, 2.5)",linestyle = "--")
plt.plot(timeVec, u_SolExp[boundary_node_index, :], label=f"Boundary Node", linestyle = "--")
plt.title("Temperature Evolution", fontsize=14)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("Temperature (°C)", fontsize=16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend()
plt.show()

#%%
from PlottingFunctions import surfPlot  # Importar la función desde tu archivo

# Especifica el índice de tiempo deseado
time_index = 100  # Cambia este valor al índice de tiempo que quieras graficar

# Verifica que el índice no exceda el tamaño de u_SolImp
if time_index < u_SolImp.shape[1]:
    # Llamar a la función para graficar la distribución de temperatura en ese tiempo
    surfPlot(u_SolImp[:, time_index].reshape(-1, 1), X, T)
else:
    print(f"Error: El índice de tiempo {time_index} está fuera del rango de datos disponibles.")
#%%
import imageio.v2 as imageio
from PlottingFunctions import surfPlot
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Directory to save frames
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

# Number of frames (time points) to include in the GIF
n_frames = 30

# Select 10 evenly spaced time indices
time_indices = np.linspace(0, u_SolImp.shape[1] - 1, n_frames).astype(int)

# Define a fixed z-axis range
z_min = np.min(u_SolImp)
z_max = np.max(u_SolImp)

# Create frames for each time index
filenames = []
for i, time_index in enumerate(time_indices):
    # Set up the figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract the temperature at the current time step
    temperatures = u_SolImp[:, time_index].reshape(-1, 1)

    # Plot the surface
    surfPlot(temperatures, X, T)

    # Adjust the view to make the figure move
    ax.view_init(elev=20 + i * 5, azim=30 + i * 10)  # Change elev and azim dynamically
    ax.set_zlim(z_min, z_max)  # Ensure consistent z-axis scaling

    # Save the frame
    frame_filename = os.path.join(output_dir, f"frame_{i:03d}.png")
    plt.savefig(frame_filename)
    plt.close()
    filenames.append(frame_filename)

# Create GIF with figure movement
gif_filename = "temperature_evolution_moving_figure.gif"
with imageio.get_writer(gif_filename, mode="I", duration=1.0) as writer:  # 1.0 second per frame
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Clean up (optional): Remove the individual frame files
for filename in filenames:
    os.remove(filename)

print(f"GIF with moving figure saved as {gif_filename}")

#%%
# EULER IMPLICIT
Tfinal = 10
timeStep = 0.001
timeIterationsImp = round(Tfinal / timeStep + 1)
timeStep = Tfinal / timeIterationsImp
timeVec = np.linspace(0, Tfinal, timeIterationsImp)

# Find the index of the node closest to (2.5, 2.5)
aux = X - [2.5, 2.5]
idx_interior = (np.sum(np.square(aux), axis=1)).argmin()

# Initialize the solution matrix
u_SolImp = np.zeros((nOfNodes, timeIterationsImp))
u_SolImp[:, 0] = 16  # Initial temperature for all nodes
Lumping = 0

# Time loop for implicit Euler
for k in range(timeIterationsImp - 1):
    # Update time-dependent Dirichlet conditions
    K_red = K[nodesUnk, :][:, nodesUnk]
    valDir = np.zeros((len(nodesDir), 1)) + g(timeVec[k] + timeStep)
    dtvalDir = np.zeros((len(nodesDir), 1)) + dg(timeVec[k] + timeStep)

    if Lumping == 0:
        M_red = M[nodesUnk, :][:, nodesUnk]
        f_red = f[nodesUnk, :] - K[nodesUnk, :][:, nodesDir] @ valDir - M[nodesUnk, :][:, nodesDir] @ dtvalDir

    # Compute right-hand side for implicit method
    rhs = M_red @ u_SolImp[nodesUnk, k] + timeStep * f_red.flatten()

    # Solve implicit equation: (M + Δt * K) u^{n+1} = rhs
    A = M_red + timeStep * K_red
    u_SolImp[nodesUnk, k + 1] = np.linalg.solve(A, rhs)

    # Update boundary nodes
    u_SolImp[nodesDir, k + 1] = valDir.flatten()

    # Stop when temperature at (2.5, 2.5) reaches 85°C
    if u_SolImp[idx_interior, k + 1] >= 85:
        print(f"Temperature at node (2.5, 2.5) reached 85°C at time {timeVec[k + 1]:.6f}s.")
        break

# Trim data to the stopping point
timeVec = timeVec[:k+2]
u_SolImp = u_SolImp[:, :k+2]

# First Plot: Temperature evolution at node (2.5, 2.5)
plt.figure(figsize=(10, 6))
plt.plot(timeVec, u_SolImp[idx_interior, :], label="Node (2.5, 2.5)", linestyle="--", linewidth=2)
plt.title("Temperature Evolution at Node (2.5, 2.5)", fontsize=14)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("Temperature (°C)", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# Second Plot: Temperature evolution with horizontal line
boundary_node_index = nodesDir[0]  # Select a boundary node

plt.figure(figsize=(10, 6))
plt.plot(timeVec, u_SolImp[idx_interior, :], label="Node (2.5, 2.5)", linestyle="--", linewidth=2)
plt.plot(timeVec, u_SolImp[boundary_node_index, :], label="Boundary Node", linestyle="--", linewidth=2)
plt.axhline(y=85, color="red", linestyle=":", label="85°C Threshold")
plt.title("Temperature Evolution with Threshold", fontsize=14)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("Temperature (°C)", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()



#%%

# EULER IMPLICIT
Tfinal = 10
timeStep = 0.01
timeIterationsImp = round(Tfinal / timeStep + 1)
timeStep = Tfinal / timeIterationsImp
timeVec = np.linspace(0, Tfinal, timeIterationsImp)

# Find the index of the node closest to (2.5, 2.5)
aux = X - [2.5, 2.5]
idx_interior = (np.sum(np.square(aux), axis=1)).argmin()

# Initialize the solution matrix
u_SolImp = np.zeros((nOfNodes, timeIterationsImp))
u_SolImp[:, 0] = 16  # Initial temperature for all nodes
Lumping = 0

# Time loop for implicit Euler
for k in range(timeIterationsImp - 1):
    # Update time-dependent Dirichlet conditions
    K_red = K[nodesUnk, :][:, nodesUnk]
    valDir = np.zeros((len(nodesDir), 1)) + g(timeVec[k] + timeStep)
    dtvalDir = np.zeros((len(nodesDir), 1)) + dg(timeVec[k] + timeStep)

    if Lumping == 0:
        M_red = M[nodesUnk, :][:, nodesUnk]
        f_red = f[nodesUnk, :] - K[nodesUnk, :][:, nodesDir] @ valDir - M[nodesUnk, :][:, nodesDir] @ dtvalDir

    # Compute right-hand side for implicit method
    rhs = M_red @ u_SolImp[nodesUnk, k] + timeStep * f_red.flatten()

    # Solve implicit equation: (M + Δt * K) u^{n+1} = rhs
    A = M_red + timeStep * K_red
    u_SolImp[nodesUnk, k + 1] = np.linalg.solve(A, rhs)

    # Update boundary nodes
    u_SolImp[nodesDir, k + 1] = valDir.flatten()

    # Check if all nodes reach 85°C
    min_temp = np.min(u_SolImp[:, k + 1])
    if min_temp >= 85:
        time_reach_85 = timeVec[k + 1]
        print(f"All nodes reached 85°C at time {time_reach_85:.8f}s.")
        break

# Trim data to the stopping point
timeVec = timeVec[:k+2]
u_SolImp = u_SolImp[:, :k+2]

# Plot: Temperature evolution at node (2.5, 2.5)
plt.figure(figsize=(10, 6))
plt.plot(timeVec, u_SolImp[idx_interior, :], label="Node (2.5, 2.5)", linestyle="--", linewidth=2)
plt.title("Temperature Evolution at Node (2.5, 2.5)", fontsize=14)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("Temperature (°C)", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# Plot: Temperature evolution for node (2.5, 2.5) and boundary node with threshold
boundary_node_index = nodesDir[0]  # Select a boundary node

plt.figure(figsize=(10, 6))
plt.plot(timeVec, u_SolImp[idx_interior, :], label="Node (2.5, 2.5)", linestyle="--", linewidth=2)
plt.plot(timeVec, u_SolImp[boundary_node_index, :], label="Boundary Node", linestyle="--", linewidth=2)
plt.axhline(y=85, color="red", linestyle=":", label="85°C Threshold")
plt.title("Temperature Evolution with Threshold", fontsize=14)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("Temperature (°C)", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()





