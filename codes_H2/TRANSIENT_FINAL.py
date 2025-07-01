import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from MeshingFunctions import *
from ReferenceElement import *
from AnalyticallInfo import *
from SystemComputation import *

## Finite element method in space
nu = 1    # Thermal conductivity
T = 0.2   # Final time

# Reference element definition
degree=1
refElement=defineRefElement1D(degree)

# Computational mesh
domain = np.array([0,1])
nOfElements=40
X,T=uniform1dMesh(domain[0],domain[1],nOfElements,degree)

# System assembly
K,M,MLumped,f = computeSystem_HeatEqn1D(X,T)

#Homogeneous Dirichlet boundary conditions at both ends
nOfNodes = X.shape[0]; 
nodesDir = np.array([0,nOfNodes-1]); 
nodesUnk = np.setdiff1d(np.arange(nOfNodes),nodesDir)
valDir = np.zeros((len(nodesDir),1))
dtvalDir = np.zeros((len(nodesDir),1))

# System reduction 
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
if Lumping == 0:
    timeStep = 1/(6*nu*(nOfElements**2+2))
elif Lumping == 1:
    timeStep = 1/(2*nu*(nOfElements**2+2))
timeIterationsExp = round(0.2/timeStep+1); timeStep = 0.2/timeIterationsExp
timeVec = np.linspace(0,0.2,timeIterationsExp); 
print('Time steps',timeIterationsExp)
if Lumping==1:
    print('Stability condition', nu*timeStep*nOfElements**2,'<=',1/2)
elif Lumping==0:
    print('Stability condition', nu*timeStep*nOfElements**2,'<=',1/6)

# Allocation of the solution at each time-step
u_SolExp = np.zeros((nOfNodes,timeIterationsExp))
# Initial condition
inCond = initial_Cond(X); inCond.shape = (nOfNodes)


if Lumping==0:
    # Euler implicit
    # Time steps
    timeStep = 0.01; timeIterationsImp = round(0.2/timeStep+1); timeStep = 0.2/timeIterationsImp
    timeVec = np.linspace(0,0.2,timeIterationsImp); 
    print('Time steps',timeIterationsImp)

    # Allocation of the solution at each time-step
    u_SolImp = np.zeros((nOfNodes,timeIterationsImp))
    # Initial condition
    u_SolImp[:,0] = inCond
    for k in np.arange(timeIterationsImp-1):
        tk = timeVec[k]
        tk1 = timeVec[k+1]
        aux1 = M_red@u_SolImp[1:nOfNodes-1,k]; aux1.shape = (nOfNodes-2,1)
        aux = np.linalg.solve(M_red+nu*timeStep*K_red,aux1+timeStep*f_red); aux.shape = nOfNodes-2
        u_SolImp[1:nOfNodes-1,k+1] = aux

    num = 3
    for k in np.arange(round(timeIterationsImp/num)):
        plt.plot(X,u_SolImp[:,num*k],'-o')
    plt.show()

    # Trapezoidal rule
    # Time steps
    timeStep = 0.001; timeIterationsTrap = round(0.2/timeStep+1); timeStep = 0.2/timeIterationsTrap
    timeVec = np.linspace(0,0.2,timeIterationsTrap); 
    print('Time steps',timeIterationsTrap)

    # Allocation of the solution at each time-step
    u_SolTrap = np.zeros((nOfNodes,timeIterationsTrap))
    # Initial condition
    u_SolTrap[:,0] = inCond
    A = M_red-nu*(timeStep/2)*K_red
    B = M_red+nu*(timeStep/2)*K_red
    for k in np.arange(timeIterationsTrap-1):
        tk = timeVec[k]
        tk1 = timeVec[k+1]
        aux1 = A@u_SolTrap[1:nOfNodes-1,k]; aux1.shape = (nOfNodes-2,1)
        aux = np.linalg.solve(B,aux1+timeStep*f_red); aux.shape = nOfNodes-2
        u_SolTrap[1:nOfNodes-1,k+1] = aux

    num = 10
    for k in np.arange(round(timeIterationsTrap/num)):
        plt.plot(X,u_SolTrap[:,num*k],'-o')
    plt.show()
elif Lumping==1:
    print('Error: Implicit methods do not work with lumped mass matrix')



#%%
# Parámetros de tiempo
timeStep = 0.01  # Paso de tiempo
T_max = 4  # Tiempo máximo de simulación
timeIterationsImp = round(T_max / timeStep)
timeVec = np.linspace(0, T_max, timeIterationsImp)
print(f'Número de pasos de tiempo: {timeIterationsImp}')

# Inicialización
U_SolImp = np.zeros((X.shape[0], timeIterationsImp))  # Solución para cada nodo y paso de tiempo
U_SolImp[:, 0] = 16.0  # Condición inicial uniforme: todos los nodos comienzan a 16°C

# Ensamblar la matriz global para el método implícito de Euler
A = M + timeStep * K  # Matriz del sistema (sin lumping)

# Simulación temporal
for k in range(timeIterationsImp - 1):
    tk = timeVec[k]
    tk1 = timeVec[k + 1]

    # Lado derecho del sistema
    b = M @ U_SolImp[:, k] + timeStep * f  # Usar la matriz completa \( M \)

    # Resolver el sistema lineal
    U_SolImp[:, k + 1] = np.linalg.solve(A, b)

# Graficar la evolución de la temperatura en varios pasos
num_steps_to_plot = 3  # Número de instantes a graficar
step_interval = max(1, timeIterationsImp // num_steps_to_plot)

plt.figure()
for k in range(0, timeIterationsImp, step_interval):
    plt.scatter(X[:, 0], X[:, 1], c=U_SolImp[:, k], cmap='coolwarm')
    plt.colorbar(label='Temperatura (°C)')
    plt.title(f'Temperatura en t = {timeVec[k]:.2f}s')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

