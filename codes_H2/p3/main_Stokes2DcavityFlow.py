# import numpy as np
# from scipy.sparse import diags
# from analyticalInfoStokes import * 
# from PlottingFunctions import *
# from ReferenceElement import *
# from ErrorComputation import *
# from MeshingFunctions import *
# from SystemComputation import *
# import matplotlib.pyplot as plt
# import matplotlib.tri as mtri
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# # This library enhances some plot options
# from mpl_toolkits.mplot3d import Axes3D
# # Plots are shown on a new window, where we can move and zoom
# #matplotlib auto  

# do_plot = 1

# # Problem definition 
# domain = np.array([0,1,0,1])
# viscosity=2

# # Stokes reference element and computational mesh
# referenceElementStokes=defineRefElementStokesQ2Q1()
# nx = 16; ny = nx; print('Number of elements',np.array([nx,ny]))
# #X,T,Xp,Tp = UniformRectangleMeshStokesQ1Q2(domain,nx,ny)
# X,T,Xp,Tp = CavityFlowMeshStokesQ1Q2(domain,nx,ny)
# if do_plot == 1:
#     plotMeshStokesQ2Q1(X,T,Xp,Tp,referenceElementStokes); plt.show()

# # FE system assembly
# # [K,f,Gt] = computeSystemStokes(X,T,Xp,Tp,referenceElementStokes)
# # nOfNodesP=Xp.shape[0]
# # A=np.block([[viscosity*K,Gt],[np.transpose(Gt),np.zeros((nOfNodesP,nOfNodesP))]])
# # b=np.vstack((f,np.zeros((nOfNodesP,1))))
# # plt.spy(A); plt.show()

# A,b=computeSystemStokesSparse(viscosity,X,T,Xp,Tp,referenceElementStokes)

# # Dirichlet boundary conditions
# x1 = domain[0]; x2 = domain[1]; y1 = domain[2]; y2 = domain[3]
# tol = 1e-8
# nodes_Left = np.where(abs(X[:,0]-x1)<tol)[0]
# nodes_Right = np.where(abs(X[:,0]-x2)<tol)[0]
# nodes_Bottom = np.where(abs(X[:,1]-y1)<tol)[0]
# nodes_Top = np.where(abs(X[:,1]-y2)<tol)[0]
# nodesDir = np.unique(np.block([nodes_Left,nodes_Right,nodes_Bottom,nodes_Top]))
# nOfNodes=X.shape[0]
# dofDir=np.hstack((nodesDir,nodesDir+nOfNodes,nodesDir[0]+2*nOfNodes)) #velocity at all boundary, pressure at 1st node of the boundary
# uDx,uDy,pD = exactSolStokes(X[nodesDir,:])
# valDir=np.hstack((uDx,uDy,pD[0])); valDir.shape = (len(valDir),1)

# # System reduction and solution 
# #sol = findSolution_SystemReduction(A,b,dofDir,valDir)
# sol = findSolutionSparse_SystemReduction(A,b,dofDir,valDir)
# ux=sol[:nOfNodes]; uy=sol[nOfNodes:2*nOfNodes]; p=sol[2*nOfNodes:]
    
# #Plots
# if do_plot==1:
#     surfPlot(p,Xp,Tp)
#     plt.title('pressure - FEM')
#     plt.show()    
#     plt.quiver(X[:,0],X[:,1],ux,uy)
#     plt.title('velocity - FEM')
#     plt.show()
#     #Computation of the stream function and plot of stream lines
#     Tboundary=boundaryConnectivityQ2Q1(X,T,Xp,Tp)
#     phi = computeStreamFunction(ux,uy,X,T,Tboundary,referenceElementStokes)
#     contourPlot(phi,X,T)
#     plt.title('Stream lines')
#     plt.show()
    
# import os

# # Obtener el directorio de trabajo actual
# current_directory = os.getcwd()

# # Cambiar el directorio de guardado al actual
# filename = os.path.join(current_directory, "sol0p001.npy")
# np.save(filename, sol)
# #%%
# # C=computeCmatrixNavierStokesSparse(sol,X,T,Xp.shape[0],referenceElementStokes)
# # S=A+C

# sol = np.load('sol0p001.npy')
# for iter in np.arange(20):
#     C1 = computeCmatrixNavierStokesSparse(sol, X, T, Xp.shape[0], referenceElementStokes)
#     S = A+C1
#     solNew = findSolutionSparse_SystemReduction(S, b, dofDir, valDir)
#     error = np.linalg.norm(sol[:2*nOfNodes]-solNew[:2*nOfNodes])/np.sqrt(nOfNodes)
#     print("k = ", iter, "error = ", error)
#     sol = solNew
#     if error<1.e-4:
#         break
# ux=sol[:nOfNodes]; uy=sol[nOfNodes:2*nOfNodes]; p=sol[2*nOfNodes:]
# if do_plot==1:
#     surfPlot(p,Xp,Tp)
#     plt.title('pressure - FEM')
#     plt.show()    
#     plt.quiver(X[:,0],X[:,1],ux,uy)
#     plt.title('velocity - FEM')
#     plt.show()



#     #Computation of the stream function and plot of stream lines
#     Tboundary=boundaryConnectivityQ2Q1(X,T,Xp,Tp)
#     phi = computeStreamFunction(ux,uy,X,T,Tboundary,referenceElementStokes)
#     contourPlot(phi,X,T)
#     plt.title('Stream lines')
#     plt.show()
# np.save("sol0p0005", sol)    


#%%



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
viscosity=2

# Stokes reference element and computational mesh
referenceElementStokes=defineRefElementStokesQ2Q1()
nx = 32; ny = nx; print('Number of elements',np.array([nx,ny]))
#X,T,Xp,Tp = UniformRectangleMeshStokesQ1Q2(domain,nx,ny)
X,T,Xp,Tp = CavityFlowMeshStokesQ1Q2(domain,nx,ny)
if do_plot == 1:
    plotMeshStokesQ2Q1(X,T,Xp,Tp,referenceElementStokes); plt.show()

# FE system assembly
# [K,f,Gt] = computeSystemStokes(X,T,Xp,Tp,referenceElementStokes)
# nOfNodesP=Xp.shape[0]
# A=np.block([[viscosity*K,Gt],[np.transpose(Gt),np.zeros((nOfNodesP,nOfNodesP))]])
# b=np.vstack((f,np.zeros((nOfNodesP,1))))
# plt.spy(A); plt.show()

A,b=computeSystemStokesSparse(viscosity,X,T,Xp,Tp,referenceElementStokes)

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
#sol = findSolution_SystemReduction(A,b,dofDir,valDir)
sol = findSolutionSparse_SystemReduction(A,b,dofDir,valDir)
ux=sol[:nOfNodes]; uy=sol[nOfNodes:2*nOfNodes]; p=sol[2*nOfNodes:]
    
#Plots
if do_plot==1:
    surfPlot(p,Xp,Tp)
    plt.title('pressure - FEM')
    plt.show()    
    plt.quiver(X[:,0],X[:,1],ux,uy)
    plt.title('velocity - FEM')
    plt.show()
    #Computation of the stream function and plot of stream lines
    Tboundary=boundaryConnectivityQ2Q1(X,T,Xp,Tp)
    phi = computeStreamFunction(ux,uy,X,T,Tboundary,referenceElementStokes)
    contourPlot(phi,X,T)
    plt.title('Stream lines')
    plt.show()

C=computeCmatrixNavierStokesSparse(sol,X,T,Xp.shape[0],referenceElementStokes)
K=A+C
sol = findSolutionSparse_SystemReduction(K, b, dofDir, valDir)

# Reynolds numbers and tolerances
reynolds_num = [2000, 2520, 3040, 3560, 4080, 4600]  # New Reynolds numbers
tolerance = 1e-8
max_iterations = 200

# Store results for plotting
velocity_solutions = []
pressure_solutions = []
streamlines = []
error_history = []  # Store convergence error history for each Re
max_pressures = []  # Store maximum pressure for each Re
L2_errors_velocity = []
L2_errors_pressure = []

for reynolds in reynolds_num:
    viscosity = 1 / reynolds
    [K_, f, Gt] = computeSystemStokes(X, T, Xp, Tp, referenceElementStokes)
    nOfNodesP = Xp.shape[0]
    A = bmat([[viscosity * K_, Gt], [np.transpose(Gt), np.zeros((nOfNodesP, nOfNodesP))]], format='csc')
    b = np.vstack((f, np.zeros((nOfNodesP, 1))))
    sol = findSolutionSparse_SystemReduction(A, b, dofDir, valDir)
    print(f"Solving for Reynolds number: {reynolds}, Viscosity: {viscosity}")
    converged = False
    iteration = 0
    sol_old = sol.copy()
    iteration_errors = []

    while not converged and iteration < max_iterations:
        iteration += 1
        print(f"  Iteration {iteration}")

        # Update system with the current solution
        C = computeCmatrixNavierStokesSparse(sol_old, X, T, Xp.shape[0], referenceElementStokes)
        K = A + C

        # Solve the updated system
        sol = findSolutionSparse_SystemReduction(K, b, dofDir, valDir)
        ux = sol[:nOfNodes]
        uy = sol[nOfNodes:2 * nOfNodes]
        p = sol[2 * nOfNodes:]

        # Compute error between iterations
        error_k = np.linalg.norm(sol[:2 * nOfNodes] - sol_old[:2 * nOfNodes]) / np.sqrt(nOfNodes)
        iteration_errors.append(error_k)
        print(f"    Error at iteration {iteration}: {error_k}")

        if error_k < tolerance:
            converged = True
        sol_old = sol.copy()

#%%
import os
import numpy as np
import pandas as pd
from scipy.sparse import bmat

# Cambiar al directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Reynolds numbers and tolerances
reynolds_num = [100, 200, 350, 800, 1000, 1500, 2000, 2520, 3040, 3560, 4080, 4600]
tolerance = 0.5e-8
max_iterations = 200

# Inicializar almacenamiento global
summary_data = []

for reynolds in reynolds_num:
    viscosity = 1 / reynolds
    [K_, f, Gt] = computeSystemStokes(X, T, Xp, Tp, referenceElementStokes)
    nOfNodesP = Xp.shape[0]
    A = bmat([[viscosity * K_, Gt], [np.transpose(Gt), np.zeros((nOfNodesP, nOfNodesP))]], format='csc')
    b = np.vstack((f, np.zeros((nOfNodesP, 1))))
    sol = findSolutionSparse_SystemReduction(A, b, dofDir, valDir)

    print(f"Solving for Reynolds number: {reynolds}, Viscosity: {viscosity}")
    converged = False
    iteration = 0
    sol_old = sol.copy()
    iteration_errors = []

    while not converged and iteration < max_iterations:
        iteration += 1
        print(f"  Iteration {iteration}")

        # Update system with the current solution
        C = computeCmatrixNavierStokesSparse(sol_old, X, T, Xp.shape[0], referenceElementStokes)
        K = A + C

        # Solve the updated system
        sol = findSolutionSparse_SystemReduction(K, b, dofDir, valDir)
        ux = sol[:nOfNodes]
        uy = sol[nOfNodes:2 * nOfNodes]
        p = sol[2 * nOfNodes:]

        # Compute error between iterations
        error_k = np.linalg.norm(sol[:2 * nOfNodes] - sol_old[:2 * nOfNodes]) / np.sqrt(nOfNodes)
        iteration_errors.append(error_k)
        print(f"    Error at iteration {iteration}: {error_k}")

        if error_k < tolerance:
            converged = True
        sol_old = sol.copy()

    # Guardar los errores por iteración en un archivo CSV
    log_errors = np.log10(iteration_errors)
    error_df = pd.DataFrame({"Iteration": np.arange(1, len(iteration_errors) + 1), "log10(Error)": log_errors})
    error_csv_path = os.path.join(script_dir, f"errors_Re_{reynolds}.csv")
    error_df.to_csv(error_csv_path, index=False)
    print(f"Saved error data for Re={reynolds} to {error_csv_path}")

    # Resumen global
    summary_data.append({
        "Reynolds Number": reynolds,
        "Viscosity": viscosity,
        "Iterations": len(iteration_errors),
        "Converged": converged,
    })

# Guardar resumen global en un archivo CSV
summary_df = pd.DataFrame(summary_data)
summary_csv_path = os.path.join(script_dir, "summary_iterations.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"Saved summary data to {summary_csv_path}")
#%%
import os
import numpy as np
from scipy.sparse import bmat
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
viscosity=2

# Stokes reference element and computational mesh
referenceElementStokes=defineRefElementStokesQ2Q1()
nx = 50; ny = nx; print('Number of elements',np.array([nx,ny]))
#X,T,Xp,Tp = UniformRectangleMeshStokesQ1Q2(domain,nx,ny)
X,T,Xp,Tp = CavityFlowMeshStokesQ1Q2(domain,nx,ny)
if do_plot == 1:
    plotMeshStokesQ2Q1(X,T,Xp,Tp,referenceElementStokes); plt.show()

# FE system assembly
# [K,f,Gt] = computeSystemStokes(X,T,Xp,Tp,referenceElementStokes)
# nOfNodesP=Xp.shape[0]
# A=np.block([[viscosity*K,Gt],[np.transpose(Gt),np.zeros((nOfNodesP,nOfNodesP))]])
# b=np.vstack((f,np.zeros((nOfNodesP,1))))
# plt.spy(A); plt.show()

A,b=computeSystemStokesSparse(viscosity,X,T,Xp,Tp,referenceElementStokes)

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
#sol = findSolution_SystemReduction(A,b,dofDir,valDir)
sol = findSolutionSparse_SystemReduction(A,b,dofDir,valDir)
ux=sol[:nOfNodes]; uy=sol[nOfNodes:2*nOfNodes]; p=sol[2*nOfNodes:]
    
#Plots
if do_plot==1:
    surfPlot(p,Xp,Tp)
    plt.title('pressure - FEM')
    plt.show()    
    plt.quiver(X[:,0],X[:,1],ux,uy)
    plt.title('velocity - FEM')
    plt.show()
    #Computation of the stream function and plot of stream lines
    Tboundary=boundaryConnectivityQ2Q1(X,T,Xp,Tp)
    phi = computeStreamFunction(ux,uy,X,T,Tboundary,referenceElementStokes)
    contourPlot(phi,X,T)
    plt.title('Stream lines')
    plt.show()

C=computeCmatrixNavierStokesSparse(sol,X,T,Xp.shape[0],referenceElementStokes)
K=A+C
sol = findSolutionSparse_SystemReduction(K, b, dofDir, valDir)

# Cambiar al directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Reynolds numbers and tolerances
reynolds_num = [100, 200, 400, 800, 1000, 1500, 2000, 2520, 3040, 3560, 4080, 4600]
# reynolds_num = [1500, 2000, 2520] 
tolerance = 1e-8
max_iterations = 200

# Archivo único para almacenar los resultados
output_file = os.path.join(script_dir, "convergence_data.txt")

# Limpiar o crear el archivo
with open(output_file, "w") as file:
    file.write("Reynolds,Iteration,log10(Error),MaxIterations\n")

for reynolds in reynolds_num:
    viscosity = 1 / reynolds
    [K_, f, Gt] = computeSystemStokes(X, T, Xp, Tp, referenceElementStokes)
    nOfNodesP = Xp.shape[0]
    A = bmat([[viscosity * K_, Gt], [np.transpose(Gt), np.zeros((nOfNodesP, nOfNodesP))]], format='csc')
    b = np.vstack((f, np.zeros((nOfNodesP, 1))))
    sol = findSolutionSparse_SystemReduction(A, b, dofDir, valDir)

    print(f"Solving for Reynolds number: {reynolds}, Viscosity: {viscosity}")
    converged = False
    iteration = 0
    sol_old = sol.copy()
    iteration_errors = []

    while not converged and iteration < max_iterations:
        iteration += 1
        print(f"  Iteration {iteration}")

        # Update system with the current solution
        C = computeCmatrixNavierStokesSparse(sol_old, X, T, Xp.shape[0], referenceElementStokes)
        K = A + C

        # Solve the updated system
        sol = findSolutionSparse_SystemReduction(K, b, dofDir, valDir)
        ux = sol[:nOfNodes]
        uy = sol[nOfNodes:2 * nOfNodes]
        p = sol[2 * nOfNodes:]

        # Compute error between iterations
        error_k = np.linalg.norm(sol[:2 * nOfNodes] - sol_old[:2 * nOfNodes]) / np.sqrt(nOfNodes)
        iteration_errors.append(error_k)
        print(f"    Error at iteration {iteration}: {error_k}")

        if error_k < tolerance:
            converged = True
        sol_old = sol.copy()

    # Guardar los errores por iteración en un archivo TXT
    log_errors = np.log10(iteration_errors)
    max_iterations_reached = len(iteration_errors)
    Tboundary=boundaryConnectivityQ2Q1(X,T,Xp,Tp)
    phi = computeStreamFunction(ux,uy,X,T,Tboundary,referenceElementStokes)
    contourPlot(phi,X,T)
    plt.title(f'Stream lines for Re={reynolds}')
    plt.show()
    with open(output_file, "a") as file:
        for i, log_error in enumerate(log_errors, start=1):
            file.write(f"{reynolds},{i},{log_error},{max_iterations_reached}\n")
    print(f"Saved error data for Re={reynolds} to {output_file}")

print(f"All data saved to {output_file}")


#%%
import pandas as pd
import matplotlib.pyplot as plt
import os


import os
print(os.getcwd())

# Cambiar al directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Adjusting the file reading process for a .txt file with proper delimiters
file_path = 'convergence_data.txt'

# Attempting to read the .txt file with appropriate delimiter settings
data = pd.read_csv(file_path, delim_whitespace=True)

# Displaying the first few rows to verify the correct parsing of the file
data.head()


# Plot 1: Convergence Behavior for Different Reynolds Numbers
plt.figure(figsize=(10, 6))
for reynolds, group in data.groupby('Reynolds'):
    plt.plot(group['Iteration'], group['log10(Error)'], marker='o', label=f"Re = {reynolds}")

plt.xlabel('Iteration')
plt.ylabel('log10(Error)')
plt.title('Convergence Behavior for Different Reynolds Numbers')
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Number of Iterations vs Reynolds Number
max_iterations = data.groupby('Reynolds')['MaxIterations'].max()

plt.figure(figsize=(10, 6))
plt.plot(max_iterations.index, max_iterations.values, marker='o')

plt.xlabel('Reynolds Number (Re)')
plt.ylabel('Number of Iterations')
plt.title('Number of Iterations vs Reynolds Number')
plt.grid(True)
plt.show()
#%%
# Check if the file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    try:
        # Specify the correct delimiter (comma) and ensure the header is interpreted properly
        data = pd.read_csv(file_path, sep=",", header=0)
        
        # Print column names to confirm proper loading
        print("Columns:", data.columns)
        
        # Plot 1: Convergence Behavior for Different Reynolds Numbers
        plt.figure(figsize=(10, 6))
        for reynolds, group in data.groupby('Reynolds'):
            plt.plot(group['Iteration'], group['log10(Error)'], marker='o', label=f"Re = {reynolds}")

        plt.xlabel('Iteration', fontsize = 14)
        plt.ylabel('log10(Error)', fontsize = 14    )
        plt.title('Convergence Behavior for Different Reynolds Numbers')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot 2: Number of Iterations vs Reynolds Number
        max_iterations = data.groupby('Reynolds')['MaxIterations'].max()

        plt.figure(figsize=(10, 6))
        plt.plot(max_iterations.index, max_iterations.values, marker='o')

        plt.xlabel('Reynolds Number (Re)', fontsize = 14)
        plt.ylabel('Number of Iterations',fontsize = 14)
        plt.title('Number of Iterations vs Reynolds Number')
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")