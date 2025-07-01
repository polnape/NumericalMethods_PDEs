import sys
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as mtri
#from analyticalInfo2 import *
from analyticalInfo import *

# f = 0 for problem 2
def computeSystemLaplace2(X,T,refElt, Kmat):
    # Initialise system matrix and rhs vector
    matrix = Kmat
    nOfNodes = X.shape[0]
    nOfElements = T.shape[0]
    K = np.zeros((nOfNodes,nOfNodes))
    f = np.zeros((nOfNodes,1))
    # loop in elements to obtain the system of equations
    for e in np.arange(nOfElements):
        Te = T[e,:]
        Xe = X[Te,:]
        Ke,fe = elementMatrices2(Xe,T,refElt, matrix)
        fe = fe*0
        for i in np.arange(len(Te)):
            fe = fe*0
            f[Te[i]] = f[Te[i]] + fe[i]
            for j in np.arange(len(Te)):
                K[Te[i],Te[j]] = K[Te[i],Te[j]] + Ke[i,j] 
                f = f*0
    return K, f

# Problema 2 
# Problema adaptado para incluir la matriz de conductividad K
def elementMatrices2(Xe, T, refElement, Kmat):
    Kmat = -Kmat
    nOfElementNodes = refElement.nOfElementNodes 
    nIP = refElement.nIP
    wIP = refElement.integrationWeights 
    N    = refElement.N
    Nxi  = refElement.Nxi 
    Neta = refElement.Neta 
    
    Ke = np.zeros((nOfElementNodes, nOfElementNodes))
    fe = np.zeros(nOfElementNodes)
    
    for ip in np.arange(nIP): 
        N_ip = N[ip,:]
        Nxi_ip = Nxi[ip,:] 
        Neta_ip = Neta[ip,:]
        
        # Jacobiano y determinante
        J = np.array([[Nxi_ip @ Xe[:,0], Nxi_ip @ Xe[:,1]], 
                      [Neta_ip @ Xe[:,0], Neta_ip @ Xe[:,1]]])
        dvolu = wIP[ip] * np.linalg.det(J)
        
        # Gradiente de las funciones de forma en el sistema de referencia
        grad_ref = np.vstack((Nxi_ip, Neta_ip)) 
        grad = np.linalg.solve(J, grad_ref)  # Cambio a las coordenadas reales
        
        # Descomponemos el gradiente en x e y
        Nx = grad[0,:] 
        Ny = grad[1,:]
        
        # Aplicamos la matriz de conductividad K
        Kgrad = np.dot(Kmat, np.vstack((Nx, Ny)))  # Multiplicamos por K1 y K2
        
        # Producto escalar de los gradientes multiplicado por la conductividad
        Ke += (np.outer(Kgrad[0,:], Nx) + np.outer(Kgrad[1,:], Ny)) * dvolu
        
        # Si no hay término fuente, no es necesario modificar fe
        # fe = fe + N_ip * sourceTerm(x_ip) * dvolu   # Si hay fuente
    fe = fe*0

    return Ke, fe

# PROBLEMA 1
def computeSystemLaplace(X,T,refElt):
    # Initialise system matrix and rhs vector
    nOfNodes = X.shape[0]
    nOfElements = T.shape[0]
    K = np.zeros((nOfNodes,nOfNodes))
    f = np.zeros((nOfNodes,1))
    # loop in elements to obtain the system of equations
    for e in np.arange(nOfElements):
        Te = T[e,:]
        Xe = X[Te,:]
        Ke,fe = elementMatrices(Xe,T,refElt)
        for i in np.arange(len(Te)):
            f[Te[i]] = f[Te[i]] + fe[i]
            for j in np.arange(len(Te)):
                K[Te[i],Te[j]] = K[Te[i],Te[j]] + Ke[i,j]  
    return K, f

def elementMatrices(Xe,T,refElement):
    nOfElementNodes = refElement.nOfElementNodes 
    nIP = refElement.nIP
    wIP = refElement.integrationWeights 
    N    = refElement.N
    Nxi  = refElement.Nxi 
    Neta = refElement.Neta 
    
    Ke = np.zeros((nOfElementNodes,nOfElementNodes))
    fe = np.zeros(nOfElementNodes)
    for ip in np.arange(nIP): 
        N_ip = N[ip,:]
        Nxi_ip = Nxi[ip,:] 
        Neta_ip = Neta[ip,:]
        J = np.array([ [ Nxi_ip@Xe[:,0], Nxi_ip@Xe[:,1]], [Neta_ip@Xe[:,0], Neta_ip@Xe[:,1]]])
        dvolu = wIP[ip]*np.linalg.det(J)
        grad_ref = np.vstack((Nxi_ip, Neta_ip)) 
        grad = np.linalg.solve(J, grad_ref)
        Nx = grad[0,:] 
        Ny = grad[1,:]
        Ke = Ke + (np.outer(Nx,Nx) + np.outer(Ny,Ny))*dvolu 
        x_ip = N_ip@Xe
        fe = fe + N_ip*sourceTerm(x_ip)*dvolu

    return Ke,fe 







def findSolution_SystemReduction(X, T, K, f, nodesDir, valDir):
    nOfNodes = X.shape[0]
    nodesUnk = np.setdiff1d(np.arange(nOfNodes), nodesDir)
    
    # Reducción del sistema (Dirichlet)
    f_red = f - K[:, nodesDir] @ valDir
    K_red = K[nodesUnk, :][:, nodesUnk]
    f_red = f_red[nodesUnk, :]
   
    
    # Solución del sistema reducido
    sol = np.linalg.solve(K_red, f_red)
    
    # Asignación de valores de Dirichlet y solución
    u = np.zeros((nOfNodes, 1))
    u[nodesDir] = valDir
    u[nodesUnk] = sol
    return u