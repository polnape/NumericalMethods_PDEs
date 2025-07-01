# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:05:40 2024

@author: polop
"""

import numpy as np

def computeSystemHeatEquation(X, T, refElt):
    # Inicializar matrices y vector del sistema
    nOfNodes = X.shape[0]
    nOfElements = T.shape[0]
    K = np.zeros((nOfNodes, nOfNodes))  # Matriz de rigidez global
    M = np.zeros((nOfNodes, nOfNodes))  # Matriz de masa global
    f = np.zeros((nOfNodes, 1))         # Vector de fuerzas

    # Bucle sobre los elementos para ensamblar el sistema
    for e in np.arange(nOfElements):
        Te = T[e, :]
        # print(Te)
        Xe = X[Te, :]
        # print(Xe)
        
        # Calcular matrices elementales
        Ke, Me, fe = elementMatrices(Xe, refElt)
        
        # Ensamblar en las matrices globales
        for i in np.arange(len(Te)):
            f[Te[i]] += fe[i]
            for j in np.arange(len(Te)):
                K[Te[i], Te[j]] += Ke[i, j]
                M[Te[i], Te[j]] += Me[i, j]
    
    # Calcular la matriz de masa lumped
    MLumped = np.zeros((nOfNodes, nOfNodes))
    for i in range(nOfNodes):
        MLumped[i, i] = np.sum(M[i, :])  # Sumar las filas para obtener la versión lumped
    
    return K, M, MLumped, f

def elementMatrices(Xe, refElement):
    nOfElementNodes = refElement.nOfElementNodes 
    nIP = refElement.nIP
    wIP = refElement.integrationWeights 
    N = refElement.N
    Nxi = refElement.Nxi 
    Neta = refElement.Neta 

    # Inicializar matrices elementales
    Ke = np.zeros((nOfElementNodes, nOfElementNodes))
    Me = np.zeros((nOfElementNodes, nOfElementNodes))
    fe = np.zeros(nOfElementNodes)
    
    # Bucle sobre los puntos de cuadratura
    for ip in np.arange(nIP): 
        N_ip = N[ip, :]
        Nxi_ip = Nxi[ip, :] 
        Neta_ip = Neta[ip, :]
        
        # Calcular el jacobiano y su determinante
        J = np.array([[Nxi_ip @ Xe[:, 0], Nxi_ip @ Xe[:, 1]],
                      [Neta_ip @ Xe[:, 0], Neta_ip @ Xe[:, 1]]])
        detJ = np.linalg.det(J)
        dvolu = wIP[ip] * detJ
        
        # Calcular gradientes en coordenadas globales
        grad_ref = np.vstack((Nxi_ip, Neta_ip)) 
        grad = np.linalg.solve(J, grad_ref)
        Nx = grad[0, :] 
        Ny = grad[1, :]
        
        # Ensamblar la matriz de rigidez elemental
        Ke += (np.outer(Nx, Nx) + np.outer(Ny, Ny)) * dvolu
        
        # Ensamblar la matriz de masa elemental
        Me += np.outer(N_ip, N_ip) * dvolu
        
        # Vector de fuerzas (fuente)
        x_ip = N_ip @ Xe
        fe += N_ip * sourceTerm(x_ip) * dvolu

    return Ke, Me, fe

def sourceTerm(x):
    # Aquí defines tu término fuente, que podría ser cero si no hay fuente
    return 0.0





# def findSolution_SystemReduction(X, T, K, f, nodesDir, valDir):
#     nOfNodes = X.shape[0]
#     nodesUnk = np.setdiff1d(np.arange(nOfNodes), nodesDir)
    
#     # Reducción del sistema (Dirichlet)
#     f_red = f - K[:, nodesDir] @ valDir
#     K_red = K[nodesUnk, :][:, nodesUnk]
#     f_red = f_red[nodesUnk, :]
   
    
#     # Solución del sistema reducido
#     sol = np.linalg.solve(K_red, f_red)
    
#     # Asignación de valores de Dirichlet y solución
#     u = np.zeros((nOfNodes, 1))
#     u[nodesDir] = valDir
#     u[nodesUnk] = sol
#     return u