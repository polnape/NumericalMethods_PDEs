# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:05:24 2024

@author: polop
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as mtri

def defineRefElement(elementType, degree):
    class refElement:
        pass
    refElement.type = elementType
    refElement.degree = degree
    if elementType == 0 and degree == 1:
        refElement.nOfElementNodes = 4
        refElement.nodesCoord = np.array([[-1,-1], [1,-1],[1,1],[-1,1]])
        a = np.sqrt(1/3)
        z = np.array([[-a,-a], [a,-a], [a,a], [-a,a]])
        refElement.integrationPoints = z
        refElement.integrationWeights = np.array([1,1,1,1])
        refElement.nIP = 4
        xi = z[:,[0]]; eta = z[:,[1]]
        refElement.N = 1/4*np.block([(xi-1)*(eta-1), -(xi+1)*(eta-1), (xi+1)*(eta+1), -(xi-1)*(eta+1)])
        refElement.Nxi = 1/4*np.block([eta-1, -eta+1, eta+1, -eta-1])
        refElement.Neta = 1/4*np.block([xi-1, -xi-1, xi+1, -xi+1])
    elif elementType == 1 and degree == 1:
        refElement.nOfElementNodes = 3
        refElement.nodesCoord = np.array([[0,0], [1,0],[0,1]])
        a = 1/2; b = 1/6
        z = np.array([[a,0], [0,a], [a,a]])
        refElement.integrationPoints = z
        refElement.integrationWeights = np.array([b,b,b])
        refElement.nIP = 3
        xi = z[:,[0]]; eta = z[:,[1]]
        zeroMat = np.reshape(np.zeros(len(xi)),(len(xi),1))
        oneMat = np.reshape(np.ones(len(xi)),(len(xi),1))
        refElement.N = np.block([1-xi-eta, xi, eta])
        refElement.Nxi = np.block([-oneMat, oneMat, zeroMat])
        refElement.Neta = np.block([-oneMat, zeroMat, oneMat])
            
    # para 9 integration points
    # elif elementType == 0 and degree == 2:
    #     refElement.nOfElementNodes = 9
    #     refElement.nodesCoord = np.array([[-1,-1], [0,-1], [1,-1], [1,0], [1,1], [0,1], [-1,1], [-1,0], [0,0]])
        
    #     # Definir 25 puntos de integración (Gauss-Legendre) para elementos cuadráticos
    #     # 5 puntos de Gauss-Legendre en cada dirección
    #     gaussPoints = np.array([-np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 3, 
    #                             -np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 3, 
    #                               0, 
    #                               np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 3, 
    #                               np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 3])
    
    #     gaussWeights = np.array([(322 - 13 * np.sqrt(70)) / 900,
    #                               (322 + 13 * np.sqrt(70)) / 900,
    #                               128 / 225,
    #                               (322 + 13 * np.sqrt(70)) / 900,
    #                               (322 - 13 * np.sqrt(70)) / 900])
    
        
    #     z = np.array([[xi, eta] for xi in gaussPoints for eta in gaussPoints])
    #     print(z.shape)
    #     refElement.integrationPoints = z
        
    #     # Crear los pesos de integración como producto cartesiano de los pesos en cada dirección
    #     weights = np.outer(gaussWeights, gaussWeights).flatten()
    #     print(weights.shape)
    #     refElement.integrationWeights = weights
    #     refElement.nIP = 25
        
    #     xi = z[:,[0]]; eta = z[:,[1]]
    #     zeroMat = np.reshape(np.zeros(len(xi)),(len(xi),1))
    #     oneMat = np.reshape(np.ones(len(xi)),(len(xi),1))
    #     refElement.N = np.block([1-xi-eta, xi, eta])
    #     refElement.Nxi = np.block([-oneMat, oneMat, zeroMat])
    #     refElement.Neta = np.block([-oneMat, zeroMat, oneMat])
        
        # ----
    elif elementType == 0 and degree == 2:
        
        refElement.nOfElementNodes = 9
        refElement.nodesCoord = np.array([
            [-1,-1], [0,-1], [1,-1],  # Nodos en el borde inferior
            [1,0],                    # Nodo en el borde derecho (medio)
            [1,1], [0,1], [-1,1],     # Nodos en el borde superior
            [-1,0],                   # Nodo en el borde izquierdo (medio)
            [0,0]                     # Nodo central
        ])
        
        # Definir 25 puntos de integración (Gauss-Legendre) para elementos cuadráticos
        # 5 puntos de Gauss-Legendre en cada dirección
        gaussPoints = np.array([-np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 3, 
                                -np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 3, 
                                  0, 
                                  np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 3, 
                                  np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 3])
    
        gaussWeights = np.array([(322 - 13 * np.sqrt(70)) / 900,
                                  (322 + 13 * np.sqrt(70)) / 900,
                                  128 / 225,
                                  (322 + 13 * np.sqrt(70)) / 900,
                                  (322 - 13 * np.sqrt(70)) / 900])
    
        # Crear un producto cartesiano para obtener los puntos en las dos direcciones
        z = np.array([[zi, zj] for zi in gaussPoints for zj in gaussPoints])
        refElement.integrationPoints = z
        
        # Crear los pesos de integración como producto cartesiano de los pesos en cada dirección
        weights = np.outer(gaussWeights, gaussWeights).flatten()
        refElement.integrationWeights = weights
        refElement.nIP = 25
        xi = z[:, [0]]  # xi coordinates of integration points
        eta = z[:, [1]]  # eta coordinates of integration points
        
        # Funciones de forma cuadráticas para los 9 nodos
        refElement.N = np.block([
            0.25 * (1 - xi) * (1 - eta) * xi*eta,  # N1 (esquina inferior izquierda)
            -0.5 * (1 - xi**2) * (1 - eta)*eta,                # N2 (borde inferior medio)
            -0.25 * (1 + xi) * (1 - eta) * xi*eta,  # N3 (esquina inferior derecha)
            0.5 * (1 + xi) * (1 - eta**2)*xi,                # N4 (borde derecho medio)
            0.25 * (1 + xi) * (1 + eta) * xi*eta,  # N5 (esquina superior derecha)
            0.5 * (1 - xi**2) * (1 + eta)*eta,                # N6 (borde superior medio)
            -0.25 * (1 - xi) * (1 + eta) * xi*eta,  # N7 (esquina superior izquierda)
            -0.5 * (1 - xi) * (1 - eta**2)*xi,                # N8 (borde izquierdo medio)
            (1 - xi**2) * (1 - eta**2)                    # N9 (centro)
        ])
        
        # Derivadas con respecto a xi
        refElement.Nxi = np.block([
            -0.25 * eta * xi * (1 - eta) + eta * (0.25 - 0.25 * xi) * (1 - eta),  # dN1/dxi (esquina inferior izquierda)
            1.0 * eta * xi * (1 - eta),                                           # dN2/dxi (borde inferior medio)
            -0.25 * eta * xi * (1 - eta) + eta * (1 - eta) * (-0.25 * xi - 0.25), # dN3/dxi (esquina inferior derecha)
            0.5 * xi * (1 - eta**2) + (1 - eta**2) * (0.5 * xi + 0.5),            # dN4/dxi (borde derecho medio)
            0.25 * eta * xi * (eta + 1) + eta * (eta + 1) * (0.25 * xi + 0.25),   # dN5/dxi (esquina superior derecha)
            -1.0 * eta * xi * (eta + 1),                                          # dN6/dxi (borde superior medio)
            0.25 * eta * xi * (eta + 1) + eta * (eta + 1) * (0.25 * xi - 0.25),   # dN7/dxi (esquina superior izquierda)
            0.5 * xi * (1 - eta**2) + (1 - eta**2) * (0.5 * xi - 0.5),            # dN8/dxi (borde izquierdo medio)
            -2 * xi * (1 - eta**2)                                                # dN9/dxi (centro)
        ])
        
        # Derivadas con respecto a eta
        refElement.Neta = np.block([
            -eta * xi * (0.25 - 0.25 * xi) + xi * (0.25 - 0.25 * xi) * (1 - eta),  # dN1/deta (esquina inferior izquierda)
            -eta * (0.5 * xi**2 - 0.5) + (1 - eta) * (0.5 * xi**2 - 0.5),          # dN2/deta (borde inferior medio)
            -eta * xi * (-0.25 * xi - 0.25) + xi * (1 - eta) * (-0.25 * xi - 0.25),# dN3/deta (esquina inferior derecha)
            -2 * eta * xi * (0.5 * xi + 0.5),                                      # dN4/deta (borde derecho medio)
            eta * xi * (0.25 * xi + 0.25) + xi * (eta + 1) * (0.25 * xi + 0.25),   # dN5/deta (esquina superior derecha)
            eta * (0.5 - 0.5 * xi**2) + (0.5 - 0.5 * xi**2) * (eta + 1),           # dN6/deta (borde superior medio)
            eta * xi * (0.25 * xi - 0.25) + xi * (eta + 1) * (0.25 * xi - 0.25),   # dN7/deta (esquina superior izquierda)
            -2 * eta * xi * (0.5 * xi - 0.5),                                      # dN8/deta (borde izquierdo medio)
            -2 * eta * (1 - xi**2)                                                 # dN9/deta (centro)
        ])



        

 

    else: 
        print("Not")
        print('Not implemented element')
        sys.exit()
        
    return refElement