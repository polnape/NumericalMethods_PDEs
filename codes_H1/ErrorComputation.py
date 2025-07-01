import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as mtri
#from analyticalInfo2 import *
from analyticalInfo import *
                
def computeL2Error(u,X,T,refElement):
    u_ex = exactSol(X)[0]
    u_ex.shape = u.shape
 
    nIP = refElement.nIP # Number of integration points
    wIP = refElement.integrationWeights 
    N    = refElement.N # Basis functions and derivatives at Gauss points
    Nxi  = refElement.Nxi 
    Neta = refElement.Neta 
    L2err = 0
    nOfElements = T.shape[0]
    for e in np.arange(nOfElements):
        Te = T[e,:]
        Xe = X[Te]
        ue = u[Te]
        for ip in np.arange(nIP):
            N_ip = N[ip,:]
            Nxi_ip = Nxi[ip,:] 
            Neta_ip = Neta[ip,:]
            J = np.array([ [ Nxi_ip@Xe[:,0], Nxi_ip@Xe[:,1]], [Neta_ip@Xe[:,0], Neta_ip@Xe[:,1]]])
            dvolu = wIP[ip]*np.linalg.det(J)
            grad_ref = np.vstack((Nxi_ip, Neta_ip)) 
            grad = np.linalg.solve(J, grad_ref)
            Nx_ip = grad[0,:] 
            Ny_ip = grad[1,:]

            x_ip = N_ip@Xe
            x_ip.shape =(1,2)
            u_ex_ip,ux_ex_ip,uy_ex_ip = exactSol(x_ip)
            
            # numerical solution
            u_ip = N_ip@ue
            # error
            L2err = L2err + (u_ip - u_ex_ip)**2*dvolu
    L2err = np.sqrt(L2err)
    return L2err

def computeH1Error(u, X, T, refElement):
    u_ex, u_x_ex, u_y_ex = exactSol(X)  # Solución exacta y derivadas exactas
    u_ex.shape = u.shape
    
    nIP = refElement.nIP  # Número de puntos de integración
    wIP = refElement.integrationWeights
    N = refElement.N  # Funciones de forma
    Nxi = refElement.Nxi
    Neta = refElement.Neta
    
    H1err = 0
    nOfElements = T.shape[0]
    
    for e in np.arange(nOfElements):
        Te = T[e, :]
        Xe = X[Te]
        ue = u[Te]
        
        for ip in np.arange(nIP):
            N_ip = N[ip, :]
            Nxi_ip = Nxi[ip, :]
            Neta_ip = Neta[ip, :]
            
            # Jacobiano y volumen
            J = np.array([[Nxi_ip @ Xe[:, 0], Nxi_ip @ Xe[:, 1]], [Neta_ip @ Xe[:, 0], Neta_ip @ Xe[:, 1]]])
            dvolu = wIP[ip] * np.linalg.det(J)
            grad_ref = np.vstack((Nxi_ip, Neta_ip))
            grad = np.linalg.solve(J, grad_ref)
            Nx_ip = grad[0, :]
            Ny_ip = grad[1, :]
            
            # Coordenadas en el punto de integración
            x_ip = N_ip @ Xe
            x_ip.shape = (1, 2)
            
            # Solución exacta en el punto de integración
            u_ex_ip, u_x_ex_ip, u_y_ex_ip = exactSol(x_ip)
            
            # Solución numérica en el punto de integración
            u_ip = N_ip @ ue
            u_x_num = Nx_ip @ ue
            u_y_num = Ny_ip @ ue
            
            # Error en H1 (L2 + derivadas primeras)
            H1err += ((u_ip - u_ex_ip)**2 + (u_x_num - u_x_ex_ip)**2 + (u_y_num - u_y_ex_ip)**2) * dvolu
    
    H1err = np.sqrt(H1err)
    return H1err





    