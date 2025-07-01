import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as mtri
#from analyticalInfo2 import *
from analyticalInfo import *
from analyticalInfoStokes import *
                
def computeL2Error(u,X,T,refElement):
 
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


def computeL2ErrorStokes(sol,X,T,Xp,Tp,refElement):
    nOfNodes=X.shape[0]
    ux=sol[:nOfNodes]; uy=sol[nOfNodes:2*nOfNodes]; p=sol[2*nOfNodes:]
 
    nIP = refElement.nIP # Number of integration points
    wIP = refElement.integrationWeights 
    N    = refElement.N # Basis functions and derivatives at Gauss points
    Nxi  = refElement.Nxi 
    Neta = refElement.Neta 
    Np =refElement.Npressure
    L2errVelocity = 0
    L2errPressure = 0
    nOfElements = T.shape[0]
    for e in np.arange(nOfElements):
        Te = T[e,:]
        Xe = X[Te,:]
        uxe = ux[Te]; uye = uy[Te]; pe=p[Tp[e,:]]
        for ip in np.arange(nIP):
            N_ip = N[ip,:]
            Nxi_ip = Nxi[ip,:] 
            Neta_ip = Neta[ip,:]
            Np_ip = Np[ip,:]
            J = np.array([ [ Nxi_ip@Xe[:,0], Nxi_ip@Xe[:,1]], [Neta_ip@Xe[:,0], Neta_ip@Xe[:,1]]])
            dvolu = wIP[ip]*np.linalg.det(J)
            grad_ref = np.vstack((Nxi_ip, Neta_ip)) 
            grad = np.linalg.solve(J, grad_ref)
            Nx_ip = grad[0,:] 
            Ny_ip = grad[1,:]
            # analytical solution 
            x_ip = N_ip@Xe; x_ip.shape =(1,2)
            ux_ex,uy_ex,p_ex = exactSolStokes(x_ip)
            # numerical solution
            ux_ip = N_ip@uxe
            uy_ip = N_ip@uye
            p_ip = Np_ip@pe
            # error
            L2errVelocity = L2errVelocity + ((ux_ip - ux_ex)**2+(uy_ip - uy_ex)**2)*dvolu
            L2errPressure = L2errPressure + ((p_ip-p_ex)**2)*dvolu
    L2errVelocity = np.sqrt(L2errVelocity)[0]
    L2errPressure = np.sqrt(L2errPressure)[0]
    return L2errVelocity,L2errPressure