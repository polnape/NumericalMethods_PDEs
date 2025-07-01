import sys
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as mtri
#from analyticalInfo2 import *
from analyticalInfo import *
from analyticalInfoCavityFlowStokes import *

def findSolution_SystemReduction(K,f,nodesDir,valDir):
    nOfNodes = K.shape[0]
    nodesUnk = np.setdiff1d(np.arange(nOfNodes),nodesDir)
    f_red = f - K[:,nodesDir]@valDir
    K_red = K[nodesUnk,:][:,nodesUnk]
    f_red = f_red[nodesUnk,:]
    sol = np.linalg.solve(K_red,f_red)
    u = np.zeros((nOfNodes,1))
    u[nodesDir] = valDir
    u[nodesUnk] = sol
    print('Resuced system:  size=',K_red.shape[0],' rank=',np.linalg.matrix_rank(K_red))
    return u

def findSolutionSparse_SystemReduction(K,f,nodesDir,valDir):
    nOfNodes = K.shape[0]
    nodesUnk = np.setdiff1d(np.arange(nOfNodes),nodesDir)
    f_red = f - K[:,nodesDir]@valDir
    K_red = K[nodesUnk,:][:,nodesUnk]
    f_red = f_red[nodesUnk,:]
    sol = spsolve(K_red,f_red); sol.shape=(len(sol),1)
    u = np.zeros((nOfNodes,1))
    u[nodesDir] = valDir
    u[nodesUnk] = sol
    return u

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
        Ke,fe = elementMatrices(Xe,refElt)
        for i in np.arange(len(Te)):
            f[Te[i]] = f[Te[i]] + fe[i]
            for j in np.arange(len(Te)):
                K[Te[i],Te[j]] = K[Te[i],Te[j]] + Ke[i,j]  
    return K, f

def computeSystemLaplaceSparse(X,T,refElt):
    # Initialise system matrix and rhs vector
    nOfNodes = X.shape[0]
    nOfElements,nOfElementNodes = T.shape
    rowInd=np.zeros((nOfElements*nOfElementNodes**2,1),dtype=int); rowInd.shape=(len(rowInd),)
    colInd=np.zeros((nOfElements*nOfElementNodes**2,1),dtype=int); colInd.shape=(len(colInd),)
    valK=np.zeros((nOfElements*nOfElementNodes**2,1)); valK.shape=(len(valK),)
    counter=0
    f = np.zeros((nOfNodes,1))  
    # loop in elements to obtain the system of equations
    for e in np.arange(nOfElements):
        Te = T[e,:]
        Xe = X[Te,:]
        Ke,fe = elementMatrices(Xe,refElt)
        for i in np.arange(len(Te)):
            f[Te[i]] = f[Te[i]] + fe[i]
            for j in np.arange(len(Te)):
                rowInd[counter]=Te[i]
                colInd[counter]=Te[j]
                valK[counter]= Ke[i,j]
                counter=counter+1
    matrix=coo_matrix((valK,(rowInd,colInd)),shape=(nOfNodes,nOfNodes))
    K= matrix.tocsc()
    return K, f

def elementMatrices(Xe,refElement):
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


def computeSystemStokes(X,T,Xp,Tp,refElt):
    # Initialise system matrix and rhs vector
    nOfNodes = X.shape[0]
    nOfNodesP = Xp.shape[0]
    nOfElements = T.shape[0]
    K = np.zeros((2*nOfNodes,2*nOfNodes))
    Gt = np.zeros((2*nOfNodes,nOfNodesP))
    f = np.zeros((2*nOfNodes,1))
    # loop in elements to obtain the system of equations
    for e in np.arange(nOfElements):
        Te = T[e,:]
        Xe = X[Te,:]
        Ke,fe,Gte = elementMatricesStokes(Xe,refElt)
        Tvelo=np.hstack((Te,Te+nOfNodes))
        Tpe=Tp[e,:]
        for i in np.arange(len(Tvelo)):
            f[Tvelo[i]] = f[Tvelo[i]] + fe[i]
            for j in np.arange(len(Tvelo)):
                K[Tvelo[i],Tvelo[j]] = K[Tvelo[i],Tvelo[j]] + Ke[i,j]  
            for j in np.arange(len(Tpe)):
                Gt[Tvelo[i],Tpe[j]] = Gt[Tvelo[i],Tpe[j]] + Gte[i,j]
    return K, f, Gt


def computeSystemStokesSparse(viscosity,X,T,Xp,Tp,refElt):
    # Initialise system matrix and rhs vector
    nOfNodesP = Xp.shape[0]
    nOfNodes = X.shape[0]
    nOfElements,nOfElementNodesP = Tp.shape
    nOfElements,nOfElementNodes = T.shape
    rowIndK=np.zeros((nOfElements*(2*nOfElementNodes)**2,1),dtype=int); rowIndK.shape=(len(rowIndK),)
    colIndK=np.zeros((nOfElements*(2*nOfElementNodes)**2,1),dtype=int); colIndK.shape=(len(colIndK),)
    valK=np.zeros((nOfElements*(2*nOfElementNodes)**2,1)); valK.shape=(len(valK),)
    rowIndGt=np.zeros((nOfElements*(2*nOfElementNodes)*nOfElementNodesP,1),dtype=int); rowIndGt.shape=(len(rowIndGt),)
    colIndGt=np.zeros((nOfElements*(2*nOfElementNodes)*nOfElementNodesP,1),dtype=int); colIndGt.shape=(len(colIndGt),)
    valGt=np.zeros((nOfElements*(2*nOfElementNodes)*nOfElementNodesP,1)); valGt.shape=(len(valGt),)
    counterK=0
    counterG=0
    b = np.zeros((2*nOfNodes+nOfNodesP,1))
    # loop in elements to obtain the system of equations
    for e in np.arange(nOfElements):
        Te = T[e,:]
        Xe = X[Te,:]
        Ke,fe,Gte = elementMatricesStokes(Xe,refElt)
        Tvelo=np.hstack((Te,Te+nOfNodes))
        Tpe=Tp[e,:]
        for i in np.arange(len(Tvelo)):
            b[Tvelo[i]] = b[Tvelo[i]] + fe[i]
            for j in np.arange(len(Tvelo)):
                rowIndK[counterK]=Tvelo[i]
                colIndK[counterK]=Tvelo[j]
                valK[counterK]= Ke[i,j]
                counterK=counterK+1
            for j in np.arange(len(Tpe)):
                rowIndGt[counterG]=Tvelo[i]
                colIndGt[counterG]=Tpe[j]
                valGt[counterG]= Gte[i,j]
                counterG=counterG+1
    valA=np.concatenate((viscosity*valK,valGt,valGt))
    rowInd=np.concatenate((rowIndK,rowIndGt,colIndGt+2*nOfNodes))
    colInd=np.concatenate((colIndK,colIndGt+2*nOfNodes,rowIndGt))
    dim=2*nOfNodes+nOfNodesP
    matrix=coo_matrix((valA,(rowInd,colInd)),shape=(dim,dim))
    A= matrix.tocsc()
    return A, b


def elementMatricesStokes(Xe,refElement):
    nOfElementNodes = (refElement.N).shape[1]
    nOfElementNodesP = (refElement.Npressure).shape[1]
    nIP = refElement.nIP
    wIP = refElement.integrationWeights 
    N    = refElement.N
    Nxi  = refElement.Nxi 
    Neta = refElement.Neta
    Np = refElement.Npressure
    zeros=np.zeros((1,nOfElementNodes))
    
    Ke = np.zeros((nOfElementNodes*2,nOfElementNodes*2))
    Gte = np.zeros((nOfElementNodes*2,nOfElementNodesP))
    fe = np.zeros(nOfElementNodes*2)
    for ip in np.arange(nIP): 
        N_ip = N[ip,:]
        Nxi_ip = Nxi[ip,:] 
        Neta_ip = Neta[ip,:]
        Np_ip = Np[ip,:]
        J = np.array([ [ Nxi_ip@Xe[:,0], Nxi_ip@Xe[:,1]], [Neta_ip@Xe[:,0], Neta_ip@Xe[:,1]]])
        dvolu = wIP[ip]*np.linalg.det(J)
        grad_ref = np.vstack((Nxi_ip, Neta_ip)) 
        grad = np.linalg.solve(J, grad_ref)
        Nx = grad[0,:] 
        Ny = grad[1,:]
        matN =  np.block([[N_ip,zeros],[zeros,N_ip]])                                        
        gradN = np.block([[Nx,zeros],[zeros,Nx],[Ny,zeros],[zeros,Ny]])
        D=np.hstack((Nx,Ny))
        Ke = Ke + ((np.transpose(gradN))@gradN)*dvolu
        fe = fe + (np.transpose(matN))@(bodyForceStokes(N_ip@Xe)*dvolu)
        Gte = Gte - np.outer(D,Np_ip)*dvolu
    return Ke,fe,Gte

def computeCmatrixNavierStokes(V,X,T,refElt):
    # Initialise system matrix and rhs vector
    nOfNodes = X.shape[0]
    nOfElements = T.shape[0]
    Vx=V[:nOfNodes]; Vy=V[nOfNodes:]
    C = np.zeros((2*nOfNodes,2*nOfNodes))
    # loop in elements to obtain the system of equations
    for e in np.arange(nOfElements):
        Te = T[e,:]
        Xe = X[Te,:]
        Vxe=Vx[Te]
        Vye=Vy[Te]
        Ce = elementCmatrixNS(Vxe,Vye,Xe,refElt)
        Tvelo=np.hstack((Te,Te+nOfNodes))
        for i in np.arange(len(Tvelo)):
            for j in np.arange(len(Tvelo)):
                C[Tvelo[i],Tvelo[j]] = C[Tvelo[i],Tvelo[j]] + Ce[i,j]  
    return C

def computeCmatrixNavierStokesSparse(V,X,T,nOfNodesP,refElt):
    # Initialise system matrix and rhs vector
    nOfNodes = X.shape[0]
    Vx=V[:nOfNodes]; Vy=V[nOfNodes:]
    nOfElements,nOfElementNodes = T.shape
    rowInd=np.zeros((nOfElements*(2*nOfElementNodes)**2,1),dtype=int); rowInd.shape=(len(rowInd),)
    colInd=np.zeros((nOfElements*(2*nOfElementNodes)**2,1),dtype=int); colInd.shape=(len(colInd),)
    valC=np.zeros((nOfElements*(2*nOfElementNodes)**2,1)); valC.shape=(len(valC),)
    # loop in elements to obtain the system of equations
    counter=0
    for e in np.arange(nOfElements):
        Te = T[e,:]
        Xe = X[Te,:]
        Vxe=Vx[Te]
        Vye=Vy[Te]
        Ce = elementCmatrixNS(Vxe,Vye,Xe,refElt)
        Tvelo=np.hstack((Te,Te+nOfNodes))
        for i in np.arange(len(Tvelo)):
            for j in np.arange(len(Tvelo)):
                rowInd[counter]=Tvelo[i]
                colInd[counter]=Tvelo[j]
                valC[counter]= Ce[i,j]
                counter=counter+1
    dim=nOfNodes*2+nOfNodesP
    matrix=coo_matrix((valC,(rowInd,colInd)),shape=(dim,dim))
    C= matrix.tocsc()
    return C


def elementCmatrixNS(Vxe,Vye,Xe,refElement):
    nOfElementNodes = (refElement.N).shape[1]
    nIP = refElement.nIP
    wIP = refElement.integrationWeights 
    N    = refElement.N
    Nxi  = refElement.Nxi 
    Neta = refElement.Neta
    zeros=np.zeros((1,nOfElementNodes))
    
    Ce = np.zeros((nOfElementNodes*2,nOfElementNodes*2))
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
        matN =  np.block([[N_ip,zeros],[zeros,N_ip]])
        a1 = N_ip@Vxe 
        a2 = N_ip@Vye
        aux = a1*Nx+a2*Ny
        matS =  np.block([[aux,zeros],[zeros,aux]])                                      
        Ce = Ce + ((np.transpose(matN))@matS)*dvolu
    return Ce
