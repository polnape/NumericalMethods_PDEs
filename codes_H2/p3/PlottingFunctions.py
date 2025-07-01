import numpy as np
import sys
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as mtri

def plotMesh(X,T,refElement):
    elementType = refElement.type
    degree = refElement.degree
    if elementType == 0 and degree==1:
        Tvertex = T
        plotOrder = [0,1,2,3,0] 
    elif elementType == 0 and degree==2:
        Tvertex = T[:,refElement.vertexNodes]
        plotOrder = [0,1,2,3,0] 
    elif elementType == 1:
        Tvertex = T
        plotOrder = [0,1,2,0] 
    for e in np.arange(T.shape[0]):
        Te = Tvertex[e,:]
        Xe = X[Te,:]
        plt.plot(Xe[plotOrder,0], Xe[plotOrder,1], 'k')
    plt.plot(X[:,0], X[:,1],'*')   
    plt.show() 

def plotSlopes(x,y):
    for i in np.arange(len(x)-1):
        dx=(x[i+1]-x[i]); dy=(y[i+1]-y[i])
        slope=dy/dx
        xm=(x[i+1]+x[i])/2; ym=(y[i+1]+y[i])/2
        plt.text(xm+dx*0,ym+dy*0.2,round(slope,2))

def contourfPlot(u,X,T,):
    eltType = np.shape(T)[1]
    nnodes = np.shape(X)[0]
    nElts  = np.shape(T)[0]
    # Solution plot
    x = X[:,0] 
    y = X[:,1]
    u = u[:,0]
    # Contour plot
    if eltType == 4:
        Ttriangles = np.concatenate((T[:,[3,0,1]],T[:,[3,1,2]]))
    elif eltType == 9:
        Ttriangles = np.concatenate((T[:,[6,7,8]],T[:,[6,8,5]],T[:,[5,8,3]],T[:,[5,3,4]],T[:,[7,0,1]],T[:,[7,1,8]],T[:,[8,1,2]],T[:,[8,2,3]]))
    else:
        Ttriangles = T
    plt.tricontourf(x,y,u,30,triangles=Ttriangles,cmap = cm.coolwarm, vmin=u.min(), vmax=u.max(), origin='lower',
          extent=[x.min(), x.max(), y.min(), y.max()])
    plt.xlabel("x")
    plt.ylabel("y")        
    plt.colorbar()
    #plt.show()
    
def contourPlot(u,X,T,):
    eltType = np.shape(T)[1]
    nnodes = np.shape(X)[0]
    nElts  = np.shape(T)[0]
    # Solution plot
    x = X[:,0] 
    y = X[:,1]
    u = u[:,0]
    # Contour plot
    if eltType == 4:
        Ttriangles = np.concatenate((T[:,[3,0,1]],T[:,[3,1,2]]))
    elif eltType == 9:
        Ttriangles = np.concatenate((T[:,[6,7,8]],T[:,[6,8,5]],T[:,[5,8,3]],T[:,[5,3,4]],T[:,[7,0,1]],T[:,[7,1,8]],T[:,[8,1,2]],T[:,[8,2,3]]))
    else:
        Ttriangles = T
    plt.tricontour(x,y,u,30,triangles=Ttriangles,cmap = cm.coolwarm, vmin=u.min(), vmax=u.max(), origin='lower',
          extent=[x.min(), x.max(), y.min(), y.max()])
    plt.xlabel("x")
    plt.ylabel("y")        
    plt.colorbar()
    #plt.show()


def surfPlot(u,X,T):
    eltType = np.shape(T)[1]
    nnodes = np.shape(X)[0]
    nElts  = np.shape(T)[0]
    # Solution plot
    x = X[:,0] 
    y = X[:,1]
    u = u[:,0]
    if eltType==4:
        Ttriangles = np.concatenate((T[:,[3,0,1]],T[:,[3,1,2]]))
    elif eltType == 9:
         Ttriangles = np.concatenate((T[:,[6,7,8]],T[:,[6,8,5]],T[:,[5,8,3]],T[:,[5,3,4]],T[:,[7,0,1]],T[:,[7,1,8]],T[:,[8,1,2]],T[:,[8,2,3]]))
    else:
        Ttriangles = T
    # Surface plot
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    surf = ax.plot_trisurf(x,y,u,triangles = Ttriangles,cmap=cm.coolwarm)
    fig.colorbar(surf, shrink=0.5, aspect=5) # Add a color bar which maps values to colors.
    #plt.show()
    
def plotMeshStokesQ2Q1(X,T,Xp,Tp,refElement):
    Tvertex = T[:,refElement.vertexNodes]
    plotOrder = [0,1,2,3,0] 
    for e in np.arange(T.shape[0]):
        Te = Tvertex[e,:]
        Xe = X[Te,:]
        plt.plot(Xe[plotOrder,0], Xe[plotOrder,1], 'k')
    plt.plot(X[:,0], X[:,1],'*')   
    plt.plot(Xp[:,0], Xp[:,1],'o')  
    plt.axis('equal')
    plt.show() 
    
    
def boundaryConnectivityQ2Q1(X,T,Xp,Tp):
    #Identification of sides on the boundary using the linear mesh (Xp,Tp)
    faceNodes=np.array([[0,1],[1,2],[2,3],[3,0]])
    nOfElements=Tp.shape[0]
    nOfNodes=Xp.shape[0]
    A=np.zeros((nOfNodes,nOfNodes),dtype=int)
    for e in np.arange(nOfElements):
        for f in np.arange(4):
            A[Tp[e,faceNodes[f,0]],Tp[e,faceNodes[f,1]]]=1
    infoBoundaryFaces=np.zeros((nOfElements*4,2),dtype=int) #number of element and local face for boundary faces
    k=0
    for e in np.arange(nOfElements):
        for f in np.arange(4):
            i=Tp[e,faceNodes[f,0]]
            j=Tp[e,faceNodes[f,1]]
            if A[j,i]==0:
                infoBoundaryFaces[k,0]=e
                infoBoundaryFaces[k,1]=f
                k=k+1
    infoBoundaryFaces=infoBoundaryFaces[:k,:]
    nOfBoundaryFaces=k
    #Connectivity matrix of the Q2 boundary
    Tboundary=np.zeros((nOfBoundaryFaces,3),dtype=int)
    faceNodes=np.array([[0,1,2],[2,3,4],[4,5,6],[6,7,0]])
    for k in np.arange(nOfBoundaryFaces):
        e=infoBoundaryFaces[k,0]
        f=infoBoundaryFaces[k,1]
        Tf=T[e,faceNodes[f,:]]
        Tboundary[k,:]=Tf
    return Tboundary
        
        
def computeStreamFunction(vx,vy,X,T,Tboundary,refElt):
    # Initialise system matrix and rhs vector
    nOfNodes = X.shape[0]
    nOfElements,nOfElementNodes = T.shape
    kk=nOfElements*nOfElementNodes**2
    rowInd=np.zeros((kk,1),dtype=int); rowInd.shape=(len(rowInd),)
    colInd=np.zeros((kk,1),dtype=int); colInd.shape=(len(colInd),)
    valK=np.zeros((kk,1)); valK.shape=(len(valK),)
    counter=0
    f = np.zeros((nOfNodes,1))  
    # loop in elements to obtain the system of equations
    for e in np.arange(nOfElements):
        Te = T[e,:]
        Xe = X[Te,:]
        vxe=vx[Te]; vye=vy[Te]
        Ke,fe = elementalComputationsStreamFunction(vxe,vye,Xe,refElt)
        for i in np.arange(len(Te)):
            f[Te[i]] = f[Te[i]] + fe[i]
            for j in np.arange(len(Te)):
                rowInd[counter]=Te[i]
                colInd[counter]=Te[j]
                valK[counter]= Ke[i,j]
                counter=counter+1
    # loop in elements of the boundary 
    wIP =refElt.integrationWeights1D; nIP=len(wIP)
    N = refElt.N1D
    Nxi = refElt.N1Dxi 
    nOfElements,nOfElementNodes=Tboundary.shape
    for e in np.arange(nOfElements):
        Te=Tboundary[e,:]
        Xe = X[Te,:]
        vxe=vx[Te]; vye=vy[Te]
        fe=np.zeros(nOfElementNodes)
        for ip in np.arange(nIP):
            t=Nxi[ip,:]@Xe
            normt= np.linalg.norm(t);  
            nx=t[1]/normt; ny=-t[0]/normt
            dl=wIP[ip]*abs(normt)
            N_ip=N[ip,:];
            uxIP=N_ip@vxe
            uyIP=N_ip@vye
            fe=fe + N_ip*((nx*uyIP-ny*uxIP)*dl)         
        for i in np.arange(len(Te)):
            f[Te[i]] = f[Te[i]] + fe[i]
    matrix=coo_matrix((valK,(rowInd,colInd)),shape=(nOfNodes,nOfNodes))
    K= matrix.tocsc()
    #set constant for the stream function (last node=0)
    #K=K[:-1,:-1]
    #f=f[:-1]
    phi = spsolve(K,f); phi.shape=(len(phi),1)
    #phi=np.append(phi,0)
    phi.shape=(len(phi),1)
    return phi


def elementalComputationsStreamFunction(vxe,vye,Xe,refElement):
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
        fe = fe + N_ip*((Ny@vxe-Nx@vye)*dvolu)
    return Ke,fe 

    
    
                                        
         
                
                
    
    
