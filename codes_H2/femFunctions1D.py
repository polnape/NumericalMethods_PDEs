import numpy as np
import sys
import matplotlib.pyplot as plt

def defineRefElement1D(degree):
    class refElement:
        pass
    refElement.degree = degree
    if degree == 1:
        refElement.nOfElementNodes = 2
        refElement.nodesCoord = np.array([-1,1])
        a = np.sqrt(1/3)
        xi = np.array([[-a],[a]])
        refElement.integrationPoints = xi
        refElement.integrationWeights = np.array([1,1])
        refElement.nIP = 2
        refElement.N = np.block([(1-xi)/2,(1+xi)/2])
        refElement.Nxi = np.array([[-1,1],[-1,1]])
#    elif degree == 2:
    else:
        print('Not implemented element')
        sys.exit()   
        
    return refElement
    

def uniform1dMesh(a,b,nOfElements,degree): 
    X = np.linspace(a,b,nOfElements+1)
    X.shape = (nOfElements+1,1)

    if degree == 1:
        aux=np.arange(0,nOfElements)
        aux.shape=(nOfElements,1)
        T=np.block([aux,aux+1])
    else:
        print('Not implemented element')
        sys.exit()   
                 
    return X,T    
            
def plotElementalConstants(X,T,u):
    nOfElements=T.shape[0]
    # print("Tamaño de X:", len(X))
    # print("Tamaño de u:", len(u))
    for e in np.arange(nOfElements):
        plt.plot([X[e],X[e+1]],[u[e],u[e]],'k-')
        # print("Índice e:", e)
        # print("Intentando acceder a X[{}] y X[{}]".format(e, e+1))
    
    
    plt.show()
    
#Splits in 2 elements the elements in the array elementsToRefine        
def refineMesh1DLinear(X,T,elementsToRefine):    
    n=elementsToRefine.shape[0]
    nOfNodesX=X.shape[0]
    Xnew=np.zeros((nOfNodesX+n,1))
    Xnew[:nOfNodesX]=X
    aux=np.copy(elementsToRefine)
    for k in np.arange(n):
        e=aux[k]
        
        xmid=(Xnew[e+1]+Xnew[e])/2
        Xnew[e+1:]=Xnew[e:-1]
        Xnew[e+1]=xmid
        aux=aux+1
    nOfElements=nOfNodesX+n-1
    aux=np.arange(0,nOfElements)
    aux.shape=(nOfElements,1)
    Tnew=np.block([aux,aux+1])        
    return Xnew,Tnew
        
        