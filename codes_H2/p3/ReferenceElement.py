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
        refElement.vertexNodes = np.array([0,1,2,3])
        a = np.sqrt(1/3)
        z = np.array([[-a,-a], [a,-a], [a,a], [-a,a]])
        refElement.integrationPoints = z
        refElement.integrationWeights = np.array([1,1,1,1])
        refElement.nIP = 4
        xi = z[:,[0]]; eta = z[:,[1]]
        refElement.N = 1/4*np.block([(xi-1)*(eta-1), -(xi+1)*(eta-1), (xi+1)*(eta+1), -(xi-1)*(eta+1)])
        refElement.Nxi = 1/4*np.block([eta-1, -eta+1, eta+1, -eta-1])
        refElement.Neta = 1/4*np.block([xi-1, -xi-1, xi+1, -xi+1])
    elif elementType == 0 and degree == 2:
        refElement.nOfElementNodes = 9
        refElement.nodesCoord = np.array([[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0],[0,0]])
        refElement.vertexNodes = np.array([0,2,4,6])
        #nGaussPoints1D = 5;  z_aux = np.array([0,(1/3)*np.sqrt(5-2*np.sqrt(10/7)),-(1/3)*np.sqrt(5-2*np.sqrt(10/7)),(1/3)*np.sqrt(5+2*np.sqrt(10/7)),-(1/3)*np.sqrt(5+2*np.sqrt(10/7))]);     w_aux = np.array([128/225,(322+13*np.sqrt(70))/900,(322+13*np.sqrt(70))/900,(322-13*np.sqrt(70))/900,(322-13*np.sqrt(70))/900])
        z_aux = np.array([-np.sqrt(3/5),0,np.sqrt(3/5)]);
        nGaussPoints1D = 3
        w_aux = np.array([5/9,8/9,5/9])
        z = np.zeros((nGaussPoints1D**2,2))
        w = np.zeros((nGaussPoints1D**2,1))
        iGauss = 0
        for i in np.arange(nGaussPoints1D):
            for j in np.arange(nGaussPoints1D):
                z[iGauss,:] = np.array([[z_aux[i]],[z_aux[j]]]).T
                w[iGauss,0] = w_aux[i]*w_aux[j]
                iGauss = iGauss+1 
        refElement.integrationPoints = z
        refElement.integrationWeights = w
        refElement.nIP = nGaussPoints1D**2
        xi = z[:,[0]]; eta = z[:,[1]]
        refElement.N = np.block([(1/4)*xi*(xi-1)*eta*(eta-1),(1/2)*(1+xi)*(1-xi)*eta*(eta-1),(1/4)*xi*(1+xi)*eta*(eta-1),(1/2)*xi*(1+xi)*(1+eta)*(1-eta),(1/4)*xi*(1+xi)*eta*(1+eta),(1/2)*(1+xi)*(1-xi)*eta*(1+eta),(1/4)*xi*(xi-1)*eta*(1+eta),(1/2)*xi*(xi-1)*(1+eta)*(1-eta),(1+xi)*(1-xi)*(1+eta)*(1-eta)])
        refElement.Nxi = np.block([(1/2)*(xi-1/2)*eta*(eta-1),-xi*eta*(eta-1),(1/2)*(xi+1/2)*eta*(eta-1),(xi+1/2)*(1+eta)*(1-eta),(1/2)*(xi+1/2)*eta*(1+eta),-xi*eta*(1+eta),(1/2)*(xi-1/2)*eta*(1+eta),(xi-1/2)*(1+eta)*(1-eta),-2*xi*(1+eta)*(1-eta)])
        refElement.Neta = np.block([(1/2)*xi*(xi-1)*(eta-1/2),(1+xi)*(1-xi)*(eta-1/2),(1/2)*xi*(1+xi)*(eta-1/2),-xi*(1+xi)*eta,(1/2)*xi*(1+xi)*(eta+1/2),(1+xi)*(1-xi)*(eta+1/2),(1/2)*xi*(xi-1)*(eta+1/2),-xi*(xi-1)*eta,-2*(1+xi)*(1-xi)*eta])
    elif elementType == 1 and degree == 1:
        refElement.nOfElementNodes = 3
        refElement.nodesCoord = np.array([[0,1],[0,0], [1,0]])
        a = 1/2; b = 1/6
        z = np.array([[a,a],[a,0], [0,a]])
        refElement.integrationPoints = z
        refElement.integrationWeights = np.array([b,b,b])
        refElement.nIP = 3
        xi = z[:,[0]]; eta = z[:,[1]]
        zeroMat = np.reshape(np.zeros(len(xi)),(len(xi),1))
        oneMat = np.reshape(np.ones(len(xi)),(len(xi),1))
        refElement.N = np.block([eta,1-xi-eta, xi])
        refElement.Nxi = np.block([zeroMat,-oneMat, oneMat])
        refElement.Neta = np.block([oneMat,-oneMat, zeroMat])
    else:
        print('Not implemented element')
        sys.exit() 
    if degree == 1:
        a = np.sqrt(1/3)
        xi = np.array([a,a])
        xi.shape=(2,1)
        refElement.integrationPoints1D = xi
        refElement.integrationWeights1D = np.array([1,1,1,1])
        refElement.N1D = 1/2*np.block([1-xi,1+xi])
        refElement.N1Dxi = (1/2)*np.array([[-1,1],[-1,1]])
    elif degree == 2:
        xi = np.array([-np.sqrt(3/5),0,np.sqrt(3/5)])
        xi.shape=(3,1)
        refElement.integrationPoints1D = xi
        refElement.integrationWeights1D = np.array([5/9,8/9,5/9])
        refElement.N1D = np.block([(xi**2-xi)/2,1-xi**2,(xi+xi**2)/2])
        refElement.N1Dxi = np.block([(2*xi-1)/2,-2*xi,(1+2*xi)/2])
    else:
        print('Not implemented element')
        sys.exit() 
            
        
    return refElement


def defineRefElementStokesQ2Q1():
    refElementStokes = defineRefElement(0,2)
    refElementP = defineRefElement(0,1)
    refElementStokes.nodesCoordPressure=refElementP.nodesCoord
    #Q1 basis functions at Q2 integration points
    z=refElementStokes.integrationPoints; xi = z[:,[0]]; eta = z[:,[1]]
    refElementStokes.Npressure=1/4*np.block([(xi-1)*(eta-1), -(xi+1)*(eta-1), (xi+1)*(eta+1), -(xi-1)*(eta+1)])
    return refElementStokes
