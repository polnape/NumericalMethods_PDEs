import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as mtri
import scipy.io as sio


def rectangleMesh(dom,nx,ny,refElement): 
    elementType = refElement.type
    degree = refElement.degree
    
    x1 = dom[0]; x2 = dom[1]
    y1 = dom[2]; y2 = dom[3]
    
    npx = degree*nx+1
    npy = degree*ny+1
    npt = npx*npy
    x = np.linspace(x1,x2,npx)
    y = np.linspace(y1,y2,npy)
    x,y = np.meshgrid(x,y)
    x.shape = (npt,1)
    y.shape = (npt,1)
    X = np.block([x,y])

    if degree == 1:
        TInit = np.zeros((nx*ny,4),dtype=int)
        for i in np.arange(ny):
            for j in np.arange(nx):
                ielem = i*nx + j
                inode = i*npx + j
                TInit[ielem,:] = [inode, inode+1, inode+npx+1, inode+npx]
        if elementType == 0:
            T = TInit
        elif elementType == 1:
            T = np.concatenate((TInit[:,[3,0,1]],TInit[:,[3,1,2]]))
    elif degree == 2:
        TInit = np.zeros((nx*ny,9),dtype=int)
        for i in np.arange(ny):
            for j in np.arange(nx):
                ielem = i*nx + j
                inode = 2*i*npx + 2*j
                TInit[ielem,:] = [inode, inode+1, inode+2, inode+npx+2,inode+2*npx+2,inode+2*npx+1,inode+2*npx,inode+npx,inode+npx+1]
        if elementType == 0:
            T = TInit
        elif elementType ==1:
            T = np.concatenate((TInit[:,[1,2,3,4,5,9]],TInit[:,[1,9,5,6,7,8]]))
    else:
        print('Not implemented element')
        sys.exit()                    
    return X,T    

def ExcavationMeshes(degree,eltType,fineness,wallDistance):
    np.set_printoptions(threshold=1e6)
    if degree == 1:
        if eltType == 0:
            if fineness == 0:
                if wallDistance == 5:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/5zanjaQ1_0_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
                elif wallDistance == 10:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/10zanjaQ1_0_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
            elif fineness == 1:
                if wallDistance == 5:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/5zanjaQ1_1_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
                elif wallDistance == 10:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/10zanjaQ1_1_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
            elif fineness == 2:
                if wallDistance == 5:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/5zanjaQ1_2_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
                elif wallDistance == 10:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/10zanjaQ1_2_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
            elif fineness == 3:
                if wallDistance == 5:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/5zanjaQ1_3_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
                elif wallDistance == 10:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/10zanjaQ1_3_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
            T1 = T1[3:len(T1)]
            T1 = np.matrix(T1) 
            T = np.zeros((T1.shape[1]//4,4))
            for e in np.arange(1,T1.shape[1]//4+1):
                T[e-1,:] = np.array([T1[0,4*(e-1)],T1[0,4*(e-1)+1],T1[0,4*(e-1)+2],T1[0,4*(e-1)+3]])
        elif eltType == 1:
            if fineness == 0:
                if wallDistance == 5:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/5zanjaP1_0_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
                elif wallDistance == 10:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/10zanjaP1_0_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
            elif fineness == 1:
                if wallDistance == 5:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/5zanjaP1_1_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
                elif wallDistance == 10:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/10zanjaP1_1_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
            elif fineness == 2:
                if wallDistance == 5:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/5zanjaP1_2_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
                elif wallDistance == 10:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/10zanjaP1_2_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
            elif fineness == 3:
                if wallDistance == 5:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/5zanjaP1_3_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
                elif wallDistance == 10:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/10zanjaP1_3_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
            T1 = T1[3:len(T1)]
            T1 = np.matrix(T1) 
            T = np.zeros((T1.shape[1]//3,3))
            for e in np.arange(1,T1.shape[1]//3+1):
                T[e-1,:] = np.array([T1[0,3*(e-1)],T1[0,3*(e-1)+1],T1[0,3*(e-1)+2]])
    elif degree == 2:
        if eltType == 0:
            if fineness == 0:
                if wallDistance == 5:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/5zanjaQ2_0_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
                elif wallDistance == 10:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/10zanjaQ2_0_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
            elif fineness == 1:
                if wallDistance == 5:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/5zanjaQ2_1_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
                elif wallDistance == 10:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/10zanjaQ2_1_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
            elif fineness == 2:
                if wallDistance == 5:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/5zanjaQ2_2_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
                elif wallDistance == 10:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/10zanjaQ2_2_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
            elif fineness == 3:
                if wallDistance == 5:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/5zanjaQ2_3_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
                elif wallDistance == 10:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/10zanjaQ2_3_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
            T1 = T1[3:len(T1)]
            T1 = np.matrix(T1)
            T2 = np.zeros((T1.shape[1]//9,9)); T = np.zeros((T1.shape[1]//9,9))
            for e in np.arange(1,T1.shape[1]//9+1):
                T2[e-1,:] = np.array([T1[0,9*(e-1)],T1[0,9*(e-1)+1],T1[0,9*(e-1)+2],T1[0,9*(e-1)+3],T1[0,9*(e-1)+4],T1[0,9*(e-1)+5],T1[0,9*(e-1)+6],T1[0,9*(e-1)+7],T1[0,9*(e-1)+8]])
            T[:,0] = T2[:,1]; T[:,1] = T2[:,5]; T[:,2] = T2[:,2]; T[:,3] = T2[:,6]; T[:,4] = T2[:,3]; T[:,5] = T2[:,7]; T[:,6] = T2[:,0]; T[:,7] = T2[:,4]; T[:,8] = T2[:,8]
        elif eltType == 1:
            if fineness == 0:
                if wallDistance == 5:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/5zanjaP2_0_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
                elif wallDistance == 10:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/10zanjaP2_0_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
            elif fineness == 1:
                if wallDistance == 5:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/5zanjaP2_1_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
                elif wallDistance == 10:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/10zanjaP2_1_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
            elif fineness == 2:
                if wallDistance == 5:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/5zanjaP2_2_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
                elif wallDistance == 10:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/10zanjaP2_2_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
            elif fineness == 3:
                if wallDistance == 5:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/5zanjaP2_3_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
                elif wallDistance == 10:
                    matFile = sio.loadmat('Assignment1Professor/Meshes/10zanjaP2_3_readed.mat',variable_names=['elementFaceInfo', 'elemInfo','T','Tb_artificial','Tb_bottom','Tb_h1','Tb_h2','Tb_symmetry','Tb_wall','X'])
                    X1 = f"X: {matFile['X']}"
                    T1 = f"T: {matFile['T']}"
            T1 = T1[3:len(T1)]
            T1 = np.matrix(T1)
            T2 = np.zeros((T1.shape[1]//6,6)); T = np.zeros((T1.shape[1]//6,6))
            for e in np.arange(1,T1.shape[1]//6+1):
                T2[e-1,:] = np.array([T1[0,6*(e-1)],T1[0,6*(e-1)+1],T1[0,6*(e-1)+2],T1[0,6*(e-1)+3],T1[0,6*(e-1)+4],T1[0,6*(e-1)+5]])
            T[:,0] = T2[:,1]; T[:,1] = T2[:,5]; T[:,2] = T2[:,2]; T[:,3] = T2[:,6]; T[:,4] = T2[:,3]; T[:,5] = T2[:,7]; T[:,6] = T2[:,0]; T[:,7] = T2[:,4]; T[:,8] = T2[:,8]
    X1 = X1[3:len(X1)]
    X1 = np.matrix(X1)
    X = np.zeros((X1.shape[1]//2,2))
    for p in np.arange(1,X1.shape[1]//2+1):
        X[p-1,:] = np.array([X1[0,2*(p-1)],X1[0,2*(p-1)+1]])
    T = T.astype(int); T=T-1
    return X,T
