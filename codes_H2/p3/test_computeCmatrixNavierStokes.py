import numpy as np
from scipy.sparse import diags
from analyticalInfoCavityFlowStokes import * 
from PlottingFunctions import *
from ReferenceElement import *
from ErrorComputation import *
from MeshingFunctions import *
from SystemComputation import *
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
# This library enhances some plot options
from mpl_toolkits.mplot3d import Axes3D
# Plots are shown on a new window, where we can move and zoom
#matplotlib auto  

domain = np.array([0,1,0,1])

# Stokes reference element and computational mesh
referenceElementStokes=defineRefElementStokesQ2Q1()
nx = 2; ny = nx; 
X,T,Xp,Tp = UniformRectangleMeshStokesQ1Q2(domain,nx,ny)
plotMeshStokesQ2Q1(X,T,Xp,Tp,referenceElementStokes); plt.show()

# Nodal values for quadratic velocity fields
x=X[:,0]; x.shape=(len(x),1)
y=X[:,1]; y.shape=(len(y),1)
W=np.block([[x**2],[x*y]])
V=np.block([[x+y**2],[x*y]])

C=computeCmatrixNavierStokes(V,X,T,referenceElementStokes)
integral=(np.transpose(W))@(C@V)
print('analytical=0.82222222  numerical=',integral)



