import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from femFunctions1D import *

#Definition of source term s=-u_xx and Dirichlet values at both end points
def sourceTerm(x):
    return -(100*x**4-120*x**3-8*x**2+28*x+2)*np.exp(10*x)
ua=0
ub=0

degree=1
referenceElement=defineRefElement1D(degree)


nOfElements = 10
he = 1/nOfElements
X,T=uniform1dMesh(0,1,nOfElements,degree)
# X, T = Xnew, Tnew

# número de elementos es el número de filas
nOfElements = T.shape[0]
# número de filas --> número de nodos totales
nOfNodes=X.shape[0]
# número de nodos por elemento
nOfElementNodes=T.shape[1]
# k es NnodesxNnodes
K=np.zeros((nOfNodes,nOfNodes)) #full matrix, sparse storage in 2D...
f=np.zeros((nOfNodes,1))
#Loop in elements for matrix computation
for e in np.arange(nOfElements):
    he=X[e+1]-X[e]
    #Elemental matrix for 1D linear element
    Ke=np.array([[1,-1],[-1,1]])*(1/he)
    #Matrix assembly
    for i in np.arange(nOfElementNodes):
        for j in np.arange(nOfElementNodes):
            K[T[e,i],T[e,j]]=K[T[e,i],T[e,j]]+Ke[i,j]
#Loop in elements for rhs vector computation
xi=referenceElement.integrationPoints
w=referenceElement.integrationWeights
nOfIntegrationPoints=xi.shape[0]
for e in np.arange(nOfElements):
    he=X[e+1]-X[e]
    #Elemental vector with numerical integration
    fe=np.zeros((1,nOfElementNodes))
    for k in np.arange(nOfIntegrationPoints):
        Nk=referenceElement.N[k,:]
        # epsilon del cambio de varaibles
        xk=(X[e]+X[e+1])/2+he*xi[k] #linear transformation
        # El he/2 viene de el cambio de gauss-legendre cuadrature points
        fe=fe+Nk*(sourceTerm(xk)*(he/2)*w[k])
    fe.shape=(nOfElementNodes,1)
    #Vector assembly
    for i in np.arange(nOfElementNodes):
        f[T[e,i]]= f[T[e,i]] + fe[i]
    
#Dirichlet boundary conditions at both ends -> system reduction
aux=K[1:-1,0]*ua+K[1:-1,-1]*ub
aux.shape=(nOfNodes-2,1)
freduced=f[1:-1]-aux
Kreduced=K[1:-1,1:-1]

#System solution
solution=np.linalg.solve(Kreduced,freduced)

u=np.zeros((nOfNodes,1))
u[0]=ua
u[1:-1]=solution
u[-1]=ub

#Postprocess
plt.plot(X,u,'-o')
plt.xlabel('x')
plt.ylabel('u')
plt.show()


#Derivative in each element
hs = X[1:]-X[:-1]
dudx=(u[1:]-u[:-1])/hs
plt.xlabel('x')
plt.ylabel('du/dx')
plotElementalConstants(X,T,dudx)

#Computation of the L2 gradient smoothing (L2 projection of du/dx in the FE space)



# Computation of the L2 gradient smoothing (L2 projection of du/dx in the FE space)
# Gradient smoothing - L2 projection of the gradient
Me = np.array([[2, 1], [1, 2]])*(he/6)   # Matriz de masas escalada por el tamaño del elemento
Be_elem = (1 / 2) * np.array([[-1, 1], [-1, 1]])  # Matriz de gradiente elemental

# Inicialización de las matrices globales
M = np.zeros((nOfNodes, nOfNodes))  # Matriz global de masas
B = np.zeros((nOfNodes, nOfNodes))  # Vector global para la suavización del gradiente

# Ensamblaje de la matriz de masas M
for e in np.arange(nOfElements):
    he = X[e+1] - X[e]  # Tamaño del elemento (1D)
    
    # Ensamblaje de la matriz M (matriz de masas)
    for i in np.arange(nOfElementNodes):
        for j in np.arange(nOfElementNodes):
            M[T[e, i], T[e, j]] += Me[i, j] # Escalado por el tamaño del elemento

# Integración numérica sobre cada elemento para ensamblar el vector B
xi = referenceElement.integrationPoints  # Puntos de integración
w = referenceElement.integrationWeights  # Pesos de integración
nOfIntegrationPoints = xi.shape[0]  # Número de puntos de integración
for e in np.arange(nOfElements):
    he = X[e+1] - X[e]  # Tamaño del elemento (1D)
    
    # Ensamblaje de la matriz M (matriz de masas)
    for i in np.arange(nOfElementNodes):
        for j in np.arange(nOfElementNodes):
            B[T[e, i], T[e, j]] += Be_elem[i, j] 
            
B = np.dot(B,u)


# DEJO ESTO COMENTADO PARA RESALTAR
# NO PUEDO APLICAR CONDICIONES DE CONTORNO DE U SOBRE UNA MATRIZ (q) 
# QUE REPRESNTA LA DERIVADA
#Dirichlet boundary conditions at both ends -> system reduction
# aux=M[1:-1,0]*ua+M[1:-1,-1]*ub
# aux.shape=(nOfNodes-2,1)
# Breduced=B[1:-1]-aux
# Mreduced=M[1:-1,1:-1]
# Solución del sistema reducido
solution = np.linalg.solve(M, B)

q = solution
# q en este caso, representa la derivada para este apartado
# q=np.zeros((nOfNodes,1))
# q[0]=ua
# q[1:-1]=solution
# q[-1]=ub

#Postprocess
plt.plot(X,q,'-o', label = "q", color = "royalblue")
plt.xlabel('X')
plt.ylabel('q')



#Derivative in each element
hs = X[1:]-X[:-1]
dudx=(u[1:]-u[:-1])/hs
plt.xlabel('X')
plt.ylabel('du/dx')
plt.legend()
plotElementalConstants(X,T,dudx)
plt.show()




#%%
#ERROR ZZ 
# TENGO uh como solución de BVP, de aquí saco la derivada de cada intervalo
# Inicialización del vector de errores por elemento
Xnew, Tnew = X, T
nOfNodes=Xnew.shape[0]
nOfElements = Tnew.shape[0]
nOfElementNodes=Tnew.shape[1]
Knew=np.zeros((nOfNodes,nOfNodes))
ZZ_error_per_element = np.zeros(nOfElements)

# Loop sobre los elementos para calcular el error en cada uno
for e in np.arange(nOfElements):
    Te = T[e, :]  # Nodos del elemento e
    Xe = X[Te]  # Coordenadas de los nodos del elemento e
    ue = u[Te]  # Valores de la solución en los nodos del elemento e
    qe = q[Te]  # Valores del gradiente suavizado en los nodos del elemento e
    # qe es un vector de dos dimensiones
    
    he = Xe[1] - Xe[0]  # Longitud de nuestro elemento
    
    # Calcular la derivada numérica en el elemento (es constante en cada elemento)
    dudx_e = (ue[1] - ue[0]) / he
    
    # Calcular el error en cada punto de integración usando la cuadratura
    # La q nos da el valor de los integration points, queremos hacer la interpolació para ver el error
    # para cualquier valor de x
    ZZ_error_element = 0  # Inicializar el error para cada elemento
    for k in range(nOfIntegrationPoints):
        N_k = referenceElement.N[k, :]  # Funciones de forma en el punto de integración k
        xk = (Xe[0] + Xe[1]) / 2 + he * xi[k] / 2  # cambio de variable (epsilon en los apuntes)
        
        # Valor interpolado del gradiente suavizado 
        # qinterp = N1*qe1+ N2*qe2+...
        q_interp = N_k @ qe
        
        # Contribución al error ZZ en el punto de integración k
        # Para cada elemento se suman los errores de los integration points
        ZZ_error_element += (q_interp - dudx_e)**2 * (he / 2) * w[k]
    
   
    # hay q hacer la raiz cuadrada
    # print(ZZ_error_element**(1/2))
    ZZ_error_per_element[e] = ZZ_error_element
    

# FALTA LA RAIZ CUADRADAAA
ZZ_error_per_element_sqrt = np.sqrt(ZZ_error_per_element)

# Plots, de elemento en elemento
# Esto me lo hace continuo, REMARK de q para cada elemento es un número constante
# element_centers = (X[:-1] + X[1:]) / 2  # Centros de los elementos (para la gráfica)
# plt.figure()
# plt.plot(element_centers, ZZ_error_per_element, '-o', label="Error ZZ por elemento")
# plt.xlabel('Centro del elemento')
# plt.ylabel('Error ZZ')
# plt.title('Distribución del error ZZ por elemento')
# plt.legend()
# plt.grid(True)
# plt.show()

# Error total
ZZ_error_total = np.sqrt(np.sum(ZZ_error_per_element_sqrt**2))
print("Error ZZ total:", ZZ_error_total)

plt.figure()

plt.title("ZZ error per element")
plt.ylabel("Error per element", fontsize = 14)
plt.xlabel("X", fontsize = 14)
plotElementalConstants(X,T, ZZ_error_per_element_sqrt)
   

# REDEFINE MESH 
# tengo que hacer un bucle TOTAL, que me itere el error de los elementos y solo pare cuando todos sean menores que un epsilon dado
# elementsToRefine=np.array([1,4])
# Xnew,Tnew=refineMesh1DLinear(X,T,elementsToRefine)
# print('Xnew=',Xnew)

# Vigilar que el numero de elementos irá cambiando segun nOfElements = T.shape[0] 
epsilon =0.01
# initial mesh
Xnew, Tnew = X, T
iteration = 0
rho_total = []
error_element = ZZ_error_per_element
list_of_index_errors = []

# rho per element inicial
rho_per_element = np.zeros(len(error_element)) 
for index in range(len(error_element)):
    #error de cada elemento
    element = error_element[index]
    
    
    if index < (len(Xnew) - 1):  # Asegurar que no se sobrepase el índice
        he = Xnew[index + 1] - Xnew[index]  # Longitud del elemento actual
        he = he.item()
        densidad = (element**2 / he)
        # print(densidad.type)
        rho_per_element[index] = densidad # Densidad de error para el elemento actual
        
        
        # Usar la densidad de error como criterio de refinamiento
        if rho_per_element[index]>epsilon:
            list_of_index_errors.append(index)

list_of_index_array = np.array(list_of_index_errors)
rho_total_sum = np.sum(rho_per_element)

#rho total de cada iteración
# rho_total.append(np.log10(rho_total_sum))
rho_total.append(rho_total_sum)
# plot de la rho, antes de volver a refinar la malla
plt.figure()
plt.title("rho iteration 0")
plt.xlabel("X", fontsize = 14)
plt.ylabel(r'$\rho_e$', fontsize = 14)
plotElementalConstants(Xnew, Tnew, rho_per_element)

plt.show()

iteration = 0
all_log_error_per_element = []
xnew_total = []
tnew_total = []
xnew_total.append(X)
tnew_total.append(T)
# all_log_error_per_element.append(np.log10(rho_per_element))
all_log_error_per_element.append(rho_per_element)
#%%
# print(q[:8])
# hay algún erro en la q pq me cambia para elementos que no me ha refinado
Xnew = X
Tnew= T
while iteration < 4:
    # refino la malla con la lista de antes, la cual la iré actualizando    
    
    # print(list_of_index_array)
    Xnew, Tnew = refineMesh1DLinear(Xnew, Tnew, list_of_index_array)
    nOfElements = Tnew.shape[0]
    # número de filas --> número de nodos totales
    nOfNodes=Xnew.shape[0]
    # número de nodos por elemento
    nOfElementNodes=Tnew.shape[1]
    # k es NnodesxNnodes
    Knew=np.zeros((nOfNodes,nOfNodes)) #full matrix, sparse storage in 2D...
    fnew=np.zeros((nOfNodes,1))
    #Loop in elements for matrix computation
    for e in np.arange(nOfElements):
        he=Xnew[e+1]-Xnew[e]
        #Elemental matrix for 1D linear element
        Ke=np.array([[1,-1],[-1,1]])*(1/he)
        #Matrix assembly
        for i in np.arange(nOfElementNodes):
            for j in np.arange(nOfElementNodes):
                Knew[Tnew[e,i],Tnew[e,j]]=Knew[Tnew[e,i],Tnew[e,j]]+Ke[i,j]
    #Loop in elements for rhs vector computation
    xi=referenceElement.integrationPoints
    w=referenceElement.integrationWeights
    nOfIntegrationPoints=xi.shape[0]
    for e in np.arange(nOfElements):
        he=Xnew[e+1]-Xnew[e]
        #Elemental vector with numerical integration
        fe=np.zeros((1,nOfElementNodes))
        for k in np.arange(nOfIntegrationPoints):
            Nk=referenceElement.N[k,:]
            # epsilon del cambio de varaibles
            xk=(Xnew[e]+Xnew[e+1])/2+he*xi[k] #linear transformation
            # El he/2 viene de el cambio de gauss-legendre cuadrature points
            fe=fe+Nk*(sourceTerm(xk)*(he/2)*w[k])
        fe.shape=(nOfElementNodes,1)
        #Vector assembly
        for i in np.arange(nOfElementNodes):
            fnew[Tnew[e,i]]= fnew[Tnew[e,i]] + fe[i]
        
    #Dirichlet boundary conditions at both ends -> system reduction
    aux=Knew[1:-1,0]*ua+Knew[1:-1,-1]*ub
    aux.shape=(nOfNodes-2,1)
    freduced=fnew[1:-1]-aux
    Kreduced=Knew[1:-1,1:-1]
    
    #System solution
    solution=np.linalg.solve(Kreduced,freduced)
    
    u=np.zeros((nOfNodes,1))
    u[0]=ua
    u[1:-1]=solution
    u[-1]=ub
     

    
    print(u[:8])
    
   
    # plt.plot(Xnew,u,'-o')
    # plt.xlabel('x')
    # plt.ylabel('u')
    # plt.show()

    # El error no está en la u, si no en la q 
    #Postprocess
    
   
    #Derivative in each element
    dudx = np.zeros(nOfElements)
    for e in range(nOfElements):
        he = Xnew[e+1] - Xnew[e]  # Tamaño del elemento (1D)
        dudx[e] = (u[Tnew[e, 1]] - u[Tnew[e, 0]]) / he
    # plt.xlabel('x')
    # plt.ylabel('du/dx')
    # plotElementalConstants(X,T,dudx)
      # Matriz de masas escalada por el tamaño del elemento
    Be_elem = (1 / 2) * np.array([[-1, 1], [-1, 1]])  # Matriz de gradiente elemental

    # Inicialización de las matrices globales
    M = np.zeros((nOfNodes, nOfNodes))  # Matriz global de masas
    B = np.zeros((nOfNodes, nOfNodes))  # Vector global para la suavización del gradiente

    # Ensamblaje de la matriz de masas M
    for e in np.arange(nOfElements):
        he = Xnew[e+1] - Xnew[e]  # Tamaño del elemento (1D)
        Me = np.array([[2, 1], [1, 2]])*(he/6)   # Escalado de la matriz de masa elemental por el tamaño del elemento
        
        # Ensamblaje de la matriz M (matriz de masas) para el elemento actual
        for i in np.arange(nOfElementNodes):
            for j in np.arange(nOfElementNodes):
                M[Tnew[e, i], Tnew[e, j]] += Me[i, j]
    # Integración numérica sobre cada elemento para ensamblar el vector B
    xi = referenceElement.integrationPoints  # Puntos de integración
    w = referenceElement.integrationWeights  # Pesos de integración
    nOfIntegrationPoints = xi.shape[0]  # Número de puntos de integración
    for e in np.arange(nOfElements):
        he = Xnew[e+1] - Xnew[e]  # Tamaño del elemento (1D)
        
        # Ensamblaje de la matriz M (matriz de masas)
        for i in np.arange(nOfElementNodes):
            for j in np.arange(nOfElementNodes):
                B[Tnew[e, i], Tnew[e, j]] += Be_elem[i, j] 
           
    # si la u no es igual, pues la q claro que me cambiará
    B = np.dot(B,u)


    solution = np.linalg.solve(M, B)

    q = solution
    # print(q[:8])
    

    
    

    # GRADIENT SMOOTHING
    # # Postprocess
    # plt.plot(Xnew,q,'-o')
    # plt.xlabel('x')
    # plt.ylabel('q')
    # #Derivative in each element
    # hs = Xnew[1:]-Xnew[:-1]
    # dudx=(u[1:]-u[:-1])/hs
    # plt.xlabel('x')
    # plt.ylabel('du/dx')
    # plotElementalConstants(Xnew,Tnew,dudx)
    
    #reinicio el erro por elemento
    error_element = np.zeros(nOfElements)
    rho_per_element = np.zeros(nOfElements)
    for e in np.arange(nOfElements):
        Te = Tnew[e, :]
        Xe = Xnew[Te]
        ue = u[Te]
        qe = q[Te]
        he = Xe[1] - Xe[0]
    
        dudx_e = (ue[1] - ue[0]) / he
        ZZ_error_element = 0
    
        for k in range(nOfIntegrationPoints):

            # ZZ_error_element +=  (qe[k]-dudx_e)**2*w[k]*(he/2)
            N_k = referenceElement.N[k, :]
            xk = (Xe[0] + Xe[1]) / 2 + he * xi[k] / 2
            q_interp = N_k @ qe
            ZZ_error_element += (q_interp - dudx_e)**2 * (he / 2) * w[k]
    
        # error_element[e] = np.sqrt(ZZ_error_element)
        error_element[e] = (ZZ_error_element)
        
    plt.figure()
    # plt.title("ZZ error per element")
    plt.ylabel("Error per element", fontsize = 14)
    plt.xlabel("X", fontsize = 14)
    plotElementalConstants(Xnew,Tnew, (error_element))  

        

    # FALTA LA RAIZ CUADRADAAA
    # error_element = np.sqrt(error_element)
    # print(error_element[:8])

    # Plots, de elemento en elemento
    # Esto me lo hace continuo, REMARK de q para cada elemento es un número constante
    element_centers = (Xnew[:-1] + Xnew[1:]) / 2  # Centros de los elementos (para la gráfica)
    
    # nueva lista a refinar
    list_of_index_errors = []

    
    
    
    rho_per_element = np.zeros(nOfElements)
    for index in range(nOfElements):
        element = error_element[index]
        he = Xnew[index + 1] - Xnew[index]
        rho_per_element[index] = (element / he)
        if rho_per_element[index] > epsilon:
            list_of_index_errors.append(index)
    list_of_index_array = np.array(list_of_index_errors)
    # print(rho_per_element[:8])
    rho_total_sum = np.sum(rho_per_element)
    #rho total de cada iteración
    xnew_total.append(Xnew)
    tnew_total.append(Tnew)
    # all_log_error_per_element.append(np.log10(rho_per_element))
    all_log_error_per_element.append(rho_per_element)
    

    # Error total
    # ZZ_error_total = np.sqrt(np.sum(error_element**2))
    # print("Error ZZ total:", ZZ_error_total)

    # plt.figure()
    # plt.title("Plot por elemental constant")
    # plt.ylabel("Error per element")
    # plt.xlabel("x")
    # plt.title(f"Iteration {iteration+1}")
    # plotElementalConstants(Xnew,Tnew, rho_per_element)
    
    # plt.plot(Xnew,u,'-o')
    # plt.xlabel('x')
    # plt.ylabel('u')
    # plt.show()
    
        

    iteration = iteration+1
    # print("first iteration completed")



#%%
# Graficar log10(error_per_element) para cada iteración en el mismo plot
import matplotlib.pyplot as plt
import numpy as np

# Configura la figura para el gráfico combinado
plt.figure()

# Define una paleta de colores para diferenciar cada iteración
colors = plt.cm.viridis(np.linspace(0, 1, len(xnew_total)))

# Bucle para graficar cada iteración
for i in range(1, len(xnew_total)):
    X = xnew_total[i]
    T = tnew_total[i]
    u = all_log_error_per_element[i]
    u = np.log10(u)
    
    nOfElements = T.shape[0]
    
    # Graficar los elementos en el dominio espacial para cada iteración
    for e in np.arange(nOfElements):
        plt.plot([X[e], X[e+1]], [u[e], u[e]], color=colors[i], label=f'Iteración {i+1}' if e == 0 else "")
        
# Configuración final del gráfico
plt.xlabel("X", fontsize = 14)
plt.ylabel(r'$log_{10}$($\rho_e$)', fontsize = 14)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0)

plt.show()


# x_iteration = [number for number in np.arange(0, iteration,1)]
# plt.figure()
# plt.plot(rho_total, x_iteration, marker = "o")
    



#%%

