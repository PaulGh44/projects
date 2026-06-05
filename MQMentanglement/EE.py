import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

##---Global parameters---##
m = 0.5 # mass in units of the lattice length a
N =1000 # size of the matrix considered for inverting
Lmin = 1 #Length of the minimal cut considered.
Lmax = 11 #Length of the maximal cut considered.

#Build the matrix K (kernel) defined as K_{ij} = \delta_{ij}(m^2+2)-(\delta_{ij+1}+\delta_{ij-1})
def Kbuilder(m:float = m, N:int = N):
    K = np.zeros((N,N))
    for i in range (N):
        for j in range (N):
            if (i==j):
                K[i,j] = m**2 +2
            elif (i==j+1):
                K[i,j] = -1
            elif (i==j-1):
                K[i,j] = -1
            else: 
                continue
    return K

#Build the covariant matrix square C_{|A} = (\sqrt{K}_{|A}\sqrt{K}^{-1}_{|A})/4
def Covariantmatrixsquare(L:int, m:float = m, N:int = N):
    leftendpoint = int((N-L)/2)
    rightendpoint = leftendpoint + L

    if (leftendpoint < 0 or rightendpoint > N):
        raise ValueError("The size of the cut is too large")
    
    squarerootK = linalg.sqrtm(Kbuilder(m,N))
    squarerootKinv = linalg.inv(squarerootK)
    Covariantsquare = np.zeros((L,L))

    for i in range (L):
        for j in range (L):
            entry = 0
            for k in range (L):
                entry += squarerootK[i+leftendpoint,k+leftendpoint] * squarerootKinv[k+leftendpoint,j+leftendpoint]
            
            Covariantsquare [i,j] = entry/4

    return Covariantsquare

#Compute the EE by using the eigenvalues of the Covariant matrix
def VonNeumannentropy(L:int, m:float = m, N:int = N):
    listeigeigenvalues = np.real(linalg.eigvals(Covariantmatrixsquare(L,m,N)))
    VNentropy = 0
    for k in range (L):
        if listeigeigenvalues[k] < 1/4:
            raise ValueError("Eigenvalue too small: some coefficients of the Boltzman entropy are negative")
        
        VNentropy += (np.sqrt(listeigeigenvalues[k]) + 1/2)*np.log(np.sqrt(listeigeigenvalues[k]) + 1/2) - (np.sqrt(listeigeigenvalues[k]) - 1/2)*np.log(np.sqrt(listeigeigenvalues[k]) - 1/2)

    return VNentropy

def VonNeumannentropy_as_a_function_of_cutsize(Lmin:int = Lmin, Lmax:int = Lmax, m:float = m, N:int = N):
    VonNeumannentropies = []
    for k in range (Lmin, Lmax + 1):
        VonNeumannentropies.append(VonNeumannentropy(k, m, N))

    return VonNeumannentropies

def plotVonNeumannentropy_as_a_function_of_cutsize(Lmin:int = Lmin, Lmax:int = Lmax, m:float = m, N:int = N):
    VonNeumannentropies = VonNeumannentropy_as_a_function_of_cutsize(Lmin, Lmax, m, N)
    list = [k for k in range(Lmin, Lmax+1)] 

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(list, VonNeumannentropies)
    ax.set_xlabel(r"$L$")
    ax.set_ylabel("$S_{VN}(L)$")
    ax.set_title("EE as a function of the cut")
    ax.grid(True)
    fig.tight_layout()
    
    return fig, ax

# ---------- MAIN/Tests ----------
if __name__ == "__main__":
    plotVonNeumannentropy_as_a_function_of_cutsize()
    plt.show()
