#Packages
from dataclasses import dataclass
from mpmath import mp, barnesg
import numpy as np
import matplotlib.pyplot as plt

# -------- Global settings --------
mp.dps = 20 #be careful with the precision, if the precision is not enough you might raise "Zero Division error"
DEMO = True  # True for Demo mode

# -------- Parameters container --------
@dataclass(frozen=True)
class ParametersPentagon:
    Po: mp.mpc
    Pq: mp.mpc
    P1: mp.mpc
    P2: mp.mpc
    P3: mp.mpc
    P4: mp.mpc
    P5: mp.mpc
    Pt: mp.mpc
    Pu: mp.mpc
@dataclass(frozen=True)
class ParametersIdempotency:
    Ps: mp.mpc
    Psprime: mp.mpc
    P1: mp.mpc
    P2: mp.mpc
    P3: mp.mpc
    P4: mp.mpc
    
    
@dataclass
class Theory:
    m: int
    n: int
    Lambda: mp.mpf = mp.mpf(2.9) # cutoff (for later integration)

    @property
    def b(self):
        return mp.sqrt(mp.mpf(self.m) / mp.mpf(self.n))

    @property
    def s(self):
        return mp.sqrt(mp.mpf(self.m) * mp.mpf(self.n))


# -------- Demo / main params --------
if DEMO:
    parPenta = ParametersPentagon(
        Po=mp.mpc(0.001+0.1j), Pq=mp.mpc(0.001-0.5j), 
        P1=mp.mpc(0.001+0.2j), P2=mp.mpc(0.001+0.6j), 
        P3=mp.mpc(0.001+0.3j), P4=mp.mpc(0.001+0.7j), 
        P5=mp.mpc(0.001-0.4j), Pt=mp.mpc(0.001-0.8j),
        Pu=mp.mpc(0.001+0.9j)
    )
    parIdem = ParametersIdempotency(
        Ps=mp.mpc(0.001+0.1j), Psprime=mp.mpc(0.001+3j), 
        P1=mp.mpc(0.001+0.2j), P2=mp.mpc(0.001+0.6j), 
        P3=mp.mpc(0.001+0.3j), P4=mp.mpc(0.001+0.7j),
    )
    theo=Theory(m=1,n=1)
else:
    parPenta = ParametersPentagon(
        Po=mp.mpc(0.0+0.1j), Pq=mp.mpc(0.0+0.2j),
        P1=mp.mpc(0.0+0.3j), P2=mp.mpc(0.0+0.4j),
        P3=mp.mpc(0.0+0.5j), P4=mp.mpc(0.0+0.6j),
        P5=mp.mpc(0.0+0.7j), Pt=mp.mpc(0.0+0.8j),
        Pu=mp.mpc(0.001+0.9j)
    )
    theo=Theory(m=1,n=1)

# =========================================
# Useful modules to plot a bunch of different things
# =========================================

def plot_phase2D(func, x_min=-2, x_max=2, y_min=-2, y_max=2, resolution=100):

   x = np.linspace(x_min, x_max, resolution)
   y = np.linspace(y_min, y_max, resolution)
   X, Y = np.meshgrid(x, y)
   Z = np.zeros_like(X, dtype=np.float64)

   for i in range(resolution):
       for j in range(resolution):
           z_val = func(mp.mpc(X[i, j], Y[i, j]))
           Z[i, j] = mp.arg(z_val)

   plt.figure(figsize=(8, 6))
   plt.contourf(X, Y, Z, levels=100, cmap='hsv')
   plt.colorbar(label='Phase (radians)')
   plt.title('Phase of the function in the complex plane')
   plt.xlabel('Real part')
   plt.ylabel('Imaginary part')
   plt.xlim(x_min, x_max)
   plt.ylim(y_min, y_max)
   plt.show()

def plot_module1D(func, name: str, x_min=-5, x_max=5, resolution=200):
    x = np.linspace(x_min, x_max, resolution)
    y = np.zeros(resolution, dtype=float)

    for i in range(resolution):
        z_val = func(mp.mpc(x[i]))
        y[i] = float(abs(z_val))

    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.title('Modulus of the function')
    plt.xlabel('p')
    plt.ylabel(name)
    plt.xlim(x_min, x_max)
    plt.grid(True)
    plt.show()

def plot_real1D(func, name: str, x_min=-3, x_max=3, resolution=200):
    x = np.linspace(x_min, x_max, resolution)
    y = np.zeros(resolution, dtype=float)

    for i in range(resolution):
        z_val = func(mp.mpc(x[i]))
        y[i] = float(mp.re(z_val))

    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.title('Real part of the function')
    plt.xlabel('p')
    plt.ylabel(name)
    plt.xlim(x_min, x_max)
    plt.grid(True)
    plt.show()

# =========================================
# Conformal dimension relation
# =========================================
def conformaldimension(Theory: Theory, P:complex):
    Q = Theory.b + 1/ Theory.b
    return Q**2/4 - P**2

# =========================================
# Special functions: Barnes G products and double gamma function for rational b^2
# =========================================

def Gmn(Theory: Theory, z:complex):
    """
    G_{m,n}(z) = Π_{k=0..m-1} Π_{l=0..n-1} BarnesG(z + k/m + l/n)
    """
    m, n = Theory.m, Theory.n
    z = mp.mpc(z)

    prod = mp.mpc(1)
    m_mp = mp.mpf(m)
    n_mp = mp.mpf(n)

    for k in range(m):
        kshift = mp.mpf(k) / m_mp
        for l in range(n):
            prod *= barnesg(z + kshift + mp.mpf(l) / n_mp)
    return prod


def Gtilde(z:complex):
    """
    \tilde G(z) = G(1+z)/G(1-z)
    """
    z = mp.mpc(z)
    return barnesg(1 + z) / barnesg(1 - z)


def Gtilde_mn(Theory: Theory, z:complex):
    """
    \tilde G_{m,n}(z) = Π_{k=0..m-1} Π_{l=0..n-1} \tilde G(z - x_{m,n}^{(k,l)})
    with your current shift convention implemented as:
      shift = k/m + 1/(2m) + l/n + 1/(2n)
      argument = z + shift - 1
    (Keep this exactly as your file defines it.)
    """
    m, n = Theory.m, Theory.n
    z = mp.mpc(z)

    prod = mp.mpc(1)
    m_mp = mp.mpf(m)
    n_mp = mp.mpf(n)

    half = mp.mpf(1) / 2
    for k in range(m):
        kshift = mp.mpf(k) / m_mp + half / m_mp
        for l in range(n):
            shift = kshift + mp.mpf(l) / n_mp + half / n_mp
            prod *= Gtilde(z + shift - 1)
    return prod

def Gamma_b(Theory: Theory, z:complex):
    m, n = Theory.m, Theory.n
    b = Theory.b
    Q = b + 1/b
    z = mp.mpc(z)
    gamma_mn = (m*n)**(-Q**2/16)*Gmn(Theory, (1/m + 1/n)/2)
    Numerator = gamma_mn * (m*n)**(z/4*(Q-z))*(2*mp.pi)**(z*mp.sqrt(m*n)/2 - (m+n)/4)
    Denominator = Gmn(Theory, z/mp.sqrt(m*n))

    return Numerator/Denominator

def shiftrelationsGamma_b(Theory: Theory, z:complex, epsilon: int):
    b =Theory.b
    LHS = Gamma_b(Theory,z+b**(epsilon))/Gamma_b(Theory,z)
    RHS = mp.sqrt(2*mp.pi)*b**(epsilon*b**(epsilon)*z -epsilon/2)/(mp.gamma(b**(epsilon)*z))

    return LHS, RHS

# =========================================
# Special functions: Structure constants
# =========================================
def Plancherel_measure(Theory: Theory, P: complex):
    b = Theory.b
    P = mp.mpc(P)
    rho_b = -4*mp.sqrt(2)*mp.sin(2*mp.pi*b*P)*mp.sin(2*mp.pi*P/b)
    return rho_b


def spacelikeB_b(Theory: Theory, P:complex):
    b = Theory.b
    P = mp.mpc(P)
    return 1/Plancherel_measure(Theory, P)

def spacelikeC_b(Theory: Theory, P1:complex, P2:complex, P3:complex):
    b = Theory.b
    Q = b + 1/b
    P1, P2, P3 = mp.mpc(P1), mp.mpc(P2), mp.mpc(P3)

    Numerator = (
        Gamma_b(Theory, 2*Q)
        * Gamma_b(Theory, Q/2 + P1 + P2 + P3)
        * Gamma_b(Theory, Q/2 + P1 - P2 + P3)
        * Gamma_b(Theory, Q/2 + P1 + P2 - P3)
        * Gamma_b(Theory, Q/2 + P1 - P2 - P3)
        * Gamma_b(Theory, Q/2 - P1 + P2 + P3)
        * Gamma_b(Theory, Q/2 - P1 + P2 - P3)
        * Gamma_b(Theory, Q/2 - P1 - P2 + P3)
        * Gamma_b(Theory, Q/2 - P1 - P2 - P3)
    )

    Denominator = (
        mp.sqrt(2) * Gamma_b(Theory, Q)**3
        * Gamma_b(Theory, Q + 2*P1) * Gamma_b(Theory, Q - 2*P1)
        * Gamma_b(Theory, Q + 2*P2) * Gamma_b(Theory, Q - 2*P2)
        * Gamma_b(Theory, Q + 2*P3) * Gamma_b(Theory, Q - 2*P3)
    )

    return Numerator / Denominator

# =========================================
# Special functions: Structure constants
# =========================================

def Plancherel_measure_timelike(Theory: Theory, P: complex):
    P = mp.mpc(P)
    return P**2 / (2*Plancherel_measure(Theory, 1j*P))


def timelikeB_b(Theory: Theory, P:complex):
    P =mp.mpc(P)

    return 2/(P**2*spacelikeB_b(Theory, 1j*P))

def timelikeC_b(Theory:Theory, P1:complex, P2:complex, P3:complex):
    return 1/spacelikeC_b(Theory, 1j*P1, 1j*P2, 1j*P3)

def PHI_b(Theory, P1:complex, P2:complex, P3:complex):
    return (mp.mpc(P2)*Plancherel_measure_timelike(Theory, P1)*timelikeC_b(Theory, P1, P2, P3))/(mp.mpc(P1))


# =========================================
# Trig helpers + a(z)
# =========================================

def smn(Theory: Theory, z:complex):
    return mp.sin(2 * mp.pi * Theory.s * mp.mpc(z))

def cmn(Theory: Theory, z:complex):
    return mp.cos(2 * mp.pi * Theory.s * mp.mpc(z))

def a(Theory: Theory, z:complex):
    # a_i = P_i - (m+n)/(2s) in your notation; here: a(Theory, z) = z - (m+n)/(2s)
    return mp.mpc(z) - (mp.mpf(Theory.m + Theory.n) / (2 * Theory.s))

# =========================================
# ===================
# FUSION KERNEL
# ===================
# =========================================


# =========================================
# Quantum modular fusion polynomial
# =========================================

# Parameters of the polynomial
def alphaf(Theory: Theory, P1:complex, P2:complex, P3:complex, P4:complex, Ps:complex, Pt:complex):
    
    P1, P2, P3, P4, Ps, Pt = mp.mpc(P1), mp.mpc(P2), mp.mpc(P3), mp.mpc(P4), mp.mpc(Ps), mp.mpc(Pt)
    m, n = Theory.m, Theory.n
    s = Theory.s

    m_inv = mp.mpf(1) / mp.mpf(m)
    n_inv = mp.mpf(1) / mp.mpf(n)

    pref_plus  = 2 * mp.pi * 1j * s**2 * (m_inv + n_inv) / 4
    pref_minus = 2 * mp.pi * 1j * s**2 * (-m_inv - n_inv) / 4

    def E(pref, lin):
        return mp.e ** (pref + mp.pi * 1j * s * lin)

    term1 = E(pref_plus,  Ps + Pt - P1 - P2 - P3 - P4)
    term2 = E(pref_plus,  Ps + Pt + P1 + P2 + P3 + P4)
    term3 = E(pref_plus, -Ps - Pt - P1 + P2 - P3 + P4)
    term4 = E(pref_plus, -Ps - Pt + P1 - P2 + P3 - P4)

    term5 = E(pref_minus, -Ps + Pt - P1 - P2 + P3 + P4)
    term6 = E(pref_minus, -Ps + Pt + P1 + P2 - P3 - P4)
    term7 = E(pref_minus,  Ps - Pt - P1 + P2 + P3 - P4)
    term8 = E(pref_minus,  Ps - Pt + P1 - P2 - P3 + P4)

    return term1 + term2 + term3 + term4 - term5 - term6 - term7 - term8

def betaf(Theory: Theory, P1:complex, P2:complex, P3:complex, P4:complex, Ps:complex, Pt:complex):
    return 4*(smn(Theory,P1)*smn(Theory,P3)+smn(Theory,P2)*smn(Theory,P4)+smn(Theory,Ps)*smn(Theory,Pt))

def gammaf(Theory: Theory, P1:complex, P2:complex, P3:complex, P4:complex, Ps:complex, Pt:complex):
    P1, P2, P3, P4, Ps, Pt = mp.mpc(P1), mp.mpc(P2), mp.mpc(P3), mp.mpc(P4), mp.mpc(Ps), mp.mpc(Pt)
    m, n = Theory.m, Theory.n
    s = Theory.s

    m_inv = mp.mpf(1) / mp.mpf(m)
    n_inv = mp.mpf(1) / mp.mpf(n)

    pref_plus  = 2 * mp.pi * 1j * s**2 * (m_inv + n_inv) / 4
    pref_minus = 2 * mp.pi * 1j * s**2 * (-m_inv - n_inv) / 4

    def E(pref, lin):
        return mp.e ** (-pref - mp.pi * 1j * s * lin)

    term1 = E(pref_plus,  Ps + Pt - P1 - P2 - P3 - P4)
    term2 = E(pref_plus,  Ps + Pt + P1 + P2 + P3 + P4)
    term3 = E(pref_plus, -Ps - Pt - P1 + P2 - P3 + P4)
    term4 = E(pref_plus, -Ps - Pt + P1 - P2 + P3 - P4)

    term5 = E(pref_minus, -Ps + Pt - P1 - P2 + P3 + P4)
    term6 = E(pref_minus, -Ps + Pt + P1 + P2 - P3 - P4)
    term7 = E(pref_minus,  Ps - Pt - P1 + P2 + P3 - P4)
    term8 = E(pref_minus,  Ps - Pt + P1 - P2 - P3 + P4)
    
    return term1 + term2 + term3 + term4 - term5 - term6 - term7 - term8


#Determinant of the quantum modular polynomial
def Gram_matrixf(Theory: Theory, P1:complex, P2:complex, P3:complex, P4:complex, Ps:complex, Pt:complex):
    P1, P2, P3, P4, Ps, Pt = mp.mpc(P1), mp.mpc(P2), mp.mpc(P3), mp.mpc(P4), mp.mpc(Ps), mp.mpc(Pt)
    m, n = Theory.m, Theory.n
    s = Theory.s
    G = mp.zeros(4, 4)
    for i in range(4):
        G[i, i] = 1
    G[0,1]=-cmn(Theory, a(Theory, P2))
    G[0,2]=-cmn(Theory, a(Theory, P3))
    G[0,3]=cmn(Theory, a(Theory, Ps))
    G[1,0]=-cmn(Theory, a(Theory, P2))
    G[1,2]=cmn(Theory, a(Theory, Pt))
    G[1,3]=-cmn(Theory, a(Theory, P1))
    G[2,0]=-cmn(Theory, a(Theory, P3))
    G[2,1]=cmn(Theory, a(Theory, Pt))
    G[2,3]=-cmn(Theory, a(Theory, P4))
    G[3,0]=cmn(Theory, a(Theory, Ps))
    G[3,1]=-cmn(Theory, a(Theory, P1))
    G[3,2]=-cmn(Theory, a(Theory, P4))
    return G

def Gram_determinantf(Theory: Theory, P1:complex, P2:complex, P3:complex, P4:complex, Ps:complex, Pt:complex):
    G = Gram_matrixf(Theory, P1, P2, P3, P4, Ps, Pt)
    return mp.det(G)

def quantum_determinantf(Theory: Theory, P1:complex, P2:complex, P3:complex, P4:complex, Ps:complex, Pt:complex):
    return -4*Gram_determinantf(Theory, P1, P2, P3, P4, Ps, Pt)/(smn(Theory, Ps)**2*smn(Theory, Pt)**2)

# roots of the polynomial
def rootsf(Theory: Theory, P1:complex, P2:complex, P3:complex, P4:complex, Ps:complex, Pt:complex):
    beta_val = betaf(Theory, P1, P2, P3, P4, Ps, Pt)
    alpha_val = alphaf(Theory, P1, P2, P3, P4, Ps, Pt)

    z_plus = (-beta_val +2j*smn(Theory, Ps)*smn(Theory, Pt)*mp.sqrt(quantum_determinantf(Theory, P1, P2, P3, P4, Ps, Pt))) / (2 * alpha_val)
    z_minus = (-beta_val -2j*smn(Theory, Ps)*smn(Theory, Pt)*mp.sqrt(quantum_determinantf(Theory, P1, P2, P3, P4, Ps, Pt))) / (2 * alpha_val)

    return z_plus, z_minus

# =========================================
# Fusion Kernels
# =========================================

# These are the elementary building blocks of the kernels
def CurlyF(Theory: Theory, P1:complex, P2:complex, P3:complex, P4:complex, Ps:complex, Pt:complex, z:complex):
    P1, P2, P3, P4, Ps, Pt, z = mp.mpc(P1), mp.mpc(P2), mp.mpc(P3), mp.mpc(P4), mp.mpc(Ps), mp.mpc(Pt), mp.mpc(z)
    m, n = Theory.m, Theory.n
    s = Theory.s

    term1num = Gtilde_mn(Theory, z -1+(1/m + 1/n)/4 - (-Ps+Pt-P1-P2+P3+P4)/(2*s))
    term2num = Gtilde_mn(Theory, z -1+(1/m + 1/n)/4 - (-Ps+Pt+P1+P2-P3-P4)/(2*s))
    term3num = Gtilde_mn(Theory, z -1+(1/m + 1/n)/4 - (Ps-Pt-P1+P2+P3-P4)/(2*s))
    term4num = Gtilde_mn(Theory, z -1+(1/m + 1/n)/4 - (Ps-Pt+P1-P2-P3+P4)/(2*s))
    Numerator = term1num*term2num*term3num*term4num

    term1den = Gtilde_mn(Theory, z -(1/m + 1/n)/4 - (Ps+Pt-P1-P2-P3-P4)/(2*s))
    term2den = Gtilde_mn(Theory, z -(1/m + 1/n)/4 - (Ps+Pt+P1+P2+P3+P4)/(2*s))
    term3den = Gtilde_mn(Theory, z -(1/m + 1/n)/4 - (-Ps-Pt-P1+P2-P3+P4)/(2*s))
    term4den = Gtilde_mn(Theory, z -(1/m + 1/n)/4 - (-Ps-Pt+P1-P2+P3-P4)/(2*s))
    Denominator = term1den*term2den*term3den*term4den

    return Numerator/Denominator
    
def F(Theory: Theory, P1:complex, P2:complex, P3:complex, P4:complex, Ps:complex, Pt:complex):
    P1, P2, P3, P4, Ps, Pt = mp.mpc(P1), mp.mpc(P2), mp.mpc(P3), mp.mpc(P4), mp.mpc(Ps), mp.mpc(Pt)
    m, n = Theory.m, Theory.n
    s = Theory.s

    term0num=1j*(2*mp.pi)**(4*s**2-(m+n))*Gmn(Theory,2*Pt/s)*Gmn(Theory,-2*Pt/s)
    term1num=Gmn(Theory,(1/m + 1/n)/2 +(P1-P2+Ps)/s)
    term2num=Gmn(Theory,(1/m + 1/n)/2 +(-P1+P2+Ps)/s)
    term3num=Gmn(Theory,(1/m + 1/n)/2 +(P1+P2-Ps)/s)
    term4num=Gmn(Theory,(1/m + 1/n)/2 +(-P1-P2-Ps)/s)
    term5num=Gmn(Theory,(1/m + 1/n)/2 +(P3-P4+Ps)/s)
    term6num=Gmn(Theory,(1/m + 1/n)/2 +(-P3+P4+Ps)/s)
    term7num=Gmn(Theory,(1/m + 1/n)/2 +(P3+P4-Ps)/s)
    term8num=Gmn(Theory,(1/m + 1/n)/2 +(-P3-P4-Ps)/s)
    Numerator = term0num*term1num*term2num*term3num*term4num*term5num*term6num*term7num*term8num

    term0den=s*Gmn(Theory,1/m +1/n + 2*Ps/s)*Gmn(Theory,1/m +1/n - 2*Ps/s)
    term1den=Gmn(Theory,(1/m + 1/n)/2 +(P2+P3+Pt)/s)
    term2den=Gmn(Theory,(1/m + 1/n)/2 +(P2-P3-Pt)/s)
    term3den=Gmn(Theory,(1/m + 1/n)/2 +(-P2-P3+Pt)/s)
    term4den=Gmn(Theory,(1/m + 1/n)/2 +(-P2+P3-Pt)/s)
    term5den=Gmn(Theory,(1/m + 1/n)/2 +(P1+P4+Pt)/s)
    term6den=Gmn(Theory,(1/m + 1/n)/2 +(P1-P4-Pt)/s)
    term7den=Gmn(Theory,(1/m + 1/n)/2 +(-P1-P4+Pt)/s)
    term8den=Gmn(Theory,(1/m + 1/n)/2 +(-P1+P4-Pt)/s)
    Denominator = term0den*term1den*term2den*term3den*term4den*term5den*term6den*term7den*term8den

    return Numerator/Denominator

#These are the plus/minus kernels/TV kernels
def Kernelsf(Theory: Theory, P1: complex, P2: complex, P3: complex, P4: complex, Ps: complex, Pt: complex, epsilon: int):

    # epsilon =\pm are the elementary kernels while epsilon=0 is the Teschner-Vartanov kernel
    s = Theory.s
    N = Theory.m * Theory.n  # this is s^2 as an integer

    zplus, zminus = rootsf(Theory, P1, P2, P3, P4, Ps, Pt)

    prefactor = (
        F(Theory, P1, P2, P3, P4, Ps, Pt)
        / (2j * smn(Theory, Pt) * smn(Theory, Ps) * mp.sqrt(quantum_determinantf(Theory, P1, P2, P3, P4, Ps, Pt)))
    )

    bigsum = mp.mpc(0)

    if epsilon == 1:
        u0 = mp.log(zplus) / (2j * mp.pi * s**2)
        for k in range(N):
            bigsum += CurlyF(Theory, P1, P2, P3, P4, Ps, Pt, u0 + mp.mpf(k) / N)
        return prefactor * bigsum

    elif epsilon == -1:
        u0 = mp.log(zminus) / (2j * mp.pi * s**2)
        for k in range(N):
            bigsum += CurlyF(Theory, P1, P2, P3, P4, Ps, Pt, u0 + mp.mpf(k) / N)
        return -prefactor * bigsum
    elif epsilon ==0:
        return (Kernelsf(Theory, P1, P2, P3, P4,
            Ps, Pt,1)+Kernelsf(Theory, P1, P2, P3, P4,
            Ps, Pt,-1))/2
    else:
        raise ValueError("epsilon must be either 0,1 or -1")


# =========================================
# Identities to be checked for the fusion kernel
# =========================================

def mixingsymmetrieskernelsf(Theory: Theory, P1: complex, P2: complex, P3: complex, P4: complex,
            Ps: complex, Pt: complex, epsilon: int):
    b = Theory.b
    vacuummomentum=(b + 1/b)/2 +0.00001j #for this momentum = P<1,1>, there is a "fake pole" numerically! For this reason, I shifted a bit by giving them a small imaginary part
    return Kernelsf(Theory,P2,P2,P3,P3,vacuummomentum,Pt,0)*Kernelsf(Theory,Pt,P2,Ps,P4,P3,P1,epsilon)/(Kernelsf(Theory,P2,P2,Ps,Ps,vacuummomentum,P1,0))

# This is the identity that Ionnis wrote down in his paper with Sylvain
def RibaultTsiaresspacelikef(Theory: Theory,Ps:complex, Pt:complex, epsilon:int):
    Ps, Pt =mp.mpc(Ps), mp.mpc(Pt)
    m, n = Theory.m, Theory.n
    if (m!=1 or n!=1):
        raise ValueError("The central charge is not 25")
    else:
        if (epsilon==1):
            return 1j*Pt*16**(Pt**2-Ps**2)*mp.exp(2j*mp.pi*Ps*Pt)/Ps
        elif (epsilon==-1):
            return -1j*Pt*16**(Pt**2-Ps**2)*mp.exp(-2j*mp.pi*Ps*Pt)/Ps
        else:
            raise ValueError("epsilon must be either 1 or -1")

# This returns the LHS and the RHS of the Pentagon identity
def Pentagon(Theory: Theory, Param: ParametersPentagon, epsilon: int):
    Po, Pq, P1, P2, P3, P4, P5, Pt, Pu = (
        Param.Po, Param.Pq, Param.P1, Param.P2,
        Param.P3, Param.P4, Param.P5, Param.Pt, Param.Pu
    )

    IntegrandLHS = lambda ps: (
        Kernelsf(Theory, Pu, 1j*(ps+0.0001j), P3, P5, Pq, P4, epsilon)
        * Kernelsf(Theory, Pu, P1, P2, Pq, Po, 1j*(ps+0.0001j), epsilon)
        * Kernelsf(Theory, P1, P2, P3, P4, 1j*(ps+0.0001j), Pt, 0)
    )

    LHS = mp.quad(IntegrandLHS, [-Theory.Lambda, Theory.Lambda])

    RHS = (
        Kernelsf(Theory, Pu, P1, Pt, P5, Po, P4, epsilon)
        * Kernelsf(Theory, Po, P2, P3, P5, Pq, Pt, epsilon)
    )

    return LHS, RHS

def Idempotencyf(Theory: Theory, Param:ParametersIdempotency, epsilon:int):
    Ps, Psprime, P1, P2, P3, P4 = (
        Param.Ps, Param.Psprime, Param.P1, Param.P2,
        Param.P3, Param.P4
    )
    Integrand = lambda p: (
        Kernelsf(Theory, P1, P2, P3, P4, Ps, 1j*(p+0.0001j), epsilon)
        * Kernelsf(Theory, P2, P3, P4, P1, 1j*(p+0.0001j), Psprime, epsilon)
    )

    LHS = mp.quad(Integrand, [-Theory.Lambda, Theory.Lambda])

    return LHS

# =========================================
# ===================
# Modular kernel
# ===================
# =========================================

# =========================================
# Quantum modular modular polynomial
# =========================================

# quantum modular determinant D
def quantum_determinantm(Theory: Theory, P0:complex, Ps:complex, Pt:complex):
    m, n= Theory.m, Theory.n
    s= Theory.s
    return 1-(smn(Theory, P0/2 - (m+n)/(4*s))/(smn(Theory, Pt)*smn(Theory, Ps)))**2

# roots of the polynomial
def rootsm(Theory: Theory, P0:complex, Ps:complex, Pt:complex):
    m, n= Theory.m, Theory.n
    s= Theory.s
    N = m*n

    prefactor = smn(Theory,Ps)/smn(Theory,P0/2 -Ps -(m+n)/(4*s))

    plusrootfactor = (-1)**N*cmn(Theory,Pt)-1j*smn(Theory,Pt)*mp.sqrt(quantum_determinantm(Theory,P0,Ps,Pt))
    minusrootfactor = (-1)**N*cmn(Theory,Pt)+1j*smn(Theory,Pt)*mp.sqrt(quantum_determinantm(Theory,P0,Ps,Pt))
    
    z_plus= prefactor*plusrootfactor
    z_minus= prefactor*minusrootfactor

    return z_plus, z_minus



# =========================================
# Modular Kernels
# =========================================

# These are the elementary building blocks of the modular kernels
def CurlyM(Theory: Theory, P0:complex, Ps:complex, Pt:complex, z:complex):
    P0, Ps, Pt, z = mp.mpc(P0), mp.mpc(Ps), mp.mpc(Pt), mp.mpc(z)
    m, n = Theory.m, Theory.n
    s = Theory.s

    term1num = mp.exp(-4j*mp.pi*Ps*z*s)*Gtilde_mn(Theory, z -1+(1/m + 1/n)/4 -(2*Pt+P0)/(2*s))
    term2num = Gtilde_mn(Theory, z -1+(1/m + 1/n)/4 -(-2*Pt+P0)/(2*s))
    Numerator = term1num*term2num

    term1den = Gtilde_mn(Theory, z -(1/m + 1/n)/4 - (2*Pt-P0)/(2*s))
    term2den = Gtilde_mn(Theory, z -(1/m + 1/n)/4 - (-2*Pt-P0)/(2*s))
    Denominator = term1den*term2den

    return Numerator/Denominator
    
def M(Theory: Theory, P0:complex, Ps:complex, Pt:complex):
    P0, Ps, Pt = mp.mpc(P0), mp.mpc(Ps), mp.mpc(Pt)
    m, n = Theory.m, Theory.n
    s = Theory.s
    b = Theory.b
    N = m*n

    Nmn = (-1)**N * (2*mp.pi)**(2*N) * mp.exp(2j*mp.pi*s*Ps)/(s**2)
    Plancherel = -4*mp.sqrt(2)*mp.sin(2*mp.pi*b*Pt)*mp.sin(2*mp.pi*Pt/b)

    term0num=s*(2*mp.pi)**(s*P0-m-n) *Nmn*Plancherel
    term1num=Gmn(Theory,1/m + 1/n +2*Pt/s)
    term2num=Gmn(Theory,(1/m + 1/n)/2 -(P0+2*Ps)/s)
    term3num=Gmn(Theory,1/m + 1/n -2*Pt/s)
    term4num=Gmn(Theory,(1/m + 1/n)/2 -(P0-2*Ps)/s)
    Numerator = term0num*term1num*term2num*term3num*term4num

    term0den=2*Gtilde_mn(Theory,-P0/s)
    term1den=Gmn(Theory,1/m + 1/n + 2*Ps/s)
    term2den=Gmn(Theory,(1/m + 1/n)/2 - (P0+2*Pt)/s)
    term3den=Gmn(Theory,1/m + 1/n - 2*Ps/s)
    term4den=Gmn(Theory,(1/m + 1/n)/2 - (P0-2*Pt)/s)
    Denominator = term0den*term1den*term2den*term3den*term4den
    return Numerator/Denominator

#These are the plus/minus kernels/TV kernels
def KernelsM(Theory: Theory, P0: complex, Ps: complex, Pt: complex, epsilon: int):

    # epsilon =\pm are the elementary kernels while epsilon=0 is the Teschner-Vartanov kernel
    s = Theory.s
    N = Theory.m * Theory.n  # this is s^2 as an integer type

    zplus, zminus = rootsm(Theory, P0, Ps, Pt)

    prefactor = (
        M(Theory, P0, Ps, Pt)
        / (2j * smn(Theory, Pt) * smn(Theory, Ps) * mp.sqrt(quantum_determinantm(Theory, P0, Ps, Pt)))
    )

    bigsum = mp.mpc(0)

    if epsilon == 1:
        u0 = mp.log(zplus) / (2j * mp.pi * s**2)
        for k in range(N):
            bigsum += CurlyM(Theory, P0, Ps, Pt, u0 + mp.mpf(k) / N)
        return prefactor * bigsum

    elif epsilon == -1:
        u0 = mp.log(zminus) / (2j * mp.pi * s**2)
        for k in range(N):
            bigsum += CurlyM(Theory, P0, Ps, Pt, u0 + mp.mpf(k) / N)
        return -prefactor * bigsum
    
    elif epsilon ==0:
        return (KernelsM(Theory, P0, Ps, Pt,1)+KernelsM(Theory, P0, Ps, Pt,-1))/2
    else:
        raise ValueError("epsilon must be either 0,1 or -1")

#Consistency checks for the modular kernels

def Identitylimit(Theory: Theory, Ps: complex, Pt: complex):
    ident = -(Theory.b+1/Theory.b + 0.00001j)/2 # This is to avoid a fake pole
    return KernelsM(Theory, ident, Ps, Pt, 0), mp.sqrt(2)*mp.cos(4*mp.pi*Ps*Pt)

def mixingsymmetriesm(Theory: Theory, P0: complex, P1: complex, P2: complex, epsilon: int ):
    LHS = KernelsM(Theory, P0, P1, P2, epsilon)
    RHS = spacelikeC_b(Theory, P2, P2, P0)*spacelikeB_b(Theory, P1)*KernelsM(Theory, P0, P2, P1, epsilon)/(spacelikeC_b(Theory, P1, P1, P0)*spacelikeB_b(Theory, P2))
    return LHS, RHS

# =========================================
# Mixed checks involving both the modular and the fusion kernels
# =========================================
def NonrationalVerlinde(Theory: Theory, P0:complex, P1: complex, P2:complex, P3:complex, epsilon:int):
    b = Theory.b
    ident = -(b+1/b + 0.0000001j)/2

    Integrand = lambda p: (KernelsM(Theory,P0,P1,1j*(p+0.0000001j),epsilon)*KernelsM(Theory,ident,P2,1j*(p+0.0000001j),epsilon)*KernelsM(Theory,P0,1j*(p+0.0000001j),P3,epsilon)/(-2*mp.sqrt(2)*mp.sin(2*mp.pi*b*1j*(p+0.0000001j))*mp.sin(2*mp.pi*1j*(p+0.0000001j)/b)))

    RHS = mp.quad(Integrand, [-Theory.Lambda, Theory.Lambda])
    
    LHS = Kernelsf(Theory,P1,P0,P3,P2,P1,P3,epsilon)

    return LHS, RHS

def Torus2pointrelation(Theory: Theory, P0:complex, P0prime:complex, P1:complex, P2:complex, P3:complex, P5:complex, epsilon:int):
    b = Theory.b
    shiftangle = mp.pi/12
    shiftphase = mp.exp(1j*shiftangle)
    IntegrandLHS = lambda p4: shiftphase*Kernelsf(Theory, P0, P0prime, P2, P2, P3, 1j*(p4*shiftphase+0.000001j),epsilon)*Kernelsf(Theory, P2, P0prime, P0, P2, 1j*(p4*shiftphase+0.000001j), P5, epsilon)*mp.exp(1j*(2*conformaldimension(Theory, 1j*(p4*shiftphase+0.000001j))-2*conformaldimension(Theory,P2)+conformaldimension(Theory,P3)/2)*mp.pi)

    IntegrandRHS = lambda p6: Kernelsf(Theory, P0prime, P0, P1, P1, P3, 1j*(p6+0.000001j),epsilon)*Kernelsf(Theory, 1j*(p6+0.000001j), P0prime, P0, 1j*(p6+0.000001j), P1, P5, epsilon)*mp.exp(1j*(conformaldimension(Theory, P0)+conformaldimension(Theory,P0prime)-conformaldimension(Theory,P5)/2)*mp.pi)*KernelsM(Theory, P5, 1j*(p6+0.000001j), P2, epsilon)

    LHS = KernelsM(Theory, P3, P1, P2, epsilon)*mp.quad(IntegrandLHS, [-Theory.Lambda, Theory.Lambda])
    RHS = mp.quad(IntegrandRHS, [-Theory.Lambda, Theory.Lambda])
    return LHS, RHS

# =========================================
# Virasoro Wick rotation of the timelike kernels 
# =========================================

def TimelikeKernelsM(Theory: Theory, P0:complex, Ps:complex, Pt:complex, epsilon:int):
    if (epsilon==1 or epsilon==-1):
        return -epsilon*1j*(Pt/Ps)*KernelsM(Theory, 1j*P0, 1j*Pt, 1j*Ps, epsilon)
    elif(epsilon==0):
        # This is the non meromorphic kernel solving the shift equation, analog of the Teschner-Vartanov kernel.
        return (TimelikeKernelsM(Theory, P0, Ps, Pt,1)+TimelikeKernelsM(Theory, P0, Ps, Pt,-1))/2
    else:
        raise ValueError("epsilon must be either 0,1 or -1")

    
def TimelikeKernelsf(Theory: Theory, P1:complex, P2:complex, P3:complex, P4:complex, Ps:complex, Pt:complex, epsilon:int):
    if (epsilon==-1 or epsilon==1):
        return -epsilon*1j*(Pt/Ps)*Kernelsf(Theory, 1j*P3, 1j*P2, 1j*P1, 1j*P4, 1j*Pt, 1j*Ps, epsilon)
    elif (epsilon==0):
        # This is the non meromorphic kernel solving the shift equation, analog of the Teschner-Vartanov kernel.
        return (TimelikeKernelsf(Theory, P1, P2, P3, P4, Ps, Pt,1)+TimelikeKernelsf(Theory, P1, P2, P3, P4, Ps, Pt,-1))/2
    else:
        raise ValueError("epsilon must be either 0,1 or -1")

# =========================================
# Consistency checks for the fusion kernel
# =========================================

def RibaultTsiarestimelikef(Theory: Theory,Ps:complex, Pt:complex, epsilon:int):
    Ps, Pt =mp.mpc(Ps), mp.mpc(Pt)
    m, n = Theory.m, Theory.n
    if (m!=1 or n!=1):
        raise ValueError("The central charge is not 1")
    else:
        if (epsilon==1):
            return 16**(Pt**2-Ps**2)*mp.exp(-2j*mp.pi*Ps*Pt)
        elif (epsilon==-1):
            return 16**(Pt**2-Ps**2)*mp.exp(2j*mp.pi*Ps*Pt)
        else:
            raise ValueError("epsilon must be either 1 or -1")

# This returns the LHS and the RHS of the Pentagon identity
def TimelikePentagon(Theory: Theory, Param: ParametersPentagon, epsilon: int):
    Po, Pq, P1, P2, P3, P4, P5, Pt, Pu = (
        Param.Po, Param.Pq, Param.P1, Param.P2,
        Param.P3, Param.P4, Param.P5, Param.Pt, Param.Pu
    )

    IntegrandLHS = lambda ps: (
        TimelikeKernelsf(Theory, Pu, 1j*(ps+1.5j), P3, P5, Pq, P4, epsilon)
        * TimelikeKernelsf(Theory, Pu, P1, P2, Pq, Po, 1j*(ps+1.5j), epsilon)
        * TimelikeKernelsf(Theory, P1, P2, P3, P4, 1j*(ps+1.5j), Pt, 0)
    )

    LHS = mp.quad(IntegrandLHS, [-Theory.Lambda, Theory.Lambda])

    RHS = (
        TimelikeKernelsf(Theory, Pu, P1, Pt, P5, Po, P4, epsilon)
        * TimelikeKernelsf(Theory, Po, P2, P3, P5, Pq, Pt, epsilon)
    )

    return LHS, RHS

def mixingsymmetriestimelikekernelsf(Theory: Theory, P1: complex, P2: complex, P3: complex, P4: complex,
            Ps: complex, Pt: complex, epsilon: int):
    LHS = TimelikeKernelsf(Theory, P1, P2, P3, P4, Ps, Pt, epsilon)
    RHS = PHI_b(Theory, P3, Pt, P2)*TimelikeKernelsf(Theory, Pt, P2, Ps, P4, P3, P1, epsilon)/PHI_b(Theory, Ps, P1, P2)
    return LHS, RHS

# =========================================
# Demo tests
# =========================================

if DEMO:
    #---momenta for consistency checks
    P1=0.00001+0.01j
    P2=0.00001+0.2j
    P3=0.00001+0.03j
    P4=0.00001+0.04j
    P5=0.00001+0.09j
    Ps=0.00001+0.05j
    Pt=0.00001+0.06j
    P0=0.00001+0.07j
    P0prime = 0.00001+0.08j

    #Consistency checks for the special functions
    print("=== Demo I: Consistency checks for the special functions ===")
    print()
    # --- Demo 1: shift identity you’re currently testing ---
    z = mp.mpc(0.6+0.8j)
    z_plus = z + mp.mpf(1) / mp.mpf(theo.n)

    ratio = Gmn(theo, z_plus) / Gmn(theo, z)

    m_mp = mp.mpf(theo.m)
    rhs = mp.power(m_mp, mp.mpf(1)/2 - m_mp * z) \
          * mp.power(2 * mp.pi, (m_mp - mp.mpf(1)) / 2) \
          * mp.gamma(m_mp * z)

    print("=== Demo: Gmn shift check ===")
    print(f"ratio = {ratio}")
    print(f"rhs   = {rhs}")
    print()

    # --- Demo 2: conjugation & inversion checks for Gtilde_mn ---
    Gv = Gtilde_mn(theo, z)
    print("=== Demo: Gtilde_mn checks ===")
    print(f"conj(Gtilde_mn(z))     = {mp.conj(Gv)}")
    print(f"Gtilde_mn(conj(z))     = {Gtilde_mn(theo, mp.conj(z))}")
    print(f"Gtilde_mn(-z)          = {Gtilde_mn(theo, -z)}")
    print(f"1/Gtilde_mn(z)         = {1/Gv}\n")
    print()

    # --- Demo 2bis: Identity limit for Gamma_b ---
    Gamma_bidentity = Gamma_b(theo, (theo.b + 1/theo.b)/2)
    print("=== Demo: Gamma_b identity limit ===")
    print(f"Gamma_b((b + 1/b)/2) = {Gamma_bidentity}")
    print()

    # --- Demo 2ter: shift equations ---
    LHSb, RHSb = shiftrelationsGamma_b(theo, z, 1)
    LHSbminus1, RHSbminus1 = shiftrelationsGamma_b(theo, z, -1)
    print("=== Demo: Gamma_b shift equations ===")
    print(f"Gamma_b(z + b)   = {LHSb}  ||  b^((1/2) - z) * Gamma_b(z) / sqrt(2*pi) = {RHSb}")
    print(f"Gamma_b(z - b)  = {LHSbminus1}  ||  b^((z - (1/2))) * Gamma_b(z) * sqrt(2*pi) = {RHSbminus1}")
    print()

    # --- Demo 2quad: identitylimit of the 3 point structure constants ---
    print("=== Demo: Identity limit of the 3-point structure constants ===")
    b =theo.b
    Q = b + 1/b
    Cidentity = spacelikeC_b(theo, P1, P2, Q/2+0.000001j)
    print(f"C(P1, P2, Q/2) = {Cidentity}")
    print()

    
    #Consistency checks for the quantum modular fusion polynomial
    print("=== Demo II: Consistency checks for the quantum modular fusion polynomial ===")
    print()
    # --- Demo 3: alpha at zero point ---
    alpha_demo = alphaf(theo, 0, 0, 0, 0, 0, 0)
    print("=== Demo: alpha(0,0) ===")
    print(f"alpha = {alpha_demo}")
    print()

    #--- Demo 4: Check that the roots obtained from the function roots(Theory: Theory,Ps,Pt) are the same that the ones obtain from (-beta \pm sqrt(beta^2-4*alpha*gamma)/2*alpha) ---
    z_plus, z_minus = rootsf(theo, P1, P2, P3, P4, Ps, Pt)
    alpha_val = alphaf(theo, P1, P2, P3, P4, Ps, Pt)
    beta_val = betaf(theo, P1, P2, P3, P4, Ps, Pt)
    gamma_val = gammaf(theo, P1, P2, P3, P4, Ps, Pt)
    z_plus_check = (-beta_val + mp.sqrt(beta_val**2 - 4*alpha_val*gamma_val)) / (2 * alpha_val)
    z_minus_check = (-beta_val - mp.sqrt(beta_val**2 - 4*alpha_val*gamma_val)) / (2 * alpha_val)
    print("=== Demo: roots consistency check ===")
    print(f"z_plus from roots() = {z_plus}")
    print(f"z_plus from formula   = {z_plus_check}")
    print(f"z_minus from roots() = {z_minus}")
    print(f"z_minus from formula   = {z_minus_check}\n")
    print()

    #Consistency checks spacelike fusion kernels
    print("=== Demo III: Consistency checks for the spacelike fusion kernels ===")
    print()
    #--- Demo 5: Check the symmetry of CurlyF under the transformation P_s\rightarrow P_3,P_3\rightarrow P_s,P_t\rightarrow P_1,P_1\rightarrow P_t ---
    CurlyF_original = CurlyF(theo, P1, P2, P3, P4, Ps, Pt, z)
    CurlyF_transformed = CurlyF(theo, Pt, P2, Ps, P4, P3, P1, z)
    print("=== Demo: CurlyF symmetry check ===")
    print(f"CurlyF original     = {CurlyF_original  }")
    print(f"CurlyF transformed  = {CurlyF_transformed}")
    print()
    # --- Demo 6: Check the single-valuedness of the sum in the Kernels function ---
    z_plus, z_minus = rootsf(theo, P1, P2, P3, P4, Ps, Pt)

    N = theo.m * theo.n
    s = theo.s
    ell = 3

    res1 = mp.mpc(0)
    res2 = mp.mpc(0)

    u0 = mp.log(z_plus) / (2j * mp.pi * s**2)

    for k in range(N):
        z1 = u0 + mp.mpf(k) / N
        z2 = u0 + mp.mpf(k + ell) / N
        res1 += CurlyF(theo, P1, P2, P3, P4, Ps, Pt, z1)
        res2 += CurlyF(theo, P1, P2, P3, P4, Ps, Pt, z2)

    print("=== Demo: CurlyF single-valuedness check ===")
    print(f"sum unshifted = {res1}")
    print(f"sum shifted   = {res2}")
    print()
    #--- Demo7: Check the reflection property of the spacelike fusion kernels.
    Fplus=Kernelsf(theo,P1,P2,P3,P4,Ps,Pt,1)
    Fminus=Kernelsf(theo,P1,P2,P3,P4,Ps,Pt,-1)
    Fplusreflected=Kernelsf(theo,P1,P2,P3,P4,-Ps,Pt,1)
    Fminusreflected=Kernelsf(theo,P1,P2,P3,P4,-Ps,Pt,-1)
    print("=== Demo: Reflection property of the plus/minus kernels ===")
    print(f"Fplus = {Fplus}")
    print(f"Fminus   = {Fminus}")
    print(f"Fplusreflected = {Fplusreflected}")
    print(f"Fminusreflected   = {Fminusreflected}")
    print()

    #---Demo8: Check the reflection property of the Teschner-Vartanov kernel
    print("=== Demo: Reflection property of Teschner-Vartanov kernel ===")
    print(f"FTeschnerVartanov = {Kernelsf(theo,P1,P2,P3,P4,Ps,Pt,0)}")
    print(f"FTeschnerVartanovreflected = {Kernelsf(theo,P1,P2,P3,P4,-Ps,Pt,0)}")
    print()

    #---Demo9: Check the mixing symmetries of the different kernels
    Fplusmixed=mixingsymmetrieskernelsf(theo,P1,P2,P3,P4,Ps,Pt,1)
    Fminusmixed=mixingsymmetrieskernelsf(theo,P1,P2,P3,P4,Ps,Pt,-1)
    print("=== Demo: mixing property of the kernels ===")
    print(f"Fplus = {Fplus}")
    print(f"Fplusmixed = {Fplusmixed}")
    print(f"Fminus = {Fminus}")
    print(f"Fminusmixed = {Fminusmixed}")
    print()

    # #---Demo10: Check Ioannis/Sylvain relation for the spacelike kernels
    # Fplusatonefourth=Kernelsf(theo,0.25,0.25,0.25,0.25,Ps,Pt,1)
    # Fminusatonefourth=Kernelsf(theo,0.25,0.25,0.25,0.25,Ps,Pt,-1)
    # Fplusfromrelation=RibaultTsiaresspacelikef(theo,Ps,Pt,1)
    # Fminusfromrelation=RibaultTsiaresspacelikef(theo,Ps,Pt,-1)
    # print("=== Demo: Ioannis/Sylvain relation for the spacelike kernels at c=25 ===")
    # print(f"Fplusatonefourth = {Fplusatonefourth}")
    # print(f"Fplusfromrelation = {Fplusfromrelation}")
    # print(f"Fminusatonefourth = {Fminusatonefourth}")
    # print(f"Fminusfromrelation = {Fminusfromrelation}")
    # print()

    # # ---Demo11: Check pentagon for the TV kernel
    # print("=== Demo: Testing the pentagon for the TeschnerVartanov kernel ===")
    # LHS, RHS = Pentagon(theo, parPenta, 0)
    # print(f"LHSPentagon = {LHS}")
    # print(f"RHSPentagon = {RHS}")
    # print()

    # # ---Demo12: Check Idempotency for the TV kernel
    # print("=== Demo: Testing the Idempotency for the TeschnerVartanov kernel ===")
    # LHS= Idempotencyf(theo, parIdem, 1)
    # print(f"Idempotency = {LHS}")
    # print()

    #Consistency checks spacelike modular kernels
    print("=== Demo IV: Consistency checks for the spacelike modular kernels ===")
    print()
    # --- Demo 13: Check the single-valuedness of the sum in the modular Kernels function ---
    z_plus, z_minus = rootsm(theo, P0, Ps, Pt)

    N = theo.m * theo.n
    s = theo.s
    ell = 3

    res1 = mp.mpc(0)
    res2 = mp.mpc(0)

    u0 = mp.log(z_plus) / (2j * mp.pi * s**2)

    for k in range(N):
        z1 = u0 + mp.mpf(k) / N
        z2 = u0 + mp.mpf(k + ell) / N
        res1 += CurlyM(theo, P0, Ps, Pt, z1)
        res2 += CurlyM(theo, P0, Ps, Pt, z2)

    print("=== Demo: CurlyM single-valuedness check ===")
    print(f"sum unshifted = {res1}")
    print(f"sum shifted   = {res2}")
    print()

    # #--- Demo 14: Check the Identity limit of the Modular kernel
    LHS, RHS = Identitylimit(theo, Ps, Pt)
    print("=== Demo: Identity limit check ===")
    print(f"Kernel at identity = {LHS}")
    print(f"Value of the cos   = {RHS}")
    print()

    # #--- Demo 15: Check reflection properties under reflection of the external momenta
    print("=== Demo: Reflection property of Modular kernels ===")
    print(f"Mplus = {KernelsM(theo,P0,Ps,Pt,1)}")
    print(f"Mminus = {KernelsM(theo,P0,Ps,Pt,-1)}")
    print(f"Mplusreflectedps = {KernelsM(theo,P0,-Ps,Pt,1)}")
    print(f"Mminusreflectedps = {KernelsM(theo,P0,-Ps,Pt,-1)}")
    print(f"Mplusreflectedpt = {KernelsM(theo,P0,Ps,-Pt,1)}")
    print(f"Mminusreflectedpt = {KernelsM(theo,P0,Ps,-Pt,-1)}")
    print(f"MTV = {KernelsM(theo,P0,Ps,Pt,0)}")
    print(f"MTVreflectedps = {KernelsM(theo,P0,-Ps,Pt,0)}")
    print(f"MTVreflectedpt = {KernelsM(theo,P0,Ps,-Pt,0)}")
    print()

    # # #--- Demo 15: Plot the TV kernels appearing in the idempotency relation on [-3,3]
    # Mfirst = lambda p: KernelsM(theo,P0,P1, 1j*(p+0.000001j),0)
    # Msecond = lambda p: KernelsM(theo,P0,1j*(p+0.000001j),P2,0)
    # integrand = lambda p: KernelsM(theo,P0,P1, 1j*(p+0.000001j),0)*KernelsM(theo,P0,1j*(p+0.000001j),P2,0)
    # plot_module1D(Mfirst, name="Mfirst")
    # plot_module1D(Msecond, name="Msecond")
    # plot_module1D(integrand, name="integrand")
    

    # # #--- Demo 16: Mixing symmetry of the modular kernels
    print("=== Demo: Mixing symmetry of the Modular kernels ===")
    LHSmixingsymmetries, RHSmixingsymmetries = mixingsymmetriesm(theo, P0, P1, P2, -1)
    print(f"LHS = {LHSmixingsymmetries}")
    print(f"RHS = {RHSmixingsymmetries}")
    print()

    # Checking the important identities involving the spacelike and the timelike kernels
    print("=== Demo V: important identities involving the spacelike modular and fusion kernels ===")
    print()

    # # #--- Demo 17: Non rational Verlinde formula
    # print("=== Demo: Non rational Verlinde formula ===")
    # LHS, RHS = NonrationalVerlinde(theo,P0,P1,P2,P3,0)
    # print(f"LHSVerlindeformula = {LHS}")
    # print(f"RHSVerlindeformula = {RHS}")
    # print()

    # # # #--- Demo 18: Plot the integrand appearing in the non rational Verlinde formula
    # print("=== Demo: plot the integrand in the non rational Verlinde formula ===")
    # b = theo.b
    # ident = -(b+1/b + 0.0000001j)/2
    # Integrand = lambda p: (KernelsM(theo,P0,P1,1j*(p+0.0000001j),0)*KernelsM(theo,ident,P2,1j*(p+0.0000001j),0)*KernelsM(theo,P0,1j*(p+0.0000001j),P3,0)/(-2*mp.sqrt(2)*mp.sin(2*mp.pi*b*1j*(p+0.0000001j))*mp.sin(2*mp.pi*1j*(p+0.0000001j)/b)))
    # plot_module1D(Integrand, name="Integrand in the non rational Verlinde formula")

    # # # # #--- Demo 19: Check the torus 2 point relation
    # print("=== Demo: Check the torus 2 point relation ===")
    # LHS, RHS = Torus2pointrelation(theo, P0, P0prime, P1, P2, P3, P5, 0)
    # print(f"LHSTorus2pt = {LHS}")
    # print(f"RHSTorus2pt = {RHS}")
    # print()

    # #--- Demo 19bis : Plot the integrand in the LHS and RHS of the torus 2 point relation
    # IntegrandLHS = lambda p4: Kernelsf(theo, P0, P0prime, P2, P2, P3, 1j*(p4+0.000001j),0)*Kernelsf(theo, P2, P0prime, P0, P2, 1j*(p4+0.000001j), P5, 0)*mp.exp(1j*(2*conformaldimension(theo, 1j*(p4+0.000001j))-2*conformaldimension(theo,P2)+conformaldimension(theo,P3)/2)*mp.pi)

    # IntegrandRHS = lambda p6: Kernelsf(theo, P0prime, P0, P1, P1, P3, 1j*(p6+0.000001j),0)*Kernelsf(theo, 1j*(p6+0.000001j), P0prime, P0, 1j*(p6+0.000001j), P1, P5, 0)*mp.exp(1j*(conformaldimension(theo, P0)+conformaldimension(theo,P0prime)-conformaldimension(theo,P5)/2)*mp.pi)*KernelsM(theo, P5, 1j*(p6+0.000001j), P2, 0)

    # plot_module1D(IntegrandLHS, name="Integrand LHS torus 2pt relation")
    # plot_module1D(IntegrandRHS, name="Integrand RHS torus 2pt relation")
    # print(KernelsM(theo,P3,P1,P2,0))

    #Checking the important identities involving the spacelike and the timelike kernels
    print("=== TIMELIKE LIOUVILLE !!! ===")
    print()

    
    #--- Demo20: Check the reflection property of the timelike fusion kernels.
    FplusTimelike=TimelikeKernelsf(theo,P1,P2,P3,P4,Ps,Pt,1)
    FminusTimelike=TimelikeKernelsf(theo,P1,P2,P3,P4,Ps,Pt,-1)
    FplusTimelikereflected=TimelikeKernelsf(theo,P1,P2,P3,P4,-Ps,Pt,1)
    FminusTimelikereflected=TimelikeKernelsf(theo,P1,P2,P3,P4,-Ps,Pt,-1)
    print("=== Demo: Reflection property of the plus/minus timelike fusion kernels ===")
    print(f"FplusTimelike = {FplusTimelike}")
    print(f"FminusTimelike   = {FminusTimelike}")
    print(f"FplusTimelikereflected = {FplusTimelikereflected}")
    print(f"FminusTimelikereflected   = {FminusTimelikereflected}")
    print()

    #---Demo21: Check Ioannis/Sylvain relation for the timelike kernels
    Fplusatonefourthj=TimelikeKernelsf(theo,0.25j,0.25j,0.25j,0.25j,Ps,Pt,1)
    Fminusatonefourthj=TimelikeKernelsf(theo,0.25j,0.25j,0.25j,0.25j,Ps,Pt,-1)
    Fplusfromrelation=RibaultTsiarestimelikef(theo,Ps,Pt,1)
    Fminusfromrelation=RibaultTsiarestimelikef(theo,Ps,Pt,-1)
    print("=== Demo: Ioannis/Sylvain relation for the timelike kernels at c=1 ===")
    print(f"Fplusatonefourthj = {Fplusatonefourthj}")
    print(f"Fplusfromrelation = {Fplusfromrelation}")
    print(f"Fminusatonefourthj = {Fminusatonefourthj}")
    print(f"Fminusfromrelation = {Fminusfromrelation}")
    print()

    # # ---Demo22: Check pentagon for the Timelike kernel
    # print("=== Demo: Testing the pentagon for timelike kernel===")
    # LHS, RHS = TimelikePentagon(theo, parPenta, 0)
    # print(f"LHSTimelikePentagonplus = {LHS}")
    # print(f"RHSTimelikePentagonplus = {RHS}")
    # print()

    # # #--- Demo 23: Mixing symmetry of the timelike fusion kernels
    print("=== Demo: Mixing symmetry of the timelike fusion kernels ===")
    LHSmixingsymmetriesFplus, RHSmixingsymmetriesFplus = mixingsymmetriestimelikekernelsf(theo, P1, P2, P3, P4, Ps, Pt, 1)
    print(f"LHS = {LHSmixingsymmetriesFplus}")
    print(f"RHS = {RHSmixingsymmetriesFplus}")
    print()


    
    
    




    



    
    




    


    
    

    

    
    
