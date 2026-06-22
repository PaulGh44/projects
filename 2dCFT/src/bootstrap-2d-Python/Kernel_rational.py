#Packages
from Special_functions_rational import Theory, special_functions_rational
from mpmath import mp


#These are global functions useful for what follows
# =========================================
# Trig helpers + a(z)
# =========================================

def smn(Theory: Theory, z:complex):
    return mp.sin(2 * mp.pi * Theory.s * mp.mpc(z))

def cmn(Theory: Theory, z:complex):
    return mp.cos(2 * mp.pi * Theory.s * mp.mpc(z))

def a(Theory: Theory, z:complex):
    # a_i = P_i - (m+n)/(2s) in our notation; here: a(Theory, z) = z - (m+n)/(2s)
    return mp.mpc(z) - (mp.mpf(Theory.m + Theory.n) / (2 * Theory.s))


# =========================================
# Quantum modular fusion polynomial
# =========================================
class Quantummodularfusion:

    # Parameters of the polynomial
    @staticmethod
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

    @staticmethod
    def betaf(Theory: Theory, P1:complex, P2:complex, P3:complex, P4:complex, Ps:complex, Pt:complex):
        return 4*(smn(Theory,P1)*smn(Theory,P3)+smn(Theory,P2)*smn(Theory,P4)+smn(Theory,Ps)*smn(Theory,Pt))

    @staticmethod
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
    @staticmethod
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
    
    @staticmethod
    def Gram_determinantf(Theory: Theory, P1:complex, P2:complex, P3:complex, P4:complex, Ps:complex, Pt:complex):
        G = Quantummodularfusion.Gram_matrixf(Theory, P1, P2, P3, P4, Ps, Pt)
        return mp.det(G)

    @staticmethod
    def quantum_determinantf(Theory: Theory, P1:complex, P2:complex, P3:complex, P4:complex, Ps:complex, Pt:complex):
        return -4*Quantummodularfusion.Gram_determinantf(Theory, P1, P2, P3, P4, Ps, Pt)/(smn(Theory, Ps)**2*smn(Theory, Pt)**2)

    # roots of the polynomial
    @staticmethod
    def rootsf(Theory: Theory, P1:complex, P2:complex, P3:complex, P4:complex, Ps:complex, Pt:complex):
        beta_val = Quantummodularfusion.betaf(Theory, P1, P2, P3, P4, Ps, Pt)
        alpha_val = Quantummodularfusion.alphaf(Theory, P1, P2, P3, P4, Ps, Pt)

        z_plus = (-beta_val +2j*smn(Theory, Ps)*smn(Theory, Pt)*mp.sqrt(Quantummodularfusion.quantum_determinantf(Theory, P1, P2, P3, P4, Ps, Pt))) / (2 * alpha_val)
        z_minus = (-beta_val -2j*smn(Theory, Ps)*smn(Theory, Pt)*mp.sqrt(Quantummodularfusion.quantum_determinantf(Theory, P1, P2, P3, P4, Ps, Pt))) / (2 * alpha_val)

        return z_plus, z_minus


# =========================================
# Fusion Kernels
# =========================================

class fkernel:

    # These are the elementary building blocks of the kernels
    @staticmethod
    def CurlyF(Theory: Theory, P1:complex, P2:complex, P3:complex, P4:complex, Ps:complex, Pt:complex, z:complex):
        P1, P2, P3, P4, Ps, Pt, z = mp.mpc(P1), mp.mpc(P2), mp.mpc(P3), mp.mpc(P4), mp.mpc(Ps), mp.mpc(Pt), mp.mpc(z)
        m, n = Theory.m, Theory.n
        s = Theory.s

        term1num = special_functions_rational.Gtilde_mn(Theory, z -1+(1/m + 1/n)/4 - (-Ps+Pt-P1-P2+P3+P4)/(2*s))
        term2num = special_functions_rational.Gtilde_mn(Theory, z -1+(1/m + 1/n)/4 - (-Ps+Pt+P1+P2-P3-P4)/(2*s))
        term3num = special_functions_rational.Gtilde_mn(Theory, z -1+(1/m + 1/n)/4 - (Ps-Pt-P1+P2+P3-P4)/(2*s))
        term4num = special_functions_rational.Gtilde_mn(Theory, z -1+(1/m + 1/n)/4 - (Ps-Pt+P1-P2-P3+P4)/(2*s))
        Numerator = term1num*term2num*term3num*term4num

        term1den = special_functions_rational.Gtilde_mn(Theory, z -(1/m + 1/n)/4 - (Ps+Pt-P1-P2-P3-P4)/(2*s))
        term2den = special_functions_rational.Gtilde_mn(Theory, z -(1/m + 1/n)/4 - (Ps+Pt+P1+P2+P3+P4)/(2*s))
        term3den = special_functions_rational.Gtilde_mn(Theory, z -(1/m + 1/n)/4 - (-Ps-Pt-P1+P2-P3+P4)/(2*s))
        term4den = special_functions_rational.Gtilde_mn(Theory, z -(1/m + 1/n)/4 - (-Ps-Pt+P1-P2+P3-P4)/(2*s))
        Denominator = term1den*term2den*term3den*term4den

        return Numerator/Denominator
    
    @staticmethod
    def F(Theory: Theory, P1:complex, P2:complex, P3:complex, P4:complex, Ps:complex, Pt:complex):
        P1, P2, P3, P4, Ps, Pt = mp.mpc(P1), mp.mpc(P2), mp.mpc(P3), mp.mpc(P4), mp.mpc(Ps), mp.mpc(Pt)
        m, n = Theory.m, Theory.n
        s = Theory.s

        term0num=1j*(2*mp.pi)**(4*s**2-(m+n))*special_functions_rational.Gmn(Theory,2*Pt/s)*special_functions_rational.Gmn(Theory,-2*Pt/s)
        term1num=special_functions_rational.Gmn(Theory,(1/m + 1/n)/2 +(P1-P2+Ps)/s)
        term2num=special_functions_rational.Gmn(Theory,(1/m + 1/n)/2 +(-P1+P2+Ps)/s)
        term3num=special_functions_rational.Gmn(Theory,(1/m + 1/n)/2 +(P1+P2-Ps)/s)
        term4num=special_functions_rational.Gmn(Theory,(1/m + 1/n)/2 +(-P1-P2-Ps)/s)
        term5num=special_functions_rational.Gmn(Theory,(1/m + 1/n)/2 +(P3-P4+Ps)/s)
        term6num=special_functions_rational.Gmn(Theory,(1/m + 1/n)/2 +(-P3+P4+Ps)/s)
        term7num=special_functions_rational.Gmn(Theory,(1/m + 1/n)/2 +(P3+P4-Ps)/s)
        term8num=special_functions_rational.Gmn(Theory,(1/m + 1/n)/2 +(-P3-P4-Ps)/s)
        Numerator = term0num*term1num*term2num*term3num*term4num*term5num*term6num*term7num*term8num

        term0den=s*special_functions_rational.Gmn(Theory,1/m +1/n + 2*Ps/s)*special_functions_rational.Gmn(Theory,1/m +1/n - 2*Ps/s)
        term1den=special_functions_rational.Gmn(Theory,(1/m + 1/n)/2 +(P2+P3+Pt)/s)
        term2den=special_functions_rational.Gmn(Theory,(1/m + 1/n)/2 +(P2-P3-Pt)/s)
        term3den=special_functions_rational.Gmn(Theory,(1/m + 1/n)/2 +(-P2-P3+Pt)/s)
        term4den=special_functions_rational.Gmn(Theory,(1/m + 1/n)/2 +(-P2+P3-Pt)/s)
        term5den=special_functions_rational.Gmn(Theory,(1/m + 1/n)/2 +(P1+P4+Pt)/s)
        term6den=special_functions_rational.Gmn(Theory,(1/m + 1/n)/2 +(P1-P4-Pt)/s)
        term7den=special_functions_rational.Gmn(Theory,(1/m + 1/n)/2 +(-P1-P4+Pt)/s)
        term8den=special_functions_rational.Gmn(Theory,(1/m + 1/n)/2 +(-P1+P4-Pt)/s)
        Denominator = term0den*term1den*term2den*term3den*term4den*term5den*term6den*term7den*term8den

        return Numerator/Denominator

    @staticmethod
    #These are the plus/minus kernels/TV kernels/RF kernels
    def Kernelsf(Theory: Theory, P1: complex, P2: complex, P3: complex, P4: complex, Ps: complex, Pt: complex, kernel_code: int):

        # epsilon =\pm are the elementary kernels while epsilon=0 is the Teschner-Vartanov kernel
        s = Theory.s
        N = Theory.m * Theory.n  # this is s^2 as an integer

        zplus, zminus = Quantummodularfusion.rootsf(Theory, P1, P2, P3, P4, Ps, Pt)

        prefactor = (
            fkernel.F(Theory, P1, P2, P3, P4, Ps, Pt)
            / (2j * smn(Theory, Pt) * smn(Theory, Ps) * mp.sqrt(Quantummodularfusion.quantum_determinantf(Theory, P1, P2, P3, P4, Ps, Pt)))
        )

        bigsum = mp.mpc(0)

        if kernel_code == 1:
            u0 = mp.log(zplus) / (2j * mp.pi * s**2)
            for k in range(N):
                bigsum += fkernel.CurlyF(Theory, P1, P2, P3, P4, Ps, Pt, u0 + mp.mpf(k) / N)
            return prefactor * bigsum

        elif kernel_code == -1:
            u0 = mp.log(zminus) / (2j * mp.pi * s**2)
            for k in range(N):
                bigsum += fkernel.CurlyF(Theory, P1, P2, P3, P4, Ps, Pt, u0 + mp.mpf(k) / N)
            return -prefactor * bigsum
        elif kernel_code ==0:
            return (fkernel.Kernelsf(Theory, P1, P2, P3, P4,
                Ps, Pt,1)+fkernel.Kernelsf(Theory, P1, P2, P3, P4,
                Ps, Pt,-1))/2
        elif kernel_code ==2:
            return (fkernel.Kernelsf(Theory, P1, P2, P3, P4,
                Ps, Pt,1)-fkernel.Kernelsf(Theory, P1, P2, P3, P4,
                Ps, Pt,-1))/2
        else:
            raise ValueError("The Kernel code must be either 0, 1, -1 or 2")
        
    @staticmethod
    def TimelikeKernelsf(Theory: Theory, P1:complex, P2:complex, P3:complex, P4:complex, Ps:complex, Pt:complex, kernel_code: int):
        if (kernel_code ==-1 or kernel_code ==1):
            return -kernel_code*1j*(Pt/Ps)*fkernel.Kernelsf(Theory, 1j*P3, 1j*P2, 1j*P1, 1j*P4, 1j*Pt, 1j*Ps, kernel_code)
        elif (kernel_code == 0):
            # This is the non meromorphic kernel solving the shift equation, analog of the Teschner-Vartanov kernel.
            return (fkernel.TimelikeKernelsf(Theory, P1, P2, P3, P4, Ps, Pt,1)+fkernel.TimelikeKernelsf(Theory, P1, P2, P3, P4, Ps, Pt,-1))/2
        elif (kernel_code == 2):
            # This is the meromorphic kernel which does not satisfy crossing
            return (fkernel.TimelikeKernelsf(Theory, P1, P2, P3, P4, Ps, Pt,1)-fkernel.TimelikeKernelsf(Theory, P1, P2, P3, P4, Ps, Pt,-1))/2
        else:
            raise ValueError("The Kernel code must be either 0, 1, -1 or 2")


# =========================================
# Quantum modular modular polynomial
# =========================================

class Quantummodularmodular:

    # quantum modular determinant D
    @staticmethod
    def quantum_determinantm(Theory: Theory, P0:complex, Ps:complex, Pt:complex):
        m, n= Theory.m, Theory.n
        s= Theory.s
        return 1-(smn(Theory, P0/2 - (m+n)/(4*s))/(smn(Theory, Pt)*smn(Theory, Ps)))**2

    # roots of the polynomial
    @staticmethod
    def rootsm(Theory: Theory, P0:complex, Ps:complex, Pt:complex):
        m, n= Theory.m, Theory.n
        s= Theory.s
        N = m*n

        prefactor = smn(Theory,Ps)/smn(Theory,P0/2 -Ps -(m+n)/(4*s))

        plusrootfactor = (-1)**N*cmn(Theory,Pt)-1j*smn(Theory,Pt)*mp.sqrt(Quantummodularmodular.quantum_determinantm(Theory,P0,Ps,Pt))
        minusrootfactor = (-1)**N*cmn(Theory,Pt)+1j*smn(Theory,Pt)*mp.sqrt(Quantummodularmodular.quantum_determinantm(Theory,P0,Ps,Pt))
        
        z_plus= prefactor*plusrootfactor
        z_minus= prefactor*minusrootfactor

        return z_plus, z_minus



# =========================================
# Modular Kernels
# =========================================

class mkernel:
    # These are the elementary building blocks of the modular kernels
    @staticmethod
    def CurlyM(Theory: Theory, P0:complex, Ps:complex, Pt:complex, z:complex):
        P0, Ps, Pt, z = mp.mpc(P0), mp.mpc(Ps), mp.mpc(Pt), mp.mpc(z)
        m, n = Theory.m, Theory.n
        s = Theory.s

        term1num = mp.exp(-4j*mp.pi*Ps*z*s)*special_functions_rational.Gtilde_mn(Theory, z -1+(1/m + 1/n)/4 -(2*Pt+P0)/(2*s))
        term2num = special_functions_rational.Gtilde_mn(Theory, z -1+(1/m + 1/n)/4 -(-2*Pt+P0)/(2*s))
        Numerator = term1num*term2num

        term1den = special_functions_rational.Gtilde_mn(Theory, z -(1/m + 1/n)/4 - (2*Pt-P0)/(2*s))
        term2den = special_functions_rational.Gtilde_mn(Theory, z -(1/m + 1/n)/4 - (-2*Pt-P0)/(2*s))
        Denominator = term1den*term2den

        return Numerator/Denominator
    
    @staticmethod
    def M(Theory: Theory, P0:complex, Ps:complex, Pt:complex):
        P0, Ps, Pt = mp.mpc(P0), mp.mpc(Ps), mp.mpc(Pt)
        m, n = Theory.m, Theory.n
        s = Theory.s
        b = Theory.b
        N = m*n

        Nmn = (-1)**N * (2*mp.pi)**(2*N) * mp.exp(2j*mp.pi*s*Ps)/(s**2)
        Plancherel = -4*mp.sqrt(2)*mp.sin(2*mp.pi*b*Pt)*mp.sin(2*mp.pi*Pt/b)

        term0num=s*(2*mp.pi)**(s*P0-m-n) *Nmn*Plancherel
        term1num=special_functions_rational.Gmn(Theory,1/m + 1/n +2*Pt/s)
        term2num=special_functions_rational.Gmn(Theory,(1/m + 1/n)/2 -(P0+2*Ps)/s)
        term3num=special_functions_rational.Gmn(Theory,1/m + 1/n -2*Pt/s)
        term4num=special_functions_rational.Gmn(Theory,(1/m + 1/n)/2 -(P0-2*Ps)/s)
        Numerator = term0num*term1num*term2num*term3num*term4num

        term0den=2*special_functions_rational.Gtilde_mn(Theory,-P0/s)
        term1den=special_functions_rational.Gmn(Theory,1/m + 1/n + 2*Ps/s)
        term2den=special_functions_rational.Gmn(Theory,(1/m + 1/n)/2 - (P0+2*Pt)/s)
        term3den=special_functions_rational.Gmn(Theory,1/m + 1/n - 2*Ps/s)
        term4den=special_functions_rational.Gmn(Theory,(1/m + 1/n)/2 - (P0-2*Pt)/s)
        Denominator = term0den*term1den*term2den*term3den*term4den
        return Numerator/Denominator

    #These are the plus/minus kernels/TV kernels/RM kernels
    def KernelsM(Theory: Theory, P0: complex, Ps: complex, Pt: complex, kernel_code: int):

        # epsilon =\pm are the elementary kernels while epsilon=0 is the Teschner-Vartanov kernel
        s = Theory.s
        N = Theory.m * Theory.n  # this is s^2 as an integer type

        zplus, zminus = Quantummodularmodular.rootsm(Theory, P0, Ps, Pt)

        prefactor = (
            mkernel.M(Theory, P0, Ps, Pt)
            / (2j * smn(Theory, Pt) * smn(Theory, Ps) * mp.sqrt(Quantummodularmodular.quantum_determinantm(Theory, P0, Ps, Pt)))
        )

        bigsum = mp.mpc(0)

        if kernel_code == 1:
            u0 = mp.log(zplus) / (2j * mp.pi * s**2)
            for k in range(N):
                bigsum += mkernel.CurlyM(Theory, P0, Ps, Pt, u0 + mp.mpf(k) / N)
            return prefactor * bigsum

        elif kernel_code == -1:
            u0 = mp.log(zminus) / (2j * mp.pi * s**2)
            for k in range(N):
                bigsum += mkernel.CurlyM(Theory, P0, Ps, Pt, u0 + mp.mpf(k) / N)
            return -prefactor * bigsum
        
        elif kernel_code ==0:
            return (mkernel.KernelsM(Theory, P0, Ps, Pt,1)+mkernel.KernelsM(Theory, P0, Ps, Pt,-1))/2
        
        elif kernel_code ==2:
            return (mkernel.KernelsM(Theory, P0, Ps, Pt,1)-mkernel.KernelsM(Theory, P0, Ps, Pt,-1))/2
        else:
            raise ValueError("The Kernel code must be either 0,1 ,-1 or 2")
        
    def TimelikeKernelsM(Theory: Theory, P0:complex, Ps:complex, Pt:complex, kernel_code:int):
        if (kernel_code==1 or kernel_code==-1):
            return -kernel_code*1j*(Pt/Ps)*mkernel.KernelsM(Theory, 1j*P0, 1j*Pt, 1j*Ps, kernel_code)
        elif(kernel_code==0):
            # This is the non meromorphic kernel solving the shift equation, analog of the Teschner-Vartanov kernel.
            return (fkernel.TimelikeKernelsM(Theory, P0, Ps, Pt,1)+fkernel.TimelikeKernelsM(Theory, P0, Ps, Pt,-1))/2
        elif(kernel_code==2):
            # This is the meromorphic kernel
            return (fkernel.TimelikeKernelsM(Theory, P0, Ps, Pt,1)-fkernel.TimelikeKernelsM(Theory, P0, Ps, Pt,-1))/2
        else:
            raise ValueError("The Kernel code must be either 0,1 ,-1 or 2")


class kernel_rational:

    def __init__(self, Theory: Theory, Liouville_type: str, kernel_type: str, external_momenta: list[complex], internal_momenta: list[complex],  kernel_code: int,  value = None):

        self.Theory = Theory
        self.Liouville_type = Liouville_type
        self.kernel_type = kernel_type
        self.external_momenta = external_momenta
        self.internal_momenta = internal_momenta
        self.kernel_code = kernel_code

        if Liouville_type not in ("spacelike", "timelike"):
            raise ValueError(
                "Invalid Liouville_type. It should be either 'spacelike' or 'timelike'."
            )

        if kernel_type not in ("fusion", "modular"):
            raise ValueError(
                "Invalid kernel_type. It should be either 'fusion' or 'modular'."
            )

        if value is not None:
            self.value = value
            return

        self.value = self.compute_kernel()

    def compute_kernel(self):
        if self.Liouville_type == "timelike":
            if self.kernel_type == "fusion":
                if len(self.internal_momenta) != 2:
                    raise ValueError(
                        "Invalid number of internal momenta for the fusion kernel."
                    )

                if len(self.external_momenta) != 4:
                    raise ValueError(
                        "Invalid number of external momenta for the fusion kernel."
                    )

                return fkernel.TimelikeKernelsf(
                    self.Theory,
                    self.external_momenta[0],
                    self.external_momenta[1],
                    self.external_momenta[2],
                    self.external_momenta[3],
                    self.internal_momenta[0],
                    self.internal_momenta[1],
                    self.kernel_code
                )
            else:
                if len(self.internal_momenta) != 2:
                    raise ValueError(
                        "Invalid number of internal momenta for the modular kernel."
                    )
                if len(self.external_momenta) != 1:
                    raise ValueError(
                        "Invalid number of external momenta for the modular kernel."
                    )
                return mkernel.TimelikeKernelsM(
                    self.Theory,
                    self.external_momenta[0],
                    self.internal_momenta[0],
                    self.internal_momenta[1],
                )
        else:
            if self.kernel_type == "fusion":
                if len(self.internal_momenta) != 2:
                    raise ValueError(
                        "Invalid number of internal momenta for the fusion kernel."
                    )

                if len(self.external_momenta) != 4:
                    raise ValueError(
                        "Invalid number of external momenta for the fusion kernel."
                    )

                return fkernel.Kernelsf(
                    self.Theory,
                    self.external_momenta[0],
                    self.external_momenta[1],
                    self.external_momenta[2],
                    self.external_momenta[3],
                    self.internal_momenta[0],
                    self.internal_momenta[1],
                    self.kernel_code
                )
            else:
                if len(self.internal_momenta) != 2:
                    raise ValueError(
                        "Invalid number of internal momenta for the modular kernel."
                    )
                if len(self.external_momenta) != 1:
                    raise ValueError(
                        "Invalid number of external momenta for the modular kernel."
                    )
                return mkernel.KernelsM(
                    self.Theory,
                    self.external_momenta[0],
                    self.internal_momenta[0],
                    self.internal_momenta[1],
                    self.kernel_code
                )
    
    def getTheory(self):
        return self.Theory
    
    def getkernel_type(self):
        return self.kernal_type
    
    def getexternal_momenta(self):
        return self.external_momenta
    
    def getinternal_momenta(self):
        return self.internal_momenta
    
    def getkernel_code(self):
        return self.kernel_code
    
    def getvalue(self):
        return self.value
    
    def getTheory(self):
        return self.Theory
    
    def getkernel_type(self):
        return self.kernal_type
    
    def getexternal_momenta(self):
        return self.external_momenta
    
    def getinternal_momenta(self):
        return self.internal_momenta
    
    def getkernel_code(self):
        return self.kernel_code
    
    def getvalue(self):
        return self.value


if __name__ == '__main__':

    z = 0.1+0.86j
    P1 = 0.1001+0.032j
    P2 = 0.02+0.76j
    P3 = 0.04+0.0084j
    P4 = 0.00776+0.0023j
    Ps = 0.125+0.097j
    Pt = 0.00942+0.012j
    theo = Theory(m=2,n=1)
    internal_momenta = [Ps,Pt]
    internal_momenta_reflected = [-Ps,Pt]
    external_momenta = [P1,P2,P3,P4]
    #Consistency checks for the quantum modular fusion polynomial
    print("=== Demo I: Consistency checks for the quantum modular fusion polynomial ===")
    print()
    # --- Demo 1: alpha at zero point ---
    # It should be 0 for b=1
    alpha_demo = Quantummodularfusion.alphaf(theo, 0, 0, 0, 0, 0, 0)
    print("=== Demo: alpha(0,0) ===")
    print(f"alpha = {alpha_demo}")
    print()

    #--- Demo 2: Check that the roots obtained from the function roots(Theory: Theory,Ps,Pt) are the same that the ones obtain from (-beta \pm sqrt(beta^2-4*alpha*gamma)/2*alpha) ---
    z_plus, z_minus = Quantummodularfusion.rootsf(theo, P1, P2, P3, P4, Ps, Pt)
    alpha_val = Quantummodularfusion.alphaf(theo, P1, P2, P3, P4, Ps, Pt)
    beta_val = Quantummodularfusion.betaf(theo, P1, P2, P3, P4, Ps, Pt)
    gamma_val = Quantummodularfusion.gammaf(theo, P1, P2, P3, P4, Ps, Pt)
    z_plus_check = (-beta_val + mp.sqrt(beta_val**2 - 4*alpha_val*gamma_val)) / (2 * alpha_val)
    z_minus_check = (-beta_val - mp.sqrt(beta_val**2 - 4*alpha_val*gamma_val)) / (2 * alpha_val)
    print("=== Demo: roots consistency check ===")
    print(f"z_plus from roots() = {z_plus}")
    print(f"z_plus from formula   = {z_plus_check}")
    print(f"z_minus from roots() = {z_minus}")
    print(f"z_minus from formula   = {z_minus_check}\n")
    print()

    #Consistency checks spacelike fusion kernels
    print("=== Demo II: Consistency checks for the spacelike fusion kernels ===")
    print()
    #--- Demo 3: Check the symmetry of CurlyF under the transformation P_s\rightarrow P_3,P_3\rightarrow P_s,P_t\rightarrow P_1,P_1\rightarrow P_t ---
    CurlyF_original = fkernel.CurlyF(theo, P1, P2, P3, P4, Ps, Pt, z)
    CurlyF_transformed = fkernel.CurlyF(theo, Pt, P2, Ps, P4, P3, P1, z)
    print("=== Demo: CurlyF symmetry check ===")
    print(f"CurlyF original     = {CurlyF_original  }")
    print(f"CurlyF transformed  = {CurlyF_transformed}")
    print()
    # --- Demo 4: Check the single-valuedness of the sum in the Kernels function ---
    z_plus, z_minus = Quantummodularfusion.rootsf(theo, P1, P2, P3, P4, Ps, Pt)

    N = theo.m * theo.n
    s = theo.s
    ell = 3

    res1 = mp.mpc(0)
    res2 = mp.mpc(0)

    u0 = mp.log(z_plus) / (2j * mp.pi * s**2)

    for k in range(N):
        z1 = u0 + mp.mpf(k) / N
        z2 = u0 + mp.mpf(k + ell) / N
        res1 += fkernel.CurlyF(theo, P1, P2, P3, P4, Ps, Pt, z1)
        res2 += fkernel.CurlyF(theo, P1, P2, P3, P4, Ps, Pt, z2)

    print("=== Demo: CurlyF single-valuedness check ===")
    print(f"sum unshifted = {res1}")
    print(f"sum shifted   = {res2}")
    print()
    #--- Demo5: Check the reflection property of the spacelike fusion kernels.
    Fplus= kernel_rational(theo,"spacelike", "fusion", external_momenta, internal_momenta, 1).getvalue()
    Fminus= kernel_rational(theo,"spacelike", "fusion", external_momenta, internal_momenta, -1).getvalue()
    Fplusreflected=kernel_rational(theo,"spacelike", "fusion", external_momenta, internal_momenta_reflected, 1).getvalue()
    Fminusreflected=kernel_rational(theo,"spacelike", "fusion", external_momenta, internal_momenta_reflected, -1).getvalue()
    F_TV = kernel_rational(theo,"spacelike", "fusion", external_momenta, internal_momenta, 0).getvalue()
    FTVreflected= kernel_rational(theo,"spacelike", "fusion", external_momenta, internal_momenta_reflected, 0).getvalue()
    print("=== Demo: Reflection property of the plus/minus kernels ===")
    print(f"Fplus = {Fplus}")
    print(f"Fminus   = {Fminus}")
    print(f"Fplusreflected = {Fplusreflected}")
    print(f"Fminusreflected   = {Fminusreflected}")
    print(f"FTeschnerVartanov = {F_TV}")
    print(f"FTeschnerVartanovreflected = {FTVreflected}")
    print()

    #Consistency checks spacelike modular kernels
    print("=== Demo III: Consistency checks for the spacelike modular kernels ===")
    print()
    # --- Demo 6: Check the single-valuedness of the sum in the modular Kernels function ---
    z_plus, z_minus = Quantummodularmodular.rootsm(theo, P1, Ps, Pt)

    N = theo.m * theo.n
    s = theo.s
    ell = 3

    res1 = mp.mpc(0)
    res2 = mp.mpc(0)

    u0 = mp.log(z_plus) / (2j * mp.pi * s**2)

    for k in range(N):
        z1 = u0 + mp.mpf(k) / N
        z2 = u0 + mp.mpf(k + ell) / N
        res1 += mkernel.CurlyM(theo, P1, Ps, Pt, z1)
        res2 += mkernel.CurlyM(theo, P1, Ps, Pt, z2)

    print("=== Demo: CurlyM single-valuedness check ===")
    print(f"sum unshifted = {res1}")
    print(f"sum shifted   = {res2}")
    print()

    # #--- Demo 7: Check reflection properties under reflection of the external momenta
    print("=== Demo: Reflection property of Modular kernels ===")
    print(f"Mplus = {kernel_rational(theo,"spacelike", "modular", [P1], [Ps,Pt], 1).getvalue()}")
    print(f"Mminus = {kernel_rational(theo,"spacelike", "modular", [P1], [Ps,Pt], -1).getvalue()}")
    print(f"Mplusreflectedps = {kernel_rational(theo,"spacelike", "modular", [P1], [-Ps,Pt], 1).getvalue()}")
    print(f"Mminusreflectedps = {kernel_rational(theo,"spacelike", "modular", [P1], [-Ps,Pt], -1).getvalue()}")
    print(f"Mplusreflectedpt = {kernel_rational(theo,"spacelike", "modular", [P1], [Ps,-Pt], 1).getvalue()}")
    print(f"Mminusreflectedpt = {kernel_rational(theo,"spacelike", "modular", [P1], [Ps,-Pt], -1).getvalue()}")
    print(f"MTV = {kernel_rational(theo,"spacelike", "modular", [P1], [Ps,Pt], 0).getvalue()}")
    print(f"MTVreflectedps = {kernel_rational(theo,"spacelike", "modular", [P1], [-Ps,Pt], 0).getvalue()}")
    print(f"MTVreflectedpt = {kernel_rational(theo,"spacelike", "modular", [P1], [Ps,-Pt], 0).getvalue()}")
    print()

    print("=== TIMELIKE LIOUVILLE !!! ===")
    print()

    
    #--- Demo20: Check the reflection property of the timelike fusion kernels.
    Fplustimelike= kernel_rational(theo,"timelike", "fusion", external_momenta, internal_momenta, 1).getvalue()
    Fminustimelike= kernel_rational(theo,"timelike", "fusion", external_momenta, internal_momenta, -1).getvalue()
    Fplusreflectedtimelike=kernel_rational(theo,"timelike", "fusion", external_momenta, internal_momenta_reflected, 1).getvalue()
    Fminusreflectedtimelike=kernel_rational(theo,"timelike", "fusion", external_momenta, internal_momenta_reflected, -1).getvalue()
    F_TVtimelike = kernel_rational(theo,"timelike", "fusion", external_momenta, internal_momenta, 0).getvalue()
    FTVreflectedtimelike= kernel_rational(theo,"timelike", "fusion", external_momenta, internal_momenta_reflected, 0).getvalue()
    print("=== Demo: Reflection property of the plus/minus kernels ===")
    print(f"Fplus = {Fplustimelike}")
    print(f"Fminus   = {Fminustimelike}")
    print(f"Fplusreflected = {Fplusreflectedtimelike}")
    print(f"Fminusreflected   = {Fminusreflectedtimelike}")
    print(f"FTeschnerVartanov = {F_TVtimelike}")
    print(f"FTeschnerVartanovreflected = {FTVreflectedtimelike}")
    print()



    