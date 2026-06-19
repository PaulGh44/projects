#Packages
from dataclasses import dataclass
from mpmath import mp, barnesg


@dataclass
class Theory:
    m: int
    n: int
    

    @property
    def b(self):
        return mp.sqrt(mp.mpf(self.m) / mp.mpf(self.n))

    @property
    def s(self):
        return mp.sqrt(mp.mpf(self.m) * mp.mpf(self.n))
    
    @property
    def Q(self):
        return self.b+1/self.b
    


class special_functions_rational:

# =========================================
# Special functions: Barnes G products and double gamma function for rational b^2
# =========================================
    @staticmethod
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

    @staticmethod
    def Gtilde(z:complex):
        """
        \tilde G(z) = G(1+z)/G(1-z)
        """
        z = mp.mpc(z)
        return barnesg(1 + z) / barnesg(1 - z)

    @staticmethod
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
                prod *= special_functions_rational.Gtilde(z + shift - 1)
        return prod

    @staticmethod
    def Gamma_b(Theory: Theory, z:complex):
        m, n = Theory.m, Theory.n
        b = Theory.b
        Q = b + 1/b
        z = mp.mpc(z)
        gamma_mn = (m*n)**(-Q**2/16)*special_functions_rational.Gmn(Theory, (1/m + 1/n)/2)
        Numerator = gamma_mn * (m*n)**(z/4*(Q-z))*(2*mp.pi)**(z*mp.sqrt(m*n)/2 - (m+n)/4)
        Denominator = special_functions_rational.Gmn(Theory, z/mp.sqrt(m*n))

        return Numerator/Denominator


#Consistency checks for the special functions
if __name__ == '__main__':
    theo=Theory(m=3,n=1)
    print("=== Demo I: Consistency checks for the special functions ===")
    print()
    # --- Demo 1: shift identity you’re currently testing ---
    z = mp.mpc(0.6+0.8j)
    z_plus = z + mp.mpf(1) / mp.mpf(theo.n)

    ratio = special_functions_rational.Gmn(theo, z_plus) / special_functions_rational.Gmn(theo, z)

    m_mp = mp.mpf(theo.m)
    rhs = mp.power(m_mp, mp.mpf(1)/2 - m_mp * z) \
          * mp.power(2 * mp.pi, (m_mp - mp.mpf(1)) / 2) \
          * mp.gamma(m_mp * z)

    print("=== Demo: Gmn shift check ===")
    print(f"ratio = {ratio}")
    print(f"rhs   = {rhs}")
    print()

    # --- Demo 2: conjugation & inversion checks for Gtilde_mn ---
    Gv = special_functions_rational.Gtilde_mn(theo, z)
    print("=== Demo: Gtilde_mn checks ===")
    print(f"conj(Gtilde_mn(z))     = {mp.conj(Gv)}")
    print(f"Gtilde_mn(conj(z))     = {special_functions_rational.Gtilde_mn(theo, mp.conj(z))}")
    print(f"Gtilde_mn(-z)          = {special_functions_rational.Gtilde_mn(theo, -z)}")
    print(f"1/Gtilde_mn(z)         = {1/Gv}\n")
    print()

    # --- Demo 2bis: Identity limit for Gamma_b ---
    Gamma_bidentity = special_functions_rational.Gamma_b(theo, (theo.b + 1/theo.b)/2)
    print("=== Demo: Gamma_b identity limit ===")
    print(f"Gamma_b((b + 1/b)/2) = {Gamma_bidentity}")
    print()

    # --- Demo 2ter: shift equations ---
    def shiftrelationsGamma_b(Theory: Theory, z:complex, epsilon: int):
        b =Theory.b
        LHS = special_functions_rational.Gamma_b(Theory,z+b**(epsilon))/special_functions_rational.Gamma_b(Theory,z)
        RHS = mp.sqrt(2*mp.pi)*b**(epsilon*b**(epsilon)*z -epsilon/2)/(mp.gamma(b**(epsilon)*z))

        return LHS, RHS
    
    LHSb, RHSb = shiftrelationsGamma_b(theo, z, 1)
    LHSbminus1, RHSbminus1 = shiftrelationsGamma_b(theo, z, -1)
    print("=== Demo: Gamma_b shift equations ===")
    print(f"Gamma_b(z + b)   = {LHSb}  ||  b^((1/2) - z) * Gamma_b(z) / sqrt(2*pi) = {RHSb}")
    print(f"Gamma_b(z - b)  = {LHSbminus1}  ||  b^((z - (1/2))) * Gamma_b(z) * sqrt(2*pi) = {RHSbminus1}")
    print()
    
    
    
    




    



    
    




    


    
    

    

    
    
