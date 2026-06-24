#Packages
from Special_functions_rational import Theory, special_functions_rational
from mpmath import mp



class spacelike_CFT_data:

    @staticmethod
    def Plancherel_measure(Theory: Theory, P: complex):
        b = Theory.b
        P = mp.mpc(P)
        rho_b = -4*mp.sqrt(2)*mp.sin(2*mp.pi*b*P)*mp.sin(2*mp.pi*P/b)
        return rho_b

    @staticmethod
    def spacelikeB_b(Theory: Theory, P:complex):
        P = mp.mpc(P)
        return 1/spacelike_CFT_data.Plancherel_measure(Theory, P)

    @staticmethod
    def spacelikeC_b(Theory: Theory, P1:complex, P2:complex, P3:complex):
        Q = Theory.Q
        P1, P2, P3 = mp.mpc(P1), mp.mpc(P2), mp.mpc(P3)

        Numerator = (
        special_functions_rational.Gamma_b(Theory, 2*Q)
        * special_functions_rational.Gamma_b(Theory, Q/2 + P1 + P2 + P3)
        * special_functions_rational.Gamma_b(Theory, Q/2 + P1 - P2 + P3)
        * special_functions_rational.Gamma_b(Theory, Q/2 + P1 + P2 - P3)
        * special_functions_rational.Gamma_b(Theory, Q/2 + P1 - P2 - P3)
        * special_functions_rational.Gamma_b(Theory, Q/2 - P1 + P2 + P3)
        * special_functions_rational.Gamma_b(Theory, Q/2 - P1 + P2 - P3)
        * special_functions_rational.Gamma_b(Theory, Q/2 - P1 - P2 + P3)
        * special_functions_rational.Gamma_b(Theory, Q/2 - P1 - P2 - P3)
        )

        Denominator = (
        mp.sqrt(2) * special_functions_rational.Gamma_b(Theory, Q)**3
        * special_functions_rational.Gamma_b(Theory, Q + 2*P1) * special_functions_rational.Gamma_b(Theory, Q - 2*P1)
        * special_functions_rational.Gamma_b(Theory, Q + 2*P2) * special_functions_rational.Gamma_b(Theory, Q - 2*P2)
        * special_functions_rational.Gamma_b(Theory, Q + 2*P3) * special_functions_rational.Gamma_b(Theory, Q - 2*P3)
        )

        return Numerator / Denominator

class timelike_CFT_data:

    @staticmethod
    def Plancherel_measure_timelike(Theory: Theory, P: complex):
        P = mp.mpc(P)
        return P**2 / (2*spacelike_CFT_data.Plancherel_measure(Theory, 1j*P))

    @staticmethod
    def timelikeB_b(Theory: Theory, P:complex):
        P =mp.mpc(P)

        return 2/(P**2*spacelike_CFT_data.spacelikeB_b(Theory, 1j*P))

    @staticmethod
    def timelikeC_b(Theory:Theory, P1:complex, P2:complex, P3:complex):
        return 1/spacelike_CFT_data.spacelikeC_b(Theory, 1j*P1, 1j*P2, 1j*P3)

    @staticmethod
    def PHI_b(Theory, P1:complex, P2:complex, P3:complex):
        return (mp.mpc(P2)*timelike_CFT_data.Plancherel_measure_timelike(Theory, P1)*timelike_CFT_data.timelikeC_b(Theory, P1, P2, P3))/(mp.mpc(P1))


if __name__ == '__main__':
    theo = Theory(m=3,n=2)
    P1 = 0.0812j
    P2 = 0.314j
    # --- Demo : identitylimit of the 3 point structure constants ---
    print("=== Demo: Identity limit of the 3-point structure constants ===")
    Cidentity = spacelike_CFT_data.spacelikeC_b(theo, P1, P2, theo.Q/2+0.000001j)
    print(f"C(P1, P2, Q/2) = {Cidentity}")
    print()