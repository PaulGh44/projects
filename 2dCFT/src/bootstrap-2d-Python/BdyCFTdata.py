#Packages
from Kernel_rational import kernel_rational
from Special_functions_rational import Theory




class spacelikebdy_CFT_data:
    
    @staticmethod
    def OPEdata_spacelikeC_b(Theory: Theory, P1:complex, P2:complex, P3:complex, Psigma1:complex, Psigma2:complex, Psigma3:complex):
        return 2*kernel_rational(Theory, "spacelike", "fusion", [Psigma3, P2, P1, Psigma1], [Psigma2, P3], 0).getvalue()