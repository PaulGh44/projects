#Packages
from Kernel_rational import kernel_rational
from Special_functions_rational import Theory
from BulkCFTdata import spacelike_CFT_data




class spacelikebdy_CFT_data:
    
    @staticmethod
    def OPEdata_spacelikeC_b(Theory: Theory, P1:complex, P2:complex, P3:complex, Psigma1:complex, Psigma2:complex, Psigma3:complex):
        return 2*kernel_rational(Theory, "spacelike", "fusion", [Psigma3, P2, P1, Psigma1], [Psigma2, P3], 0).getvalue()

    @staticmethod
    def bdy_to_bulk_data_spacelikeR_b(Theory: Theory, P1:complex, P2:complex, Psigma:complex):
        return 2*kernel_rational(Theory, "spacelike", "modular", [P2], [Psigma,P1], 0).getvalue()/spacelike_CFT_data.Plancherel_measure(Theory, P1)