#Packages
from mpmath import mp
from Blocks import Block, BlockNum
from CFT import Charge, Dimension
import cmath, math
from Special_functions_rational import Theory
from Kernel_rational import kernel_rational
from BulkCFTdata import spacelike_CFT_data, timelike_CFT_data
from BdyCFTdata import spacelikebdy_CFT_data
from Utils_plot import utils_plot

from pathlib import Path
from datetime import datetime



# -------- Global settings --------
mp.dps = 10 #be careful with the precision, if the precision is not enough you might raise "Zero Division error"

# =========================================
# Conformal dimension relation
# =========================================
def conformaldimension(Theory: Theory, P:complex):
    Q = Theory.b + 1/ Theory.b
    return Q**2/4 - P**2

def complex_to_str(z: complex, digits: int = 16) -> str:
    z = mp.mpc(z)
    return (
        f"{mp.nstr(mp.re(z), digits)}"
        f"{mp.nstr(mp.im(z), digits, strip_zeros=False, min_fixed=0)}j"
    )
# =========================================
# Identities to be checked for the fusion kernel
# =========================================

def mixingsymmetrieskernelsspacelikef(Theory: Theory, P1: complex, P2: complex, P3: complex, P4: complex,
            Ps: complex, Pt: complex, kernel_code: int):
    b = Theory.b
    vacuummomentum=(b + 1/b)/2 +0.001j #for this momentum = P<1,1>, there is a "fake pole" numerically! For this reason, I shifted a bit by giving them a small imaginary part
    RHS = kernel_rational(Theory, "spacelike", "fusion", [P2,P2,P3,P3], [vacuummomentum,Pt], 0).getvalue()*kernel_rational(Theory, "spacelike", "fusion", [Pt,P2,Ps,P4], [P3,P1],kernel_code).getvalue()/kernel_rational(Theory, "spacelike", "fusion", [P2,P2,Ps,Ps], [vacuummomentum,P1],0).getvalue()
    LHS = kernel_rational(Theory, "spacelike", "fusion", [P1,P2,P3,P4], [Ps,Pt], kernel_code).getvalue()
    
    return LHS,RHS

# This is the identity that Ionnis wrote down in his paper with Sylvain
def RibaultTsiaresspacelikef(Theory: Theory,Ps:complex, Pt:complex, kernel_code:int):
    m, n = Theory.m, Theory.n
    if (m!=1 or n!=1):
        print("The central charge is not 25")
        return
    else:
        if (kernel_code==1):
            return 1j*Pt*16**(Pt**2-Ps**2)*mp.exp(2j*mp.pi*Ps*Pt)/Ps
        elif (kernel_code==-1):
            return -1j*Pt*16**(Pt**2-Ps**2)*mp.exp(-2j*mp.pi*Ps*Pt)/Ps
        else:
            raise ValueError("The Kernel code must be either 1 or -1")

def SpacelikePentagon(Theory: Theory, Po: complex, Pq: complex, P1: complex, P2: complex, P3: complex, P4:complex, P5:complex, Pt:complex, Pu:complex, kernel_code:int):
    IntegrandLHS = lambda ps: kernel_rational(Theory, "spacelike", "fusion", [Pu, 1j*(ps+0.0001j), P3, P5], [Pq, P4], 0).getvalue()*kernel_rational(Theory, "spacelike", "fusion", [Pu, P1, P2, Pq], [Po, 1j*(ps+0.0001j)], 0).getvalue()*kernel_rational(Theory, "spacelike", "fusion", [P1, P2, P3, P4], [1j*(ps+0.0001j), Pt], kernel_code).getvalue()

    LHS = mp.quad(IntegrandLHS, [-Theory.Lambda, Theory.Lambda])

    RHS = kernel_rational(Theory, "spacelike", "fusion", [Pu,P1,Pt,P5], [Po, P4], 0).getvalue()*kernel_rational(Theory, "spacelike", "fusion", [Po,P2,P3,P5], [Pq, Pt], 0).getvalue()

    return LHS, RHS

def Identitylimit(Theory: Theory, Ps: complex, Pt: complex):
    ident = -(Theory.b+1/Theory.b + 0.000000001j)/2 # This is to avoid a fake pole
    return kernel_rational(Theory, "spacelike", "modular", [ident], [Ps, Pt], 0).getvalue(), mp.sqrt(2)*mp.cos(4*mp.pi*Ps*Pt)

# =========================================
# Identities to be checked for the modular kernel
# =========================================

def mixingsymmetriesm(Theory: Theory, P0: complex, P1: complex, P2: complex, kernel_code: int ):
    LHS = kernel_rational(Theory, "spacelike", "modular", [P0], [P1, P2], kernel_code).getvalue()
    RHS = spacelike_CFT_data.spacelikeC_b(Theory, P2, P2, P0)*spacelike_CFT_data.spacelikeB_b(Theory, P1)*kernel_rational(Theory, "spacelike", "modular", [P0], [P2, P1], kernel_code).getvalue()/(spacelike_CFT_data.spacelikeC_b(Theory, P1, P1, P0)*spacelike_CFT_data.spacelikeB_b(Theory, P2))
    return LHS, RHS

# =========================================
# Mixed checks involving both the modular and the fusion kernels
# =========================================
def NonrationalVerlinde(Theory: Theory, P0:complex, P1: complex, P2:complex, P3:complex, kernel_code:int):
    b = Theory.b
    ident = -(b+1/b + 0.00000001j)/2

    Integrand = lambda p: (kernel_rational(Theory, "spacelike", "modular", [P0], [P1,1j*(p+0.00000001j)] ,kernel_code).getvalue()*kernel_rational(Theory, "spacelike", "modular", [ident], [P2,1j*(p+0.00000001j)], kernel_code).getvalue()*kernel_rational(Theory, "spacelike", "modular", [P0], [1j*(p+0.00000001j),P3] ,kernel_code).getvalue()/(-2*mp.sqrt(2)*mp.sin(2*mp.pi*b*1j*(p+0.00000001j))*mp.sin(2*mp.pi*1j*(p+0.00000001j)/b)))

    RHS = mp.quad(Integrand, [-Theory.Lambda, Theory.Lambda])
    
    LHS = kernel_rational(Theory, "spacelike", "fusion", [P1,P0,P3,P2] , [P1,P3], kernel_code).getvalue()

    return LHS, RHS

def Torus2pointrelation(Theory: Theory, P0:complex, P0prime:complex, P1:complex, P2:complex, P3:complex, P5:complex, kernel_code:int):
    b = Theory.b
    shiftangle = mp.pi/12
    shiftphase = mp.exp(1j*shiftangle)

    IntegrandLHS = lambda p4: shiftphase*kernel_rational(Theory,"spacelike", "fusion", [P0, P0prime, P2, P2] , [P3, 1j*(p4*shiftphase+0.000001j)],kernel_code).getvalue()*kernel_rational(Theory, "spacelike", "fusion", [P2, P0prime, P0, P2] , [1j*(p4*shiftphase+0.000001j), P5] , kernel_code).getvalue()*mp.exp(1j*(2*conformaldimension(Theory, 1j*(p4*shiftphase+0.000001j))-2*conformaldimension(Theory,P2)+conformaldimension(Theory,P3)/2)*mp.pi)

    IntegrandRHS = lambda p6: kernel_rational(Theory, "spacelike", "fusion", [P0prime, P0, P1, P1], [P3, 1j*(p6+0.000001j)], kernel_code).getvalue()*kernel_rational(Theory, "spacelike", "fusion", [1j*(p6+0.000001j), P0prime, P0, 1j*(p6+0.000001j)], [P1, P5], kernel_code).getvalue()*mp.exp(1j*(conformaldimension(Theory, P0)+conformaldimension(Theory,P0prime)-conformaldimension(Theory,P5)/2)*mp.pi)*kernel_rational(Theory, "spacelike", "modular", [P5], [1j*(p6+0.000001j), P2], kernel_code).getvalue()

    LHS = kernel_rational(Theory, "spacelike", "modular", [P3], [P1, P2], kernel_code).getvalue()*mp.quad(IntegrandLHS, [-Theory.Lambda, Theory.Lambda])
    RHS = mp.quad(IntegrandRHS, [-Theory.Lambda, Theory.Lambda])
    return LHS, RHS


# # =========================================
# # Consistency checks for the fusion kernel
# # =========================================

def RibaultTsiarestimelikef(Theory: Theory,Ps:complex, Pt:complex, kernel_code:int):
    m, n = Theory.m, Theory.n
    if (m!=1 or n!=1):
        print("The central charge is not 1")
        return
    else:
        if (kernel_code==1):
            return 16**(Pt**2-Ps**2)*mp.exp(-2j*mp.pi*Ps*Pt)
        elif (kernel_code==-1):
            return 16**(Pt**2-Ps**2)*mp.exp(2j*mp.pi*Ps*Pt)
        else:
            raise ValueError("The Kernel code must be either 1 or -1")
        
def mixingsymmetriestimelikekernelsf(Theory: Theory, P1: complex, P2: complex, P3: complex, P4: complex,
            Ps: complex, Pt: complex, kernel_code: int):
    
    LHS = kernel_rational(Theory, "timelike", "fusion", [P1, P2, P3, P4], [Ps, Pt], kernel_code).getvalue()
    
    RHS = timelike_CFT_data.PHI_b(Theory, P3, Pt, P2)*kernel_rational(Theory, "timelike", "fusion", [Pt, P2, Ps, P4], [P3, P1], kernel_code).getvalue()/timelike_CFT_data.PHI_b(Theory, Ps, P1, P2)

    return LHS, RHS

# # This returns the LHS and the RHS of the Pentagon identity
# def TimelikePentagon(Theory: Theory, Param: ParametersPentagon, epsilon: int):
#     Po, Pq, P1, P2, P3, P4, P5, Pt, Pu = (
#         Param.Po, Param.Pq, Param.P1, Param.P2,
#         Param.P3, Param.P4, Param.P5, Param.Pt, Param.Pu
#     )

#     IntegrandLHS = lambda ps: (
#         TimelikeKernelsf(Theory, Pu, 1j*(ps+1.5j), P3, P5, Pq, P4, epsilon)
#         * TimelikeKernelsf(Theory, Pu, P1, P2, Pq, Po, 1j*(ps+1.5j), epsilon)
#         * TimelikeKernelsf(Theory, P1, P2, P3, P4, 1j*(ps+1.5j), Pt, 0)
#     )

#     LHS = mp.quad(IntegrandLHS, [-Theory.Lambda, Theory.Lambda])

#     RHS = (
#         TimelikeKernelsf(Theory, Pu, P1, Pt, P5, Po, P4, epsilon)
#         * TimelikeKernelsf(Theory, Po, P2, P3, P5, Pq, Pt, epsilon)
#     )

#     return LHS, RHS


# =========================================
# Merging the blocks with the Kernels
# =========================================

def four_point_block_crossing(
    Theory: Theory,
    Liouville_type: str,
    kernel_code: int,
    P1: complex,
    P2: complex,
    P3: complex,
    P4: complex,
    Ps: complex,
    q: complex = 0.1,
    N: int = 10,
    eps_b = 1e-5
    # It should not be too small, otherwise it blows up near b=1
):

    b_block = complex(Theory.b)+1j*eps_b
    if (Liouville_type == "spacelike"):
        charge = Charge("b", b_block)
        contour_shift = 0.00001
    elif (Liouville_type == "timelike"):
        charge = Charge("b", 1j*b_block)
        contour_shift = 0.3
    else:
        raise ValueError("Invalid Liouville_type. It should be either 'spacelike' or 'timelike'.")

    P1, P2, P3, P4, Ps = map(complex, [P1, P2, P3, P4, Ps])
    q = complex(q)

    dims_momenta_s = [
        Dimension("P", 1j * P, charge)
        for P in [P1, P2, P3, P4]
    ]

    q_rhs = cmath.exp(math.pi**2 / cmath.log(q))
    dims_momenta_rhs = [
    Dimension("P", 1j * P, charge)
    for P in [P1, P4, P3, P2]
    ]

    block_s = Block(dims_momenta_s, N, t_channel=False)
    block_t = Block(dims_momenta_rhs, N, t_channel=False)

    num_s = BlockNum(block_s, q)
    num_t = BlockNum(block_t,q_rhs)

    #This is the s-channel conformal block
    val_s = num_s.value(-Ps**2, True)
    # There is a thing in Sylvain's code which is $\delta = -P**2$ which is an important parameter

    Integrand = lambda pt: mp.mpc(num_t.value((pt+contour_shift)**2, True))*kernel_rational(Theory, Liouville_type, "fusion", [P1, P2, P3, P4], [Ps, 1j*(pt+contour_shift)], kernel_code).getvalue()

    RHS = mp.quad(Integrand, [-Theory.Lambda, Theory.Lambda])

    return val_s, RHS, num_s.x, num_t.x

def four_point_block_crossing_bdy(
    Theory: Theory,
    P1: complex,
    P2: complex,
    P3: complex,
    P4: complex,
    Psigma1: complex,
    Psigma2: complex,
    Psigma3: complex,
    Psigma4: complex,
    q: complex = 0.05,
    N: int = 10,
    eps_b = 1e-5
    # It should not be too small, otherwise it blows up near b=1
):

    b_block = complex(Theory.b)+1j*eps_b
    charge = Charge("b", b_block)

    P1, P2, P3, P4 = map(complex, [P1, P2, P3, P4])
    q = complex(q)

    dims_momenta_s = [
        Dimension("P", 1j * P, charge)
        for P in [P1, P2, P3, P4]
    ]

    q_rhs = cmath.exp(math.pi**2 / cmath.log(q))
    dims_momenta_rhs = [
    Dimension("P", 1j * P, charge)
    for P in [P1, P4, P3, P2]
    ]



    block_s = Block(dims_momenta_s, N, t_channel=False)
    block_t = Block(dims_momenta_rhs, N, t_channel=False)

    num_s = BlockNum(block_s, q)
    num_t = BlockNum(block_t,q_rhs)


    IntegrandLHS = lambda ps: mp.mpc(num_s.value((ps+0.000001j)**2, True))*spacelikebdy_CFT_data.OPEdata_spacelikeC_b(Theory, P1, P2, 1j*(ps+0.000001j), Psigma1, Psigma2, Psigma3)*spacelikebdy_CFT_data.OPEdata_spacelikeC_b(Theory, 1j*(ps+0.000001j), P3, P4, Psigma1, Psigma3, Psigma4)

    IntegrandRHS = lambda pt: mp.mpc(num_t.value((pt+0.000001j)**2, True))*spacelikebdy_CFT_data.OPEdata_spacelikeC_b(Theory, P2, P3, 1j*(pt+0.000001j), Psigma2, Psigma3, Psigma4)*spacelikebdy_CFT_data.OPEdata_spacelikeC_b(Theory, P1, 1j*(pt+0.000001j), P4, Psigma1, Psigma2, Psigma4)

    LHS = mp.quad(IntegrandLHS, [-Theory.Lambda, Theory.Lambda])
    RHS = mp.quad(IntegrandRHS, [-Theory.Lambda, Theory.Lambda])

    return LHS, RHS, num_s.x

def four_point_block_crossing_bdy_generate_data(
    Theory: Theory,
    P1: complex,
    P2: complex,
    P3: complex,
    P4: complex,
    Psigma1: complex,
    Psigma2: complex,
    Psigma3: complex,
    Psigma4: complex,
    listq: list[complex],
    N: int = 10,
    eps_b=1e-5,
    save_path: str | None = None,
):
    xlist = []
    Relist = []
    Imlist = []
    qlist = []

    for q in listq:
        LHS, RHS, x = four_point_block_crossing_bdy(
            Theory,
            P1,
            P2,
            P3,
            P4,
            Psigma1,
            Psigma2,
            Psigma3,
            Psigma4,
            q,
            N=N,
            eps_b=eps_b,
        )

        err = LHS/RHS - 1

        qlist.append(q)
        Relist.append(float(abs(mp.re(err))))
        Imlist.append(float(abs(mp.im(err))))
        xlist.append(float(mp.re(x)))

    if save_path is not None:
        save_path = Path(save_path)

        parameters = {
            "P1": P1,
            "P2": P2,
            "P3": P3,
            "P4": P4,
            "Psigma1": Psigma1,
            "Psigma2": Psigma2,
            "Psigma3": Psigma3,
            "Psigma4": Psigma4,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("# four_point_block_crossing_bdy_generate\n")
            f.write(f"# created_at = {datetime.now().isoformat()}\n")
            f.write(f"# m = {Theory.m}\n")
            f.write(f"# n = {Theory.n}\n")
            f.write(f"# b = {mp.nstr(Theory.b, 16)}\n")
            f.write(f"# N = {N}\n")
            f.write(f"# eps_b = {eps_b}\n")

            for name, value in parameters.items():
                f.write(f"# {name} = {complex_to_str(value)}\n")

            f.write("#\n")
            f.write("# columns: q_real q_imag x abs_Re_rel_error abs_Im_rel_error\n")

            for q, x, re_err, im_err in zip(qlist, xlist, Relist, Imlist):
                q = mp.mpc(q)
                f.write(
                    f"{mp.nstr(mp.re(q), 18)}\t"
                    f"{mp.nstr(mp.im(q), 18)}\t"
                    f"{x:.18e}\t"
                    f"{re_err:.18e}\t"
                    f"{im_err:.18e}\n"
                )

        print(f"Saved data to {save_path}")

    return Relist, Imlist, xlist

# def three_point_block_crossing_bdy(
#     Theory: Theory,
#     Palpha: complex,
#     P1: complex,
#     P2: complex,
#     Psigma1: complex,
#     Psigma2: complex,
#     eta: complex,
#     N: int = 10,
#     eps_b = 1e-5
#     # It should not be too small, otherwise it blows up near b=1
# ):

#     b_block = complex(Theory.b)+1j*eps_b
#     charge = Charge("b", b_block)
    

#     Palpha, P1, P2 = map(complex, [Palpha, P1, P2])
#     q = complex(mp.qfrom(m=eta))
#     etaprime = eta.conjugate()
#     qprime = complex(mp.qfrom(m=etaprime))

#     dims_momenta = [
#     Dimension("P", 1j * P, charge)
#     for P in [Palpha, Palpha, P1, P2]
#     ]


#     block = Block(dims_momenta, N, t_channel=False)

#     num_q = BlockNum(block, q)
#     num_qprime = BlockNum(block, qprime)


#     IntegrandLHS = lambda ps: mp.mpc(num_q.value((ps+0.000001j)**2, False))*spacelikebdy_CFT_data.OPEdata_spacelikeC_b(Theory, P1, 1j*(ps+0.000001j), P2, Psigma1, Psigma2, Psigma2)*spacelikebdy_CFT_data.bdy_to_bulk_data_spacelikeR_b(Theory, Palpha, 1j*(ps+0.000001j), Psigma2)*mp.exp(-1j*mp.pi*(conformaldimension(Theory,1j*(ps+0.000001j))-2*conformaldimension(Theory, Palpha))/2)

#     IntegrandRHS = lambda psprime: mp.mpc(num_qprime.value((psprime+0.000001j)**2, False))*spacelikebdy_CFT_data.OPEdata_spacelikeC_b(Theory, 1j*(psprime+0.000001j), P1, P2, Psigma1, Psigma1, Psigma2)*spacelikebdy_CFT_data.bdy_to_bulk_data_spacelikeR_b(Theory, Palpha, 1j*(psprime+0.000001j), Psigma1)*mp.exp(1j*mp.pi*(conformaldimension(Theory,1j*(psprime+0.000001j))-2*conformaldimension(Theory,Palpha))/2)

#     LHS = mp.quad(IntegrandLHS, [-Theory.Lambda, Theory.Lambda])*(eta)**(conformaldimension(Theory,P1)-conformaldimension(Theory,P2))*(1-eta)**(conformaldimension(Theory, Palpha)+(conformaldimension(Theory,P2)-conformaldimension(Theory,P1))/2)
#     RHS = mp.quad(IntegrandRHS, [-Theory.Lambda, Theory.Lambda])*(etaprime)**(conformaldimension(Theory,P1)-conformaldimension(Theory,P2))*(1-etaprime)**(conformaldimension(Theory, Palpha)+(conformaldimension(Theory,P2)-conformaldimension(Theory,P1))/2)

#     return LHS, RHS, num_q.x, eta, etaprime

        


# =========================================
# Demo tests
# =========================================

if __name__ == '__main__':
    #---momenta for consistency checks
    P1 = 0.1001+0.032j
    P2 = 0.02+0.76j
    P3 = 0.04+0.0084j
    P4 = 0.00776+0.0023j
    P5 = 0.06 + 0.033j
    Ps = 0.125+0.097j
    Pt = 0.00942+0.012j
    theo = Theory(m=3, n=1, Lambda = 2.3)
    Po = 0.0235 + 0.0134j
    Pq = 0.00963  + 0.023j
    Pu = 0.0834 + 0.023j
    Psigma1 = 0.002 + 0.023j
    Psigma2 = 0.0032 + 0.04j
    Psigma3 = 0.012 + 0.0378j
    Psigma4 = 0.0076 + 0.00982j
    

    # #---Demo1: Check the mixing symmetries of the different kernels
    # FplusLHS, FplusmixedLHS = mixingsymmetrieskernelsspacelikef(theo,P1,P2,P3,P4,Ps,Pt,1)
    # FminusLHS, FminusmixedLHS = mixingsymmetrieskernelsspacelikef(theo,P1,P2,P3,P4,Ps,Pt,-1)
    # print("=== Demo: mixing property of the kernels ===")
    # print(f"FplusLHS = {FplusLHS}")
    # print(f"FplusmixedLHS = {FplusmixedLHS}")
    # print(f"FminusLHS = {FminusLHS}")
    # print(f"FminusmixedLHS = {FminusmixedLHS}")
    
    # print()

    # #---Demo2: Check Ioannis/Sylvain relation for the spacelike kernels
    # Fplusatonefourth=kernel_rational(theo, "spacelike", "fusion", [0.25,0.25,0.25,0.25], [Ps,Pt] , 1).getvalue()
    # Fminusatonefourth=kernel_rational(theo, "spacelike", "fusion", [0.25,0.25,0.25,0.25], [Ps,Pt] , -1).getvalue()
    # Fplusfromrelation=RibaultTsiaresspacelikef(theo,Ps,Pt,1)
    # Fminusfromrelation=RibaultTsiaresspacelikef(theo,Ps,Pt,-1)
    # print("=== Demo: Ioannis/Sylvain relation for the spacelike kernels at c=25 ===")
    # print(f"Fplusatonefourth = {Fplusatonefourth}")
    # print(f"Fplusfromrelation = {Fplusfromrelation}")
    # print(f"Fminusatonefourth = {Fminusatonefourth}")
    # print(f"Fminusfromrelation = {Fminusfromrelation}")
    # print()

    # # ---Demo3: Check pentagon for the TV kernel
    # print("=== Demo: Testing the pentagon for the TV kernel ===")
    # LHSTV, RHSTV = SpacelikePentagon(theo, Po, Pq, P1, P2, P3, P4, P5, Pt, Pu, 0)
    # print(f"LHSPentagonTV = {LHSTV}")
    # print(f"RHSPentagonTV = {RHSTV}")

    

    # # #--- Demo 4: Check the Identity limit of the Modular kernel
    # LHS, RHS = Identitylimit(theo, Ps, Pt)
    # print("=== Demo: Identity limit check ===")
    # print(f"Kernel at identity = {LHS}")
    # print(f"Value of the cos   = {RHS}")
    # print()

    # # #--- Demo 5: Check reflection properties under reflection of the external momenta
    # print("=== Demo: Reflection property of Modular kernels ===")
    # print(f"Mplus = {kernel_rational(theo,"spacelike", "modular", [P1], [Ps,Pt] , 1).getvalue()}")
    # print(f"Mminus = {kernel_rational(theo,"spacelike", "modular", [P1], [Ps,Pt] , -1).getvalue()}")
    # print(f"Mplusreflectedps = {kernel_rational(theo,"spacelike", "modular", [P1], [-Ps,Pt] , 1).getvalue()}")
    # print(f"Mminusreflectedps = {kernel_rational(theo,"spacelike", "modular", [P1], [-Ps,Pt] , -1).getvalue()}")
    # print(f"Mplusreflectedpt = {kernel_rational(theo,"spacelike", "modular", [P1], [Ps,-Pt] , 1).getvalue()}")
    # print(f"Mminusreflectedpt = {kernel_rational(theo,"spacelike", "modular", [P1], [Ps,-Pt] , -1).getvalue()}")
    # print(f"MTV = {kernel_rational(theo,"spacelike", "modular", [P1], [Ps,Pt] , 0).getvalue()}")
    # print(f"MTVreflectedps = {kernel_rational(theo,"spacelike", "modular", [P1], [-Ps,Pt] , 0).getvalue()}")
    # print(f"MTVreflectedpt = {kernel_rational(theo,"spacelike", "modular", [P1], [Ps,-Pt] , 0).getvalue()}")
    # print()


    # # # #--- Demo 6: Mixing symmetry of the modular kernels
    # print("=== Demo: Mixing symmetry of the Modular kernels ===")
    # LHSmixingsymmetries, RHSmixingsymmetries = mixingsymmetriesm(theo, P1, P2, P3, -1)
    # print(f"LHS = {LHSmixingsymmetries}")
    # print(f"RHS = {RHSmixingsymmetries}")
    # print()

    # # # #--- Demo 7: Non rational Verlinde formula
    # print("=== Demo: Non rational Verlinde formula ===")
    # LHS, RHS = NonrationalVerlinde(theo,P1,P2,P3,P4,0)
    # print(f"LHSVerlindeformula = {LHS}")
    # print(f"RHSVerlindeformula = {RHS}")
    # print()

    # # # # #--- Demo 8: Check the torus 2 point relation
    # print("=== Demo: Check the torus 2 point relation ===")
    # LHS, RHS = Torus2pointrelation(theo, P1, P2, P3, P4, P5, Po, 0)
    # print(f"LHSTorus2pt = {LHS}")
    # print(f"RHSTorus2pt = {RHS}")
    # print()


    # #Checking the important identities involving the spacelike and the timelike kernels
    # print("=== TIMELIKE LIOUVILLE !!! ===")
    # print()

    
    # #--- Demo9: Check the reflection property of the timelike fusion kernels.
    # FplusTimelike=kernel_rational(theo,"timelike", "fusion", [P1,P2,P3,P4], [Ps,Pt], 1).getvalue()
    # FminusTimelike=kernel_rational(theo, "timelike", "fusion", [P1,P2,P3,P4], [Ps,Pt], -1).getvalue()
    # FplusTimelikereflected=kernel_rational(theo, "timelike", "fusion", [P1,P2,P3,P4], [-Ps,Pt], 1).getvalue()
    # FminusTimelikereflected=kernel_rational(theo, "timelike", "fusion", [P1,P2,P3,P4], [-Ps,Pt], -1).getvalue()
    # print("=== Demo: Reflection property of the plus/minus timelike fusion kernels ===")
    # print(f"FplusTimelike = {FplusTimelike}")
    # print(f"FminusTimelike   = {FminusTimelike}")
    # print(f"FplusTimelikereflected = {FplusTimelikereflected}")
    # print(f"FminusTimelikereflected   = {FminusTimelikereflected}")
    # print()

    # #---Demo10: Check Ioannis/Sylvain relation for the timelike kernels
    # Fplusatonefourthj=kernel_rational(theo, "timelike", "fusion", [0.25j,0.25j,0.25j,0.25j], [Ps,Pt], 1).getvalue()
    # Fminusatonefourthj=kernel_rational(theo, "timelike", "fusion", [0.25j,0.25j,0.25j,0.25j], [Ps,Pt], -1).getvalue()
    # Fplusfromrelation=RibaultTsiarestimelikef(theo,Ps,Pt,1)
    # Fminusfromrelation=RibaultTsiarestimelikef(theo,Ps,Pt,-1)
    # print("=== Demo: Ioannis/Sylvain relation for the timelike kernels at c=1 ===")
    # print(f"Fplusatonefourthj = {Fplusatonefourthj}")
    # print(f"Fplusfromrelation = {Fplusfromrelation}")
    # print(f"Fminusatonefourthj = {Fminusatonefourthj}")
    # print(f"Fminusfromrelation = {Fminusfromrelation}")
    # print()

    # # # #--- Demo 11: Mixing symmetry of the timelike fusion kernels
    # print("=== Demo: Mixing symmetry of the timelike fusion kernels ===")
    # LHSmixingsymmetriesFplus, RHSmixingsymmetriesFplus = mixingsymmetriestimelikekernelsf(theo, P1, P2, P3, P4, Ps, Pt, 1)
    # print(f"LHS = {LHSmixingsymmetriesFplus}")
    # print(f"RHS = {RHSmixingsymmetriesFplus}")
    # print()

    # # ---Demo12: Check pentagon for the Timelike kernel
    # print("=== Demo: Testing the pentagon for timelike kernel===")
    # LHS, RHS = TimelikePentagon(theo, parPenta, 0)
    # print(f"LHSTimelikePentagonplus = {LHS}")
    # print(f"RHSTimelikePentagonplus = {RHS}")
    # print()

    # # ---Demo13: Check pentagon for the Timelike kernel
    # print("=== Demo: Testing the pentagon for timelike kernel===")
    # LHS, RHS = TimelikePentagon(theo, parPenta, 0)
    # print(f"LHSTimelikePentagonplus = {LHS}")
    # print(f"RHSTimelikePentagonplus = {RHS}")
    # print()


    # #--- Demo 14: Check crossing blocks
    print("=== Demo: Check crossing blocks spacelike ===")
    LHS, RHS, eta, one_minus_eta = four_point_block_crossing(theo, "spacelike", 0, P1, P2, P3, P4, Ps)
    print(f"Block_s = {LHS}")
    print(f"RHS = {RHS}")
    print(f"cross-ratio_s = {eta}")
    print(f"cross-ratio_t = {one_minus_eta}")
    print()

    # # # #--- Demo 15: Check crossing blocks boundary 4 point
    # print("=== Demo: Check crossing blocks boundary ===")
    # LHS, RHS, cross_ratio = four_point_block_crossing_bdy(theo, P1, P2, P3, P4, Psigma1, Psigma2, Psigma3, Psigma4)
    # print(f"LHS = {LHS}")
    # print(f"RHS = {RHS}")
    # print(f"cross_ratio = {cross_ratio}")
    # print()

    # # # #--- Demo 16: Check crossing blocks boundary 1 bulk 2 bdy
    # print("=== Demo: Check crossing blocks boundary 2 ===")
    # LHS, RHS, cross_ratio, eta, etaprime = three_point_block_crossing_bdy(theo, P1, P2, P3, Psigma1, Psigma2, 1.05+0.000001j)
    # print(f"LHS = {LHS}")
    # print(f"RHS = {RHS}")
    # print(f"cross_ratio = {cross_ratio}")
    # print(f"cross_ratio_eta = {eta}")
    # print(f"cross_ratio_etaprime = {etaprime}")
    # print()


    # # #--- Demo 16: Check crossing blocks
    # print("=== Demo: Plot relative error ===")
    # Relist, Imlist, xlist = four_point_block_crossing_bdy_generate_data(theo, P1, P2, P3, P4, Psigma1, Psigma2, Psigma3, Psigma4, [k*0.005 for k in range (1,21)], N=10, eps_b=1e-5, save_path="crossing_bdy_data.txt")
    # utils_plot.plot_realandimag(theo, Relist, Imlist, xlist, "Relative error")

    # #--- Demo 14: Check crossing blocks
    print("=== Demo: Check crossing blocks timelike ===")
    LHS, RHS, eta, one_minus_eta = four_point_block_crossing(theo, "timelike", 0, P1, P2, P3, P4, Ps)
    print(f"Block_s = {LHS}")
    print(f"RHS = {RHS}")
    print(f"cross-ratio_s = {eta}")
    print(f"cross-ratio_t = {one_minus_eta}")
    print()
    
    




    



    
    




    


    
    

    

    
    
