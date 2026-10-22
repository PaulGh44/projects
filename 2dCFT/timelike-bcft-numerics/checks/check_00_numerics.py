"""
CHECK 00 - validation of the numerical building blocks (special functions, kernels, blocks).
=============================================================================================

Before any physics is tested, every ingredient is compared with an independent computation:

 1. Barnes G at known values; Gamma_b by the rational reduction vs the integral representation (two independent codes);
    the normalisation Gamma_b(Q/2) = 1, the shift relation, and Gamma_b(2P)Gamma_b(-2P)/Gamma_b(Q +- 2P) = sqrt2/rho0(P)
    (paper Sec. S1, "Barnes functions and bulk data").
 2. The closed-form fusion kernel F at rational b^2 vs the Teschner-Vartanov Barnes integral at b^2 = 1, 1/2, 2/3, 2/5, 1/3
    (paper Sec. S1: "agrees ... to 1e-11 - 1e-13"; at the points used here the agreement is 1e-12 - 4e-10, the
    closed form losing a few digits as mn grows); its symmetries and evenness; the c = 25 elementary kernel.
 3. The c <= 1 example of paper Eq. (3) / (S4): at bhat = 1, all externals i/4, RF = 2^{4Pt^2-4Ps^2} sin(2 pi Ps Pt),
    and the physical kernel -i(T1 + T2) = 2^{4Pt^2-4Ps^2} cos(2 pi Ps Pt).
 4. The modular kernel from F at b' = sqrt2 b (paper Eq. (S5)) vs Teschner's integral for S.
 5. The general-b kernel (Barnes integrals with crossed-pole residues, paper Sec. S8) vs the closed form at
    bhat^2 = 5/7 for momenta on the physical contour ("to 1e-10 - 1e-12").
 6. Sphere blocks: level-1 and level-2 coefficients; the exact c = 1, h = 1/16 block, also continued to
    tau = +-1/2 + i/2 where the disc V Psi Psi is evaluated; there the same block computed with explicit inverses of
    the Gram matrices (as in the original vblock.py) loses 4-6 digits, which is why tlbcft.blocks solves them;
    torus blocks vs sphere blocks through the Poghossian relation (paper Sec. S8, "Methods").

Usage:  python checks/check_00_numerics.py            (~20 s)
"""
import numpy as np
from scipy.special import loggamma

from _common import Report, options
from tlbcft.special import G, RationalB, GeneralB, lgamma_b_int
from tlbcft.kernels import FusionKernel, ModularKernel, GeneralFusionKernel, tv_kernel, mod_kernel_tv
from tlbcft.blocks import SphereBlock, block_c1_sixteenth, poghossian_values, theta_logs


def main():
    options(__doc__)
    R = Report(__file__, "numerical building blocks", "Sec. S1 (conventions, kernels), Sec. S8 (methods)")

    R.section("1. Barnes G and Gamma_b")
    A = 1.2824271291006226
    R.agree("G(6) = 288", G(6.0), 288.0, 3e-13)
    R.agree("G(1/2)", G(0.5), 2**(1/24)*np.exp(1/8)*np.pi**(-1/4)*A**(-1.5), 3e-13)
    z = -2.3 + 1.1j
    R.agree("G(z+1) = Gamma(z) G(z) at z = -2.3+1.1i", G(z + 1), np.exp(loggamma(z))*G(z), 1e-12)
    for (m, n) in [(1, 1), (1, 2), (2, 3), (5, 7)]:
        rb = RationalB(m, n)
        gb = GeneralB(rb.b)
        for x in [1.3 + 0.4j, -0.4 + 1.7j]:
            a = np.exp(rb.lgamma_b(x))
            R.agree(f"Gamma_b({x}) at b^2={m}/{n}: reduction vs quadrature", a, np.exp(lgamma_b_int(x, rb.b)), 3e-10)
            R.agree(f"Gamma_b({x}) at b^2={m}/{n}: reduction vs Gauss-Legendre", a, np.exp(gb.lgamma_b(x)), 1e-11)
    rb = RationalB(1, 2)
    b, Q = rb.b, rb.Q
    R.agree("Gamma_b(Q/2) = 1", np.exp(rb.lgamma_b(Q/2)), 1.0, 1e-14)
    x = 0.4 + 0.3j
    R.agree("Gamma_b(x+b)/Gamma_b(x) = sqrt(2pi) b^{bx-1/2}/Gamma(bx)", np.exp(rb.lgamma_b(x + b) - rb.lgamma_b(x)),
            np.sqrt(2*np.pi)*b**(b*x - 0.5)/np.exp(loggamma(b*x)), 3e-13)
    P = 0.23j
    rho0 = -4*np.sqrt(2)*np.sin(2*np.pi*b*P)*np.sin(2*np.pi*P/b)
    R.agree("Gamma_b(2P)Gamma_b(-2P)/Gamma_b(Q+-2P) = sqrt2/rho0(P)",
            np.exp(rb.lgamma_b(2*P) + rb.lgamma_b(-2*P) - rb.lgamma_b(Q + 2*P) - rb.lgamma_b(Q - 2*P)), np.sqrt(2)/rho0, 3e-13)

    R.section("2. closed-form fusion kernel vs the Teschner-Vartanov integral (imaginary momenta)")
    for (m, n) in [(1, 1), (1, 2), (2, 3), (2, 5), (1, 3)]:
        K = FusionKernel(m, n)
        for args in [(0.13j, 0.29j, 0.07j, 0.11j, 0.17j, 0.23j), (0.31j, 0.12j, 0.21j, 0.05j, 0.16j, 0.27j)]:
            lab = ", ".join(f"{a.imag:g}i" for a in args)
            R.agree(f"b^2={m}/{n}, (Ps,Pt,P1..P4)=({lab})", K.F(*args), tv_kernel(m, n, *args), 1e-9,
                    names=("closed", "TV"))
    R.note("(the closed form loses accuracy as mn grows: up to ~4e-10 at b^2 = 2/5; the TV integral itself is")
    R.note(" stable to ~1e-12, as one checks by evaluating it with the independent GeneralB Gamma_b)")
    K = FusionKernel(1, 2)
    Ps, Pt, P1, P2, P3, P4 = 0.13j, 0.05 + 0.29j, 0.07j, 0.11j, -0.02 + 0.17j, 0.23j
    a = K.F(Ps, Pt, P1, P2, P3, P4)
    R.agree("symmetry F[P2 P3;P1 P4] = F[P1 P4;P2 P3]", a, K.F(Ps, Pt, P2, P1, P4, P3), 1e-12)
    R.agree("symmetry F[P2 P3;P1 P4] = F[P3 P2;P4 P1]", a, K.F(Ps, Pt, P4, P3, P2, P1), 1e-12)
    R.agree("F even in Ps", a, K.F(-Ps, Pt, P1, P2, P3, P4), 1e-10)
    R.agree("F even in Pt", a, K.F(Ps, -Pt, P1, P2, P3, P4), 1e-10)
    R.agree("F even in P3", a, K.F(Ps, Pt, P1, P2, -P3, P4), 1e-10)
    K11 = FusionKernel(1, 1)
    for Ps, Pt in [(0.12j, 0.21j), (0.15 + 0.19j, 0.3j)]:
        R.agree(f"c=25, externals -1/4: F = -2^(4Pt^2-4Ps^2)(Pt/Ps)sin(2pi Ps Pt), Ps={Ps}, Pt={Pt}",
                K11.F(Ps, Pt, -0.25, -0.25, -0.25, -0.25), -2**(-4*Ps**2 + 4*Pt**2)*(Pt/Ps)*np.sin(2*np.pi*Ps*Pt), 1e-11)

    R.section("3. paper Eq. (3): bhat = 1, all externals i/4")
    for Ps, Pt in [(0.12j, 0.21j), (0.15 + 0.19j, 0.3j), (0.31, -0.44 + 0.2j)]:
        pref = 2**(-4*Ps**2 + 4*Pt**2)
        R.agree(f"RF = 2^(4Pt^2-4Ps^2) sin(2pi Ps Pt) at Ps={Ps}, Pt={Pt}", K11.RF(Ps, Pt, 0.25j, 0.25j, 0.25j, 0.25j),
                pref*np.sin(2*np.pi*Ps*Pt), 1e-11)
        T1, T2 = K11.root_terms(1j*Pt, 1j*Ps, 0.25j*1j, 0.25j*1j, 0.25j*1j, 0.25j*1j)
        phys = -1j*(Pt/Ps)*(T1 + T2)
        R.agree(f"physical kernel -i(T1+T2) = +- 2^(4Pt^2-4Ps^2) cos(2pi Ps Pt) (sign = sheet)", abs(phys),
                abs(pref*np.cos(2*np.pi*Ps*Pt)), 1e-11)

    R.section("4. modular kernel from F (Roussillon) vs Teschner's integral")
    for (m, n) in [(1, 2), (1, 1)]:
        MK = ModularKernel(m, n)
        for (Ps, Pt, P0) in [(0.23j, 0.31j, 0.17j), (0.11j, 0.42j, 0.29j), (0.35j, 0.19j, 0.05j)]:
            R.agree(f"b^2={m}/{n}, (Ps,Pt,P0)=({Ps},{Pt},{P0})", MK.S(Ps, Pt, P0), mod_kernel_tv(m, n, Ps, Pt, P0), 1e-11,
                    names=("from F", "Teschner"))

    R.section("5. general-b kernel (Barnes integrals + crossed residues) vs closed form, bhat^2 = 5/7, Route-2 regime")
    K57 = FusionKernel(5, 7)
    G57 = GeneralFusionKernel(np.sqrt(5/7))
    u = np.array([0.13, 0.21, 0.17, 0.29]); v = np.array([0.19, 0.21, 0.24, 0.28])
    for y in (0.3, 1.7, 3.8):
        Ph = 0.15 + 1j*y
        for name, args in (("internal slot", (1j*Ph, -v[1], -u[0], -u[1], -v[2], -v[0])),
                           ("external slot", (-u[3], -v[2], 1j*Ph, -u[2], -v[3], -v[0]))):
            R.agree(f"Phat = 0.15 + {y}i, {name}", G57.F(*args), K57.F(*args), 1e-9, names=("general b", "closed"))

    R.section("6. conformal blocks")
    c, h = -2.0, 0.37
    h1, h2, h3, h4 = 0.13, 0.21, 0.17, 0.29
    a = SphereBlock(c, h1, h2, h3, h4, Nmax=3).z_coeffs(h)[0]
    R.agree("level 1: (h+h2-h1)(h+h3-h4)/(2h)", a[1], (h + h2 - h1)*(h + h3 - h4)/(2*h), 1e-14)
    Gm = np.array([[4*h*(2*h + 1), 6*h], [6*h, 4*h + c/2]])
    vL = np.array([(h + h3 - h4)*(h + h3 - h4 + 1), h + 2*h3 - h4])
    vR = np.array([(h + h2 - h1)*(h + h2 - h1 + 1), h + 2*h2 - h1])
    R.agree("level 2: Gram-matrix formula", a[2], vL @ np.linalg.solve(Gm, vR), 1e-13)
    sb = SphereBlock(1.0, 1/16, 1/16, 1/16, 1/16, Nmax=12)
    Ps = np.array([0.31j, 0.15 + 0.6j, 0.15 + 1.2j])
    H = sb.H_series(-Ps**2)
    for z in (0.35, 0.5):
        vals = sb.value(z, -Ps**2, H)
        for k, P in enumerate(Ps):
            R.agree(f"c=1, h_i=1/16, level 12, z={z}, P={P}: elliptic vs exact", vals[k], block_c1_sixteenth(z, P), 1e-13)
    # the same exact block continued to tau = +-1/2 + i/2 (z = 2 +- i0), where the disc V Psi Psi is evaluated:
    # V = 2^{-4P^2} e^{-i pi P^2 tau} / ([z(1-z)]^{1/8} theta3(q)), all factors continued holomorphically from tau = i oo
    Ps = np.array([0.5 + 0.5j, 0.5 + 1.0j, 0.5 + 2.0j])
    H = sb.H_series(-Ps**2)
    for tau in (0.5 + 0.5j, -0.5 + 0.5j):
        q, l2, l3, l4 = theta_logs(tau)
        exact = np.exp(-4*Ps**2*np.log(2) - 1j*np.pi*Ps**2*tau - (4*(l2 - l3) + 4*(l4 - l3))/8 - l3)
        vals = sb.at_tau(tau, -Ps**2, H)
        for k, P in enumerate(Ps):
            R.agree(f"c=1, h_i=1/16, level 12, tau={tau}, P={P}: continued block vs exact", vals[k], exact[k],
                    2e-9 if abs(P) < 1.5 else 2e-7)    # truncation at q^13, |q| = 0.21
    # Why the Gram matrices are solved, not inverted: the same exact block with G^{-1} applied as an explicit inverse
    # (the project's original vblock.py).  At |q| = e^{-pi/2} the coefficients a_N 16^N of the z-series are large and
    # cancel in H(q), so the roundoff of the inverse is amplified.
    sbi = SphereBlock(1.0, 1/16, 1/16, 1/16, 1/16, Nmax=12, gram='inverse')
    Ps = np.array([0.15 + 0.6j, 0.15 + 1.2j, 0.5 + 2.0j])
    tau = 0.5 + 0.5j
    q, l2, l3, l4 = theta_logs(tau)
    exact = np.exp(-4*Ps**2*np.log(2) - 1j*np.pi*Ps**2*tau - (4*(l2 - l3) + 4*(l4 - l3))/8 - l3)
    e_sol = np.abs(sb.at_tau(tau, -Ps**2)/exact - 1)
    e_inv = np.abs(sbi.at_tau(tau, -Ps**2)/exact - 1)
    for k, P in enumerate(Ps):
        R.true(f"tau = 1/2 + i/2, P = {P}: solved Gram matrices beat the explicit inverse by > 100",
               e_sol[k] < 1e-2*e_inv[k], f"error solve {e_sol[k]:.1e}, inverse {e_inv[k]:.1e}")
    for (c, P0, P, lab) in [(28.0, 0.21j, 0.3j, "c=28"), (-2.0, 0.17j, 0.15 + 0.4j, "c=-2"),
                            (1 + 6*(0.83 + 0.11j + 1/(0.83 + 0.11j))**2, 0.19 - 0.08j, 0.27 + 0.05j, "complex c")]:
        lhs, rhs = poghossian_values(c, P0, P, 1j)
        R.agree(f"torus block (Verma module) vs sphere block at b'^2 = 2b^2 (Poghossian), {lab}, tau = i", lhs, rhs, 1e-13)
    R.done()


if __name__ == "__main__":
    main()
