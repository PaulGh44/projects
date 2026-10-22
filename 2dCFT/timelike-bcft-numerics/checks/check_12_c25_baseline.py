"""
CHECK 12 - the c >= 25 baseline: the source theory whose kernel identities are rotated (Sec. S2).
==================================================================================================

The c >= 25 boundary Liouville data in the bootstrap normalization (Table I, middle column; tlbcft.theory.Spacelike),
spectrum P in iR, satisfy the sewing constraints as kernel identities.  Verified here (b^2 = 1/2, c = 28, unless stated):
 1. the balancing identities, Eq. (S7):
      F_{P1,Pt}[P2 P2; P3 P3] rho0(P1) C0(P1,P2,P3)^2 = F_{Pt,P1}[P2 P3; P2 P3] rho0(Pt) C0(Pt,P2,P2) C0(Pt,P3,P3),
      S_{P1,P'}[P0] rho0(P1) C0(P1,P1,P0) = S_{P',P1}[P0] rho0(P') C0(P',P',P0)          (paper: 1e-13);
 2. the non-rational Verlinde formula, Eq. (S9) (the cylinder):
      F_{P1,P3}[P0 P3; P1 P2] = int_{iR} dP/i S_{P1,P}[P0] S_{P2,P}[1] S_{P,P3}[P0] / S_{1,P}[1],  S_{1,P}[1] = rho0(P)/2
      with S_{P2,P}[1] = sqrt2 cos(4 pi P2 P)                                              (paper: 2e-12);
 3. the disc two-point identity, Eq. (S10):
      int_{iR} dP/i rho0 C0(P,P1,P2) R0(P|s) F_{P,Pb}[P1 P1; P2 P2]/2 = R0(P1,Pb|s) R0(P2,Pb|s)/(2 S0(Pb;s,s))  (paper: 1e-12);
 4. the pentagon identity (boundary crossing), Eq. (S8), at b = 1                          (paper: 1e-10);
 5. at c = 25 with all externals of weight 15/16 (paper Eq. (S4)) crossing is the exact identity
      J(T) = T^{-3/2} J(1/T),  J(T) = int_R dp p sinh^2(pi p/2)/sinh(2 pi p) e^{-pi T p^2};
 6. S0 > 0 on the spectrum, its vacuum behaviour S0(-Q/2+e; s,s) = pi rho0(s) e/rho0(-Q/2+e) + O(e^2), and cyclicity
    of the vertex (C0) = 2 S0 F;
 7. the ZZ brane: R0(P|P_(1,1)) - R0(P|P_(1,-1)) has the self-annulus chi^deg_(1,1) (the identity character, multiplicity one);
 8. the disc two-point function and the cylinder as AMPLITUDES with c = 28 blocks (paper: 1e-10);
 [--full]
 9. the c >= 25 identities behind the discs with three insertions: the torus two-point Moore-Seiberg relation (S28)
    (paper: 1e-7 - 4e-7) and the new identity (S31) for the disc V V Psi (paper: 1.4e-10), on tilted contours.

Usage:  python checks/check_12_c25_baseline.py          (items 1-8, ~10 s)
        python checks/check_12_c25_baseline.py --full   (also item 9, ~1.5 min)
"""
import time

import numpy as np
from scipy import integrate

from _common import Report, options
from tlbcft.theory import Spacelike
from tlbcft.blocks import SphereBlock, TorusBlock
from tlbcft.numerics import line, simpson, midpoint


def main():
    A = options(__doc__)
    R = Report(__file__, "the c >= 25 baseline (bootstrap normalization)", "Sec. S2, Eqs. (S7)-(S10); Sec. S6, Eqs. (S28), (S31)")
    S = Spacelike((1, 2))
    Q = S.Q
    rng = np.random.default_rng(5)

    R.section("1. balancing identities (S7)")
    for trial in range(2):
        P1, Pt, P2, P3 = 1j*(0.1 + 0.4*rng.random(4)) + 0.05*rng.random(4)
        lhs = S.Fm(P1, Pt, P2, P2, P3, P3)*S.rho0(P1)*S.C0(P1, P2, P3)**2
        rhs = S.Fm(Pt, P1, P2, P3, P2, P3)*S.rho0(Pt)*S.C0(Pt, P2, P2)*S.C0(Pt, P3, P3)
        R.agree(f"fusion balancing #{trial}", lhs, rhs, 1e-11)
        P1, Pp, P0 = 1j*(0.1 + 0.4*rng.random(3)) + 0.05*rng.random(3)
        R.agree(f"modular balancing #{trial}", S.S(P1, Pp, P0)*S.rho0(P1)*S.C0(P1, P1, P0),
                S.S(Pp, P1, P0)*S.rho0(Pp)*S.C0(Pp, Pp, P0), 1e-11)

    R.section("2. non-rational Verlinde formula (S9)")
    y = np.linspace(-6, 6, 2400)
    P = 1j*y
    for (P1, P2, P3, P0) in [(0.23j, 0.31j, 0.17j, 0.21j), (0.11j, 0.42j, 0.29j, 0.13j)]:
        f = S.S(P1, P, P0)*np.sqrt(2)*np.cos(4*np.pi*P2*P)*S.S(P, P3, P0)/(S.rho0(P)/2)
        R.agree(f"(P1,P2,P3,P0) = ({P1},{P2},{P3},{P0})", S.Fm(P1, P3, P0, P3, P1, P2), simpson(f, y), 1e-10)

    R.section("3. disc two-point identity (S10)")
    P1, P2, s = 0.13j, 0.21j, 0.19j
    for Pb in (0.27j, 0.53j):
        f = S.rho0(P)*S.C0(P, P1, P2)*S.R1(P, s)*S.Fm(P, Pb, P1, P1, P2, P2)/2
        R.agree(f"Pb = {Pb}", simpson(f, y), S.R2(P1, Pb, s)*S.R2(P2, Pb, s)/(2*S.S0(Pb, s, s)), 1e-10,
                names=("bulk", "boundary"))

    R.section("4. pentagon identity (S8), b = 1")
    S1 = Spacelike((1, 1))
    Pv = 1j*np.array([0.13, 0.21, 0.17, 0.29])
    sv = 1j*np.array([0.19, 0.21, 0.24, 0.28])
    for Pt in (0.37j, 0.61j):
        f = (S1.Fm(sv[1], P, Pv[1], Pv[0], sv[2], sv[0])*S1.Fm(sv[2], Pv[3], Pv[2], P, sv[3], sv[0])
             * S1.F(P, Pt, Pv[0], Pv[1], Pv[2], Pv[3]))
        rhs = S1.Fm(sv[1], Pv[3], Pt, Pv[0], sv[3], sv[0])*S1.Fm(sv[2], Pt, Pv[2], Pv[1], sv[3], sv[1])
        R.agree(f"Pt = {Pt}", simpson(f, y), rhs, 1e-9)

    R.section("5. c = 25, externals of weight 15/16: crossing is J(T) = T^{-3/2} J(1/T)")
    J = lambda T: 2*integrate.quad(lambda p: p*np.sinh(np.pi*p/2)**2/np.sinh(2*np.pi*p)*np.exp(-np.pi*T*p*p), 1e-12, 12,
                                   limit=400, epsabs=1e-15)[0]                     # even integrand
    for T in (0.5, 0.8, 1.7):
        R.agree(f"T = {T}", J(T), T**-1.5*J(1/T), 1e-11)

    R.section("6. two-point function and vertex")
    yy = np.linspace(0.05, 4, 40)
    s0 = S.S0(1j*yy, 0.19j, 0.24j)
    R.true("S0(P; s1, s2) > 0 on the spectrum P in iR (40 points)", np.all(s0.real > 0) and np.max(np.abs(s0.imag/s0.real)) < 1e-12,
           f"min {s0.real.min():.3e}")
    ratio = lambda e: S.S0(-Q/2 + e, 0.19j, 0.19j)/(np.pi*S.rho0(0.19j)*e/S.rho0(-Q/2 + e))
    R.agree("vacuum behaviour S0(-Q/2+e; s,s) / [pi rho0(s) e/rho0(-Q/2+e)] -> 1 (Richardson in e)",
            (4*(2*ratio(1e-4) - ratio(2e-4)) - (2*ratio(2e-4) - ratio(4e-4)))/3, 1.0, 1e-8)
    for trial in range(2):
        Q1, Q2, Q3 = 1j*(0.1 + 0.3*rng.random(3)) + 0.05*rng.random(3)
        s1, s2, s3 = 1j*(0.1 + 0.3*rng.random(3)) + 0.05*rng.random(3)
        a = S.C3(Q3, Q2, Q1, s3, s2, s1)
        R.agree(f"cyclicity of (C0)^{{s3 s2 s1}}_{{P3 P2 P1}} #{trial}", a, S.C3(Q2, Q1, Q3, s2, s1, s3), 1e-10)

    R.section("7. the ZZ brane (1,1): self-annulus = the identity character")
    b = S.b
    for T in (0.7, 1.3):
        yz = np.linspace(-8, 8, 160000)              # even number of points: P = 0 (where rho0 = 0) is not on the grid
        Pz = 1j*yz
        # Psi_ZZ = R0(P|P_(1,1)) - R0(P|P_(1,-1)), P_(r,s) = (r b + s/b)/2; closed channel int dP/i (rho0/2) Psi^2 chi_P(qt)
        psi = S.R1(Pz, 0.5*(b + 1/b)) - S.R1(Pz, 0.5*(b - 1/b))
        Zc = simpson((S.rho0(Pz)/2*psi**2*np.exp(2*np.pi*Pz**2/T)).real, yz)/np.sqrt(T)      # eta(q) Z
        q = np.exp(-2*np.pi*T)
        hdeg = lambda r, s: (Q*Q/4 - (0.5*(r*b + s/b))**2) - (S.c - 1)/24     # h - (c-1)/24 = -P^2 + Q^2/4 - (c-1)/24
        Zo = q**hdeg(1, 1) - q**hdeg(1, -1)
        R.agree(f"T = {T}: closed channel = chi^deg_(1,1) (eta(q) Z)", Zc, Zo, 1e-10, names=("closed", "open"))

    R.section("8. amplitudes with c = 28 blocks: disc two-point function and cylinder")
    t0 = time.time()
    c = S.c
    yh, Ph = line(0.0, 4.5, 1200)
    h1, h2 = S.h(P1), S.h(P2)
    bs, bt = SphereBlock(c, h2, h1, h1, h2, Nmax=12), SphereBlock(c, h1, h1, h2, h2, Nmax=12)
    Hs, Ht = bs.H_series(S.h(Ph)), bt.H_series(S.h(Ph))
    db = S.rho0(Ph)/2*S.C0(Ph, P1, P2)*S.R1(Ph, s)
    dt = S.R2(P1, Ph, s)*S.R2(P2, Ph, s)/(2*S.S0(Ph, s, s))
    for z in (0.35, 0.5):
        R.agree(f"disc two-point, z = {z}", simpson(db*bs.value(z, S.h(Ph), Hs), yh), simpson(dt*bt.value(1 - z, S.h(Ph), Ht), yh),
                1e-10, names=("bulk", "boundary"))
    sa, sb, P0 = 0.19j, 0.24j, 0.17j
    TB = TorusBlock(c, P0, Nmax=8)
    Tc = TB.coeffs(Ph)
    do = S.Fm(sa, Ph, P0, Ph, sa, sb)
    dc = S.rho0(Ph)/2*S.R2(Ph, P0, sa)*S.R1(Ph, sb)
    for tau in (1j, 1.3j):
        Zo = (-1j*tau)**S.h(P0)*simpson(do*TB.value(tau, Ph, Tc), yh)
        Zc = simpson(dc*TB.value(-1/tau, Ph, Tc), yh)
        R.agree(f"cylinder, tau = {tau}", Zo, Zc, 1e-10, names=("open", "closed"))
    R.note(f"[{time.time() - t0:.0f} s]")

    if A.full:
        R.section("9a. torus two-point Moore-Seiberg relation (S28), tilted contours P3, P4 in i t e^{i theta} R")
        t0 = time.time()
        P1_, Pp_, P2_, P0_, P0p_ = 0.19j, 0.23j, 0.13j, 0.21j, 0.17j
        for P5 in (0.27j, 0.41j):
            lhs = S.S(Pp_, P2_, P5)*S.Fm(P1_, P5, P0p_, P0_, Pp_, Pp_)
            t, _, hst = midpoint(0.0, 7.0, 400)      # the tilt gives Gaussian decay e^{-pi t^2 sin(2 theta)/2}; |t| <= 7 needed
            e3 = e4 = np.exp(0.15j)
            P3, P4 = 1j*t.real*e3, 1j*t.real*e4
            A3 = S.S(P1_, P2_, P3)*S.Fm(Pp_, P3, P0p_, P0_, P1_, P1_)*np.exp(1j*np.pi*S.h(P3)/2)
            A4 = S.Fm(P4, P5, P0p_, P0_, P2_, P2_)*np.exp(2j*np.pi*S.h(P4))
            M = S.Fm(P3[:, None], P4[None, :], P0p_, P2_, P0_, P2_)
            I = e3*e4*hst*hst*np.sum(A3[:, None]*M*A4[None, :])
            I *= np.exp(1j*np.pi*(-2*S.h(P2_) - S.h(P0_) - S.h(P0p_) + S.h(P5)/2))
            R.agree(f"P5 = {P5}", I, lhs, 1e-6, names=("integral", "lhs"))   # ~1e-8 with this grid
        R.note(f"[{time.time() - t0:.0f} s]")
        R.section("9b. the new identity (S31) for the disc V V Psi at c = 28: fusion, half braid, two fusions")
        t0 = time.time()
        P1_, P2_, Pb_, s_ = 0.13j, 0.21j, 0.17j, 0.19j
        hP_ = 2*5.0/200
        Pg = 1j*(-5.0 + (np.arange(200) + 0.5)*hP_)
        hc = 2*6.0/240
        cg = 1j*(-6.0 + (np.arange(240) + 0.5)*hc)*np.exp(0.15j)
        Bv = S.C0(P1_, P2_, Pg)*S.S(s_, Pg, Pb_)                             # (rho0/2) C0 R0(P,Pb|s) = C0 S_{s,P}[Pb]
        Fpc = S.Fm(Pg[:, None], cg[None, :], P1_, Pb_, P2_, Pg[:, None])
        ph = np.exp(1j*np.pi*(S.h(cg) - S.h(P1_) - S.h(Pb_)))
        for Ppp in (0.31j,):
            Fppp = S.Fm(Pg[:, None], Ppp, cg[None, :], P1_, P2_, P2_)
            Mx = Bv[:, None]*Fpc*Fppp*ph[None, :]
            for Pp in (0.27j, 0.40j):
                G = np.nan_to_num(Mx*S.Fm(cg, Pp, P1_, P1_, Pb_, Ppp)[None, :])
                I = hP_*hc*np.exp(0.15j)*np.sum(G)
                rhs = (S.R2(P1_, Pp, s_)*S.R2(P2_, Ppp, s_)*S.C3(Pb_, Ppp, Pp, s_, s_, s_)
                       / (2*S.S0(Pp, s_, s_)*2*S.S0(Ppp, s_, s_)))
                phase = np.exp(-0.5j*np.pi*(S.h(Pp) - S.h(Ppp) + S.h(Pb_)))
                R.agree(f"(P', P'') = ({Pp}, {Ppp})", I, phase*rhs, 1e-9, names=("lhs", "e^{-i pi(..)/2} rhs"))
        R.note(f"[{time.time() - t0:.0f} s]")
    R.done()


if __name__ == "__main__":
    main()
