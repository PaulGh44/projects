"""
CHECK 01 - the c <= 1 data of Table I: definitions, algebraic identities, parities, special columns.
=====================================================================================================

The data (tlbcft/theory.py, class Timelike, final normalization):
    rhohat(P)  = P^2/(2 rho0(iP)),              Chat(P1,P2,P3) = 1/C0(iP1,iP2,iP3),
    Shat(P;s1,s2) = 2 sqrt2 rho0(is1) rho0(is2) C0(iP,is1,is2)/(s1 s2 P),
    (Chat)^{s3 s2 s1}_{P3 P2 P1} = 2 Shat(P3;s1,s3) RF_{s2,P3}[P2 P1; s3 s1],
    Rhat(Pa,Pb|s) = 2 sqrt2 RS_{s,Pa}[Pb]/Pa,   Rhat(Pa|s) = 4 cos(4 pi s Pa)/Pa.

What is verified (all at bhat^2 = 1/2, chat = -2, except where stated):
 1. Bulk.  Chat = 1/C0(i.) solves the degenerate shift equation (S8) and its dual (bhat -> 1/bhat);
    Chat(0,0,0) = sqrt2 Gamma_b(Q)^9/[Gamma_b(2Q) Gamma_b(Q/2)^8];  the rotated kernel reaches the source vacuum row at
    chat-weight one:  lim_{Pt -> (i/2)(bhat+1/bhat)} RF_{P1,Pt}[P2 P2; P3 P3] = i(bhat+1/bhat)/(4 P1) rho0(iP1) C0(iP1,iP2,iP3)
    (Sec. S3 "Bulk coefficient");  at c >= 25 the vacuum row F_{1,P1}[P2 P3; P2 P3] = rho0(P1) C0(P1,P2,P3)/2 (Sec. S1).
 2. Kernel columns (Sec. S1): S_{Ps,Pt}[1] = sqrt2 cos(4 pi Ps Pt);  S at source weight one = -sqrt2 (Pt/Ps) sin(4 pi Ps Pt);
    RS at chat-weight one = sqrt2 (Pt/Ps) cos(4 pi Ps Pt);  RS at 0hat = sqrt2 sin(4 pi Ps Pt).
    Parities: RF odd in each internal and even in each external momentum; RS odd in its internal momenta.
 3. Tetrahedral symmetry and the exchange identity, Eq. (S6), both the source form and its rotation (complex momenta).
 4. The boundary vertex: the cyclicity step (S10); g = Shat * Phi depends only on the shared label, with
    g = s/(sqrt2 rhohat(s)) for kappa = 1/sqrt2; cyclicity of (Chat); parities (even in operator momenta, odd in labels);
    Shat odd in P and in each label.
 5. The weight-one pole of Shat at equal labels, Eq. (S12): Shat ~ -4 kappa/(pi (bhat+1/bhat) rhohat(s) delta),
    = -98.72744079/delta at s = 0.19i (paper).
 6. Bulk-boundary columns, Eq. (S20): Rhat(Pa, weight one | s) = (Pa/s) Rhat(Pa|s);
    Rhat(Pa, 0hat | s) = 4 sin(4 pi s Pa)/Pa = -(4 pi Pa)^{-1} d/ds Rhat(Pa|s);  parities of Rhat;
    both P Rhat(P|s) and P Rhat(P,0hat|s) satisfy f(P + i nu/2) + f(P - i nu/2) = 2 cosh(2 pi nu s) f(P), nu = bhat^{+-1}.
Limits at singular points of the closed forms are taken by symmetric displacement + Richardson extrapolation.

Usage:  python checks/check_01_table_I.py           (~1 s)
"""
import numpy as np
from scipy.special import loggamma

from _common import Report, options
from tlbcft.theory import Timelike, Spacelike


def richardson(f, d=1e-3):
    """lim_{delta->0} f(delta) for f(delta) = f0 + f1 delta + f2 delta^2 + ...: two-step Richardson extrapolation."""
    a, b, c = f(d), f(d/2), f(d/4)
    r1, r2 = 2*b - a, 2*c - b
    return (4*r2 - r1)/3


def main():
    options(__doc__)
    R = Report(__file__, "Table I: definitions and algebraic identities of the c <= 1 data",
               "Table I; Sec. S1 (columns, Eq. (S6)); Sec. S3 (Eqs. (S8)-(S12), (S18)-(S20))")
    T = Timelike((1, 2))
    S = Spacelike((1, 2))
    bh, Q = T.bh, T.Q
    lg = T.gb.lgamma_b
    rng = np.random.default_rng(7)

    R.section("1. bulk coefficient and vacuum rows")
    P1, P2, P3 = 0.13 + 0.05j, 0.21 - 0.03j, 0.17 + 0.02j
    for nu, lab in ((bh, "bhat"), (1/bh, "1/bhat (dual)")):
        o = 1/nu if nu == bh else bh      # the other parameter
        lhs = (T.Ch(P1 - 0.5j*nu, P2, P3)/T.Ch(P1 + 0.5j*nu, P2, P3)
               * np.exp(lg(o + 2j*P1) + lg(2*nu + o - 2j*P1) - lg(2*nu + o + 2j*P1) - lg(o - 2j*P1)))
        rhs = nu**(-8j*nu*P1)
        for e2 in (1, -1):
            for e3 in (1, -1):
                rhs *= np.exp(loggamma(0.5 + 1j*nu*(P1 + e2*P2 + e3*P3)) - loggamma(0.5 + 1j*nu*(-P1 + e2*P2 + e3*P3)))
        R.agree(f"shift equation (S8), nu = {lab}", lhs, rhs, 1e-12)
    R.agree("Chat(0,0,0) = sqrt2 Gamma_b(Q)^9/(Gamma_b(2Q) Gamma_b(Q/2)^8)", T.Ch(0, 0, 0),
            np.sqrt(2)*np.exp(9*lg(Q) - lg(2*Q) - 8*lg(Q/2)), 1e-12)
    pred = 1j*(bh + 1/bh)/(4*P1)*T.rho0(1j*P1)*T.C0(1j*P1, 1j*P2, 1j*P3)
    lim = richardson(lambda d: T.RFm(P1, T.one + d, P2, P2, P3, P3), 1e-3)
    R.agree("RF_{P1,Pt}[P2 P2;P3 P3] at Pt -> chat-weight one = i(bh+1/bh)/(4P1) rho0 C0", lim, pred, 1e-7)
    Pa, Pb, Pc = 0.02 + 0.13j, 0.21j, -0.01 + 0.17j
    lim = richardson(lambda d: S.Fm(-Q/2 + d, Pa, Pb, Pc, Pb, Pc), 1e-3)
    R.agree("c >= 25 vacuum row F_{1,P1}[P2 P3;P2 P3] = rho0(P1) C0(P1,P2,P3)/2", lim, 0.5*S.rho0(Pa)*S.C0(Pa, Pb, Pc), 1e-7)
    R.agree("rhohat(P) = P^2/(2 rho0(iP))", T.rhoh(0.15 + 0.4j), (0.15 + 0.4j)**2/(2*T.rho0(1j*(0.15 + 0.4j))), 1e-15)

    R.section("2. special columns and parities of the kernels")
    Ps, Pt = 0.23j, 0.31j
    lim = richardson(lambda d: S.S(Ps, Pt, -Q/2 + d), 1e-4)
    R.agree("identity column S_{Ps,Pt}[1] = sqrt2 cos(4 pi Ps Pt)", lim, np.sqrt(2)*np.cos(4*np.pi*Ps*Pt), 1e-7)
    R.agree("S at source weight one = -sqrt2 (Pt/Ps) sin(4 pi Ps Pt)", S.S(Ps, Pt, 0.5*(S.b - 1/S.b)),
            -np.sqrt(2)*Pt/Ps*np.sin(4*np.pi*Ps*Pt), 1e-11)
    x, Y = 0.03 + 0.19j, 0.4 + 0.3j
    R.agree("RS at chat-weight one = sqrt2 (Pt/Ps) cos(4 pi Ps Pt)", T.RS(x, Y, T.one), np.sqrt(2)*Y/x*np.cos(4*np.pi*x*Y), 1e-11)
    R.agree("RS at 0hat = sqrt2 sin(4 pi Ps Pt)", T.RS(x, Y, T.zero), np.sqrt(2)*np.sin(4*np.pi*x*Y), 1e-11)
    args = (0.15 + 0.3j, 0.1 + 0.42j, 0.13j, 0.05 + 0.21j, 0.17j, 0.29j)
    a = T.RF(*args)
    R.agree("RF odd in Ps", -a, T.RF(-args[0], *args[1:]), 1e-10)
    R.agree("RF odd in Pt", -a, T.RF(args[0], -args[1], *args[2:]), 1e-10)
    R.agree("RF even in P2", a, T.RF(*args[:3], -args[3], *args[4:]), 1e-10)
    R.agree("RS odd in Ps", -T.RS(x, Y, 0.17j), T.RS(-x, Y, 0.17j), 1e-10)
    R.agree("RS odd in Pt", -T.RS(x, Y, 0.17j), T.RS(x, -Y, 0.17j), 1e-10)

    R.section("3. tetrahedral symmetry and the exchange identity, Eq. (S6)")

    def Phi(A, B, C):
        return (B/A)*T.rhoh(A)*T.Ch(A, B, C)
    for trial in range(3):
        Ps_, Pt_, Q1, Q2, Q3, Q4 = 1j*(0.1 + 0.4*rng.random(6)) + 0.1*rng.random(6)
        lhs = S.F(Ps_, Pt_, Q1, Q2, Q3, Q4)
        rhs = S.rho0(Pt_)*S.C0(Pt_, Q3, Q2)/(S.rho0(Q1)*S.C0(Q1, Ps_, Q2))*S.F(Q3, Q1, Pt_, Q2, Ps_, Q4)
        R.agree(f"source form, random complex momenta #{trial}", lhs, rhs, 3e-11)
    for trial in range(3):
        Ps_, Pt_, Q1, Q2, Q3, Q4 = (0.15 + 0.1*rng.random(6)) + 1j*(0.1 + 0.4*rng.random(6))
        lhs = T.RF(Ps_, Pt_, Q1, Q2, Q3, Q4)
        rhs = Phi(Q3, Pt_, Q2)/Phi(Ps_, Q1, Q2)*T.RF(Q3, Q1, Pt_, Q2, Ps_, Q4)
        R.agree(f"rotated form RF = Phi/Phi RF, random complex momenta #{trial}", lhs, rhs, 1e-11)

    R.section("4. the boundary vertex: cyclicity and parities")
    for trial in range(3):
        Q1, Q2, Q3 = 1j*(0.1 + 0.3*rng.random(3)) + 0.05*rng.random(3)
        s1, s2, s3 = 1j*(0.1 + 0.3*rng.random(3)) + 0.05*rng.random(3)
        lhs = T.RFm(s2, Q3, Q2, Q1, s3, s1)
        rhs = Phi(s1, Q3, s3)/Phi(s2, Q2, s3)*T.RFm(s1, Q2, Q1, Q3, s2, s3)
        R.agree(f"cyclicity step (S10) #{trial}", lhs, rhs, 1e-10)
        g1 = T.Sh(Q3, s1, s3)*Phi(s1, Q3, s3)
        g2 = T.Sh(Q2, s3, s2)*Phi(s2, Q2, s3)
        R.agree(f"g = Shat Phi depends on the shared label only #{trial}", g1, g2, 1e-12)
        R.agree(f"g = s3/(sqrt2 rhohat(s3)) (kappa = 1/sqrt2) #{trial}", g1, s3/(np.sqrt(2)*T.rhoh(s3)), 1e-12)
        a = T.Ch3(Q3, Q2, Q1, s3, s2, s1)
        R.agree(f"cyclicity (C)^{{s3s2s1}}_{{P3P2P1}} = (C)^{{s2s1s3}}_{{P2P1P3}} #{trial}", a, T.Ch3(Q2, Q1, Q3, s2, s1, s3), 1e-10)
        R.agree(f"cyclicity (C)^{{s3s2s1}}_{{P3P2P1}} = (C)^{{s1s3s2}}_{{P1P3P2}} #{trial}", a, T.Ch3(Q1, Q3, Q2, s1, s3, s2), 1e-10)
        R.agree(f"(C) even in P3 #{trial}", a, T.Ch3(-Q3, Q2, Q1, s3, s2, s1), 1e-10)
        R.agree(f"(C) even in P2 #{trial}", a, T.Ch3(Q3, -Q2, Q1, s3, s2, s1), 1e-10)
        R.agree(f"(C) odd in s2 #{trial}", -a, T.Ch3(Q3, Q2, Q1, s3, -s2, s1), 1e-10)
        R.agree(f"Shat odd in P #{trial}", -T.Sh(Q3, s1, s3), T.Sh(-Q3, s1, s3), 1e-13)
        R.agree(f"Shat odd in s1 #{trial}", -T.Sh(Q3, s1, s3), T.Sh(Q3, -s1, s3), 1e-13)

    R.section("5. the weight-one pole of Shat, Eq. (S12)")
    s = 0.19j
    pred = -4/np.sqrt(2)/(np.pi*(bh + 1/bh)*T.rhoh(s))
    R.paper("residue -4 kappa/(pi (bh+1/bh) rhohat(s)), s = 0.19i", pred, "-98.72744079")
    for d in (1e-4, 1e-5):
        R.agree(f"delta * Shat(weight one + delta) -> residue, delta = {d:g}", d*T.Sh(T.one + d, s, s), pred, 30*d)
    rem = [T.Sh(T.one + d, s, s) - pred/d for d in (1e-3, 5e-4)]
    R.agree("the O(1) remainder Shat - residue/delta is finite (delta = 1e-3 vs 5e-4)", rem[0], rem[1], 1e-2)

    R.section("6. bulk-boundary coefficient: columns and parities, Eqs. (S19)-(S20)")
    sg = 0.03 + 0.19j
    for Pa in (0.15 + 0.4j, 0.3 - 0.2j):
        R.agree(f"Rhat(Pa, weight one | s) = (Pa/s) Rhat(Pa|s), Pa = {Pa}", T.Rh2(Pa, T.one, sg), Pa/sg*T.Rh1(Pa, sg), 1e-11)
        R.agree(f"Rhat(Pa, 0hat | s) = 4 sin(4 pi s Pa)/Pa, Pa = {Pa}", T.Rh2(Pa, T.zero, sg), T.Rh_psi0(Pa, sg), 1e-11)
        dds = (T.Rh1(Pa, sg + 1e-5) - T.Rh1(Pa, sg - 1e-5))/2e-5
        R.agree(f"Rhat(Pa, 0hat | s) = -(4 pi Pa)^-1 d/ds Rhat(Pa|s), Pa = {Pa}", T.Rh_psi0(Pa, sg), -dds/(4*np.pi*Pa), 1e-8)
        R.agree(f"Rhat(Pa, Pb | s) even in Pa, Pa = {Pa}", T.Rh2(Pa, 0.17j, sg), T.Rh2(-Pa, 0.17j, sg), 1e-10)
        R.agree(f"Rhat(Pa, Pb | s) even in Pb, Pa = {Pa}", T.Rh2(Pa, 0.17j + 0.05, sg), T.Rh2(Pa, -0.17j - 0.05, sg), 1e-10)
        R.agree(f"Rhat(Pa, Pb | s) odd in s, Pa = {Pa}", -T.Rh2(Pa, 0.17j, sg), T.Rh2(Pa, 0.17j, -sg), 1e-10)
        R.agree(f"Rhat(Pa | s) odd in Pa, Pa = {Pa}", -T.Rh1(Pa, sg), T.Rh1(-Pa, sg), 1e-15)
        for nu in (bh, 1/bh):
            for name, f in (("P Rhat(P|s)", lambda P: P*T.Rh1(P, sg)), ("P Rhat(P,0hat|s)", lambda P: P*T.Rh_psi0(P, sg))):
                R.agree(f"shift equation for {name}, nu = {nu:.4f}, Pa = {Pa}", f(Pa + 0.5j*nu) + f(Pa - 0.5j*nu),
                        2*np.cosh(2*np.pi*nu*sg)*f(Pa), 1e-12)
    R.done()


if __name__ == "__main__":
    main()
