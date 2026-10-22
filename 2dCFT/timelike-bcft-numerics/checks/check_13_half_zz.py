"""
CHECK 13 - half-ZZ branes of the Virasoro minimal string and the boundary states of Table I (Sec. S4 "Half-ZZ branes").
======================================================================================================================

The half-ZZ branes of the Virasoro minimal string have, relative to the measure rhohat (paper Eq. (S24)),
      Psi_(m,+)(P) = 4 sinh(2 pi m bhat P)/P,   Psi_(m,-)(P) = 4 sinh(2 pi m P/bhat)/P,     m = 1, 2, ...
They are EVEN in P, while all boundary states of Table I (Rhat(P|s) = 4 cos(4 pi s P)/P) are odd.  Verified here:
 (a) the annulus between ZZ_(m,+) and our state s = iv,
        int dP/i rhohat Psi_(m,+) Rhat(.|s) chi_P(qt) = sum_{r = 1-m, 3-m, ..., m-1} sum_{s' odd} sum_+- chi_{P(r,s') +- s}(q),
     with multiplicity ONE exactly in the normalization k^2 = 2 (the literal rotation gives 1/sqrt2): an independent
     confirmation of k^2 = 2; and the self-annulus of ZZ_(m,+) = sum_{s odd} sum_{r = 1,3,..,2m-1} chi^deg_(r,s);
 (b) Psi_(m,+-) is i^{-1} times the Psi_0 coefficient of Table I at a quantized label:
        Rhat(P, 0hat | s) = i Psi_(m,+-)(P)  at  s = (i/2) m bhat^{+-1},
     and with B_+-(s) = [Rhat(.|s) -+ i Rhat(., 0hat|s)]/2 = 2 e^{-+ 4 pi i s P}/P the half-ZZ state is B_+ - B_- at s = (i/2) m bhat;
 (c) the differences of our states at degenerate labels (the would-be ZZ branes),
        Rhat(P | (i/2)(m bhat + n/bhat)) - Rhat(P | (i/2)(m bhat - n/bhat)) = 8 sinh(2 pi m bhat P) sinh(2 pi n P/bhat)/P,
     have NEGATIVE self-annuli, -2 sum chi^deg (e.g. -2 chi^deg_(1,1) for m = n = 1): not Cardy branes;
 (d) the disc two-point function on ZZ_(1,+) (Table S-hzz): the bulk channel
        G(z) = int_{eps+iR} dP/i rhohat(P) Chat(P,P1,P2) Psi_(1,+)(P) V_z(P; P2 P1 P1 P2)        (even density: a continuum)
     equals a discrete sum over degenerate boundary fields sum_{(1,s), s odd <= 11} c_(1,s) V_{1-z}(P_(1,s); P1 P1 P2 P2),
     fitted by least squares at twelve z in [0.45, 0.9]: residuals ~1e-10, c_(1,1) = N_1 Psi(P1) Psi(P2) with the same
     N_1 for all externals (N_1 = 0.0754027552 at bhat^2 = 13/19, 0.0708883733 at 17/23), and factorization of c_(1,3)
     (paper: to 1e-4; with the more accurate blocks of tlbcft the residuals are ~1e-13, c_(1,3) factorizes to ~1e-8 and
     c_(1,5) ~ 1e-7 becomes resolvable and factorizes to ~1e-2; c_(1,7) is still below the resolution);
     [--full] for ZZ_(2,+) (fields (1,s), (3,s), s <= 7, 24 z in [0.35, 0.92]): N_2 = 0.0689305, and
     N_1/N_2 = Psi_(2,+)(0hat)/Psi_(1,+)(0hat) = -2 cos(pi bhat^2) = 1.0938963, as for Cardy states.
The degenerate blocks are regular (the fusion rules hold for r, s odd); they are evaluated as the average over
displacements +-1e-5, +-1e-5 i of the internal momentum.  bhat^2 = 13/19, 17/23: no coincidence with other Kac weights
below level 36.

Usage:  python checks/check_13_half_zz.py          (a)-(c) and the ZZ_(1,+) fit at 13/19, ~35 s
        python checks/check_13_half_zz.py --full   also 17/23 and ZZ_(2,+), ~2 min
"""
import time

import numpy as np
from scipy import integrate

from _common import Report, options
from tlbcft.theory import Timelike
from tlbcft.blocks import SphereBlock
from tlbcft.numerics import line, simpson


def eta_Z_closed(density, T, eps=0.11, L=10.0, N=400001):
    """eta(q) Z = T^{-1/2} int dy density(eps + i y) e^{2 pi P^2/T}, P = eps + i y (chi_P(qt) = qt^{-P^2}/eta(qt))."""
    y = np.linspace(-L, L, N)
    P = eps + 1j*y
    return integrate.simpson(density(P)*np.exp(2*np.pi*P**2/T), x=y)/np.sqrt(T)


def disc_fit(T, mZZ, pairs, zs, smax, Nmax=12, eps=0.3, Y=4.5, npts=1201):
    """least-squares fit of the bulk channel by degenerate boundary blocks; returns {pair: (coefficients, residual)}."""
    labels = [(r, s) for r in range(1, 2*mZZ, 2) for s in range(1, smax + 1, 2)]
    y, P = line(eps, Y, npts)
    out = {}
    for (u1, u2) in pairs:
        P1, P2 = 1j*u1, 1j*u2
        h1, h2 = T.hh(P1), T.hh(P2)
        dens = T.rhoh(P)*T.Ch(P, P1, P2)*4*np.sinh(2*np.pi*mZZ*T.bh*P)/P
        bs = SphereBlock(T.chat, h2, h1, h1, h2, Nmax=Nmax)
        H = bs.H_series(T.hh(P))
        G = np.array([simpson(dens*bs.value(z, T.hh(P), H), y) for z in zs])
        bt = SphereBlock(T.chat, h1, h1, h2, h2, Nmax=Nmax)
        M = np.zeros((len(zs), len(labels)), dtype=complex)
        for j, (r, s) in enumerate(labels):
            pts = T.Pdeg(r, s) + np.array([1e-5, -1e-5, 1e-5j, -1e-5j])
            Hd = bt.H_series(T.hh(pts))
            M[:, j] = [np.mean(bt.value(1 - z, T.hh(pts), Hd)) for z in zs]
        c, *_ = np.linalg.lstsq(M, G, rcond=None)
        out[(u1, u2)] = (c, np.max(np.abs(M @ c - G))/np.max(np.abs(G)), labels)
    return out


def main():
    A = options(__doc__)
    R = Report(__file__, "half-ZZ branes: annuli with our states, Psi_0 at quantized labels, the disc two-point function",
               "Sec. S4 'Half-ZZ branes', Eq. (S24); main text 'Half-ZZ branes'; Table S-hzz")
    bh = 0.7                                                  # generic (irrational) bhat for the annuli
    T = Timelike(bh**2)
    v = 0.13
    R.section("(a) annuli of ZZ_(m,+) with our boundary state s = 0.13i, and self-annuli (bhat = 0.7)")
    for m in (1, 2, 3):
        for k2, lab, mult in ((2.0, "Table I (k^2 = 2)", 1.0), (1.0, "literal rotation (k = 1)", 1/np.sqrt(2))):
            k = np.sqrt(k2)
            dens = lambda P: P**2/(2*T.rho0(1j*P))*4*np.sinh(2*np.pi*m*bh*P)/P*(k/np.sqrt(2))*4*np.cos(4*np.pi*1j*v*P)/P
            for Tm in (0.7, 1.3):
                q = np.exp(-2*np.pi*Tm)
                open_ = sum(q**((0.5*(r*bh - s/bh) + pm*v)**2) for r in range(1 - m, m, 2) for s in range(1, 201, 2)
                            for pm in (1, -1))
                R.agree(f"m = {m}, {lab}, T = {Tm}: closed = {mult:.4f} x sum_r,s,+- chi", eta_Z_closed(dens, Tm), mult*open_, 1e-11,
                        names=("closed", "open"))
        dens_self = lambda P: P**2/(2*T.rho0(1j*P))*(4*np.sinh(2*np.pi*m*bh*P)/P)**2
        for Tm in (0.7, 1.3):
            q = np.exp(-2*np.pi*Tm)
            deg = sum(q**((0.5*(r*bh - s/bh))**2) - q**((0.5*(r*bh + s/bh))**2) for r in range(1, 2*m, 2) for s in range(1, 201, 2))
            R.agree(f"m = {m}: self-annulus = sum_{{s odd}} sum_{{r=1,..,2m-1 odd}} chi^deg_(r,s), T = {Tm}", eta_Z_closed(dens_self, Tm),
                    deg, 1e-11, names=("closed", "open"))

    R.section("(b) Psi_(m,+-) from the Psi_0 coefficient at quantized labels; B_+ - B_-")
    Pq = np.array([0.15 + 0.4j, 0.3 - 0.2j])
    for m in (1, 2):
        for sgn, nu in (("+", bh), ("-", 1/bh)):
            s = 0.5j*m*nu
            R.agree(f"Rhat(P, 0hat | (i/2) m bhat^(+-1)) = i Psi_(m,{sgn}), m = {m}", T.Rh_psi0(Pq, s),
                    1j*4*np.sinh(2*np.pi*m*nu*Pq)/Pq, 1e-13)
        s = 0.5j*m*bh
        Bp = (T.Rh1(Pq, s) - 1j*T.Rh_psi0(Pq, s))/2
        Bm = (T.Rh1(Pq, s) + 1j*T.Rh_psi0(Pq, s))/2
        R.agree(f"B_+ = 2 e^(-4 pi i s P)/P, m = {m}", Bp, 2*np.exp(-4j*np.pi*s*Pq)/Pq, 1e-13)
        R.agree(f"B_+ - B_- = Psi_(m,+) (half-ZZ = difference of the chiral states), m = {m}", Bp - Bm,
                4*np.sinh(2*np.pi*m*bh*Pq)/Pq, 1e-13)

    R.section("(c) ZZ-type differences of our states: negative self-annuli")
    for (m, n) in ((1, 1), (2, 1)):
        sp, sm = 0.5j*(m*bh + n/bh), 0.5j*(m*bh - n/bh)
        D = lambda P: T.Rh1(P, sp) - T.Rh1(P, sm)
        R.agree(f"(m,n) = ({m},{n}): difference = 8 sinh(2pi m bhat P) sinh(2pi n P/bhat)/P", D(Pq),
                8*np.sinh(2*np.pi*m*bh*Pq)*np.sinh(2*np.pi*n*Pq/bh)/Pq, 1e-12)
    for Tm in (0.7, 1.3):
        q = np.exp(-2*np.pi*Tm)
        sp, sm = 0.5j*(bh + 1/bh), 0.5j*(bh - 1/bh)
        dens = lambda P: T.rhoh(P)*(T.Rh1(P, sp) - T.Rh1(P, sm))**2
        chideg11 = q**((0.5*(bh - 1/bh))**2) - q**((0.5*(bh + 1/bh))**2)
        R.agree(f"(m,n) = (1,1), T = {Tm}: self-annulus = -2 chi^deg_(1,1)", eta_Z_closed(dens, Tm), -2*chideg11, 1e-10,
                names=("closed", "open"))

    R.section("(d) disc two-point function on ZZ_(1,+): fit by degenerate boundary blocks (1,s), s <= 11 odd")
    cases = [((13, 19), "0.0754027552", (("0.13,0.21", "24.52255931", "1.58950e-2"), ("0.13,0.13", "27.92135465", "-0.348034"),
                                         ("0.21,0.21", "21.53749068", "-7.2587e-4")))]
    if A.full:
        cases.append(((17, 23), "0.0708883733", (("0.13,0.21", "24.32064841", "-1.24776e-2"), ("0.13,0.13", "28.00111287", "-0.196152"),
                                                  ("0.21,0.21", "21.12394395", "-7.9374e-4"))))
    for (mn, N1, rows) in cases:
        t0 = time.time()
        Tz = Timelike(mn)
        pairs = [tuple(float(x) for x in r[0].split(',')) for r in rows]
        res = disc_fit(Tz, 1, pairs, np.linspace(0.45, 0.9, 12), 11)
        psi = lambda u: 4*np.sinh(2*np.pi*Tz.bh*1j*u)/(1j*u)
        for (pair, c11, c13), (u1, u2) in zip(rows, pairs):
            c, resid, labels = res[(u1, u2)]
            R.small(f"bhat^2 = {mn[0]}/{mn[1]}, (u1,u2) = ({pair}): fit residual", resid, 1e-8)
            R.paper(f"   N_1 = c_(1,1)/(Psi(P1) Psi(P2))", c[0]/(psi(u1)*psi(u2)), N1)
            R.paper(f"   c_(1,1)", c[0], c11)
            R.paper(f"   c_(1,3)", c[1], c13, rtol=2e-4)
        cab, caa, cbb = (res[p][0] for p in pairs)
        R.agree(f"bhat^2 = {mn[0]}/{mn[1]}: factorization c_(1,1)(a,b)^2 = c_(1,1)(a,a) c_(1,1)(b,b)", cab[0]**2, caa[0]*cbb[0], 1e-9)
        R.agree(f"bhat^2 = {mn[0]}/{mn[1]}: factorization c_(1,3)(a,b)^2 = c_(1,3)(a,a) c_(1,3)(b,b)", cab[1]**2, caa[1]*cbb[1], 3e-4)
        for j in (2,):              # resolvable with the equilibrated-solve blocks (the paper's fit resolved only c_(1,3))
            R.agree(f"bhat^2 = {mn[0]}/{mn[1]}: factorization c_(1,{2*j+1})(a,b)^2 = c_(1,{2*j+1})(a,a) c_(1,{2*j+1})(b,b)",
                    cab[j]**2, caa[j]*cbb[j], 2e-2)
        R.note(f"[{time.time() - t0:.0f} s]")
    if A.full:
        R.section("(d') disc two-point function on ZZ_(2,+): fields (1,s), (3,s), s <= 7, 24 z in [0.35, 0.92], bhat^2 = 13/19")
        t0 = time.time()
        Tz = Timelike((13, 19))
        pairs = [(0.13, 0.21), (0.13, 0.13), (0.21, 0.21)]
        res = disc_fit(Tz, 2, pairs, np.linspace(0.35, 0.92, 24), 7)
        psi2 = lambda u: 4*np.sinh(4*np.pi*Tz.bh*1j*u)/(1j*u)
        N2s = []
        for (u1, u2) in pairs:
            c, resid, labels = res[(u1, u2)]
            R.small(f"(u1,u2) = ({u1},{u2}): fit residual", resid, 1e-8)
            N2s.append(c[0]/(psi2(u1)*psi2(u2)))
            R.paper(f"   N_2 = c_(1,1)/(Psi(P1) Psi(P2))", N2s[-1], "0.0689305")
        N1 = 0.0754027552
        R.paper("N_1/N_2", N1/N2s[0], "1.093896")
        R.agree("N_1/N_2 = Psi_(2,+)(0hat)/Psi_(1,+)(0hat) = -2 cos(pi bhat^2)", N1/N2s[0], -2*np.cos(np.pi*13/19), 1e-6)
        R.note(f"[{time.time() - t0:.0f} s]")
    R.done()


if __name__ == "__main__":
    main()
