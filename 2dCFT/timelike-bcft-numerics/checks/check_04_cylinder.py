"""
CHECK 04 - the cylinder (annulus) with a boundary insertion: open vs closed channel (Table II, Table S-cyl).
===========================================================================================================

A boundary field Psi_{P0} of type (a a) on one boundary (label s_a) of an annulus whose other boundary has label s_b.
Open channel (trace over the open-string states between s_a and s_b) and closed channel (paper Eq. (S15)):

  Z_ab(tau) = int_{eps+iR} dP/i  (Chat)^{a a b}_{P P0 P}/Shat(P; s_b, s_a) (-i tau)^{h0} V_tau(P; P0)
            = int_{eps+iR} dP/i  rhohat(P) Rhat(P, P0 | s_a) Rhat(P | s_b) V_{-1/tau}(P; P0),

with (Chat)/Shat = 2 RF_{s_a,P}[P0 P; s_a s_b] (the trace coefficient) and torus one-point blocks V_tau (c <= 1).
The open density is odd in P (a residue sum on the label-shifted lattice); the closed density is odd as well and its
poles are the discrete closed-string states (zeros of rho0(iP)).

Parameters (paper Sec. S8 and Table S-cyl): s_a = 0.19i, s_b = 0.24i, P0 = 0.17i, eps = 0.15, 1200 points on [-4.5, 4.5],
torus blocks to q^8, tau = i, 1.3i, 0.8i.  The values are imaginary (label factor 1/s_a of the inserted field).
Expected Z/i:
   chat = -2:  7.0457595358e-2, 3.5540341528e-2, 1.1085542829e-1
   chat = 1:   1.8690547140e-1, 1.1141832919e-1, 2.6262536964e-1
   chat = 0:   1.3550906932e-1, 7.6463322732e-2, 1.9760277078e-1
(the irrational point chat = 0.272 is in check_11_irrational_b.py).

Usage:  python checks/check_04_cylinder.py          (chat = -2, 1, 0; ~10 s)
        python checks/check_04_cylinder.py --full   (also eps-independence, ~15 s)
"""
import time

import numpy as np

from _common import Report, options
from tlbcft.theory import Timelike
from tlbcft.blocks import TorusBlock
from tlbcft.numerics import line, simpson


def cylinder(T, taus, sa=0.19j, sb=0.24j, P0=0.17j, eps=0.15, Y=4.5, npts=1200, Nmax=8):
    y, P = line(eps, Y, npts)
    dens_open = T.Ch3(P, P0, P, sa, sa, sb)/T.Sh(P, sb, sa)                  # = 2 RF_{sa,P}[P0 P; sa sb]
    dens_closed = T.rhoh(P)*T.Rh2(P, P0, sa)*T.Rh1(P, sb)
    TB = TorusBlock(T.chat, P0, Nmax=Nmax)
    Tc = TB.coeffs(P)
    h0 = T.hh(P0)
    out = {}
    for tau in taus:
        fo = dens_open*TB.value(tau, P, Tc)
        fc = dens_closed*TB.value(-1/tau, P, Tc)
        out[tau] = ((-1j*tau)**h0*simpson(fo, y), simpson(fc, y), max(abs(fo[0]), abs(fo[-1]), abs(fc[0]), abs(fc[-1])))
    return out, dens_open, P


CASES = [((1, 2), "-2", ("7.0457595358e-2", "3.5540341528e-2", "1.1085542829e-1")),
         ((1, 1), "1", ("1.8690547140e-1", "1.1141832919e-1", "2.6262536964e-1")),
         ((2, 3), "0", ("1.3550906932e-1", "7.6463322732e-2", "1.9760277078e-1"))]
TAUS = (1j, 1.3j, 0.8j)


def main():
    A = options(__doc__)
    R = Report(__file__, "cylinder with a boundary insertion: open channel = closed channel",
               "Eq. (S15); Table II row 'cylinder'; Table S-cyl")
    for (mn, chat, printed) in CASES:
        T = Timelike(mn)
        R.section(f"chat = {chat} (bhat^2 = {mn[0]}/{mn[1]}), s_a = 0.19i, s_b = 0.24i, P0 = 0.17i, eps = 0.15")
        t0 = time.time()
        res, dens_open, P = cylinder(T, TAUS)
        Pq = 0.15 + 0.7j
        R.agree("trace coefficient (Chat)^{aab}_{P P0 P}/Shat(P;b,a) = 2 RF_{sa,P}[P0 P; sa sb] at P = 0.15+0.7i",
                T.Ch3(Pq, 0.17j, Pq, 0.19j, 0.19j, 0.24j)/T.Sh(Pq, 0.24j, 0.19j), 2*T.RFm(0.19j, Pq, 0.17j, Pq, 0.19j, 0.24j), 1e-12)
        for tau, pr in zip(TAUS, printed):
            Zo, Zc, ends = res[tau]
            R.agree(f"tau = {tau}: modular covariance (|integrands| at the ends {ends:.0e})", Zo, Zc, 1e-9,
                    names=("open", "closed"))
            R.paper(f"tau = {tau}: Z/i", Zo, pr, part='imag')
        R.note(f"[{time.time() - t0:.0f} s]")
    if A.full:
        R.section("eps-independence: chat = -2 at eps = 0.3")
        res, _, _ = cylinder(Timelike((1, 2)), (1j,), eps=0.3)
        R.paper("tau = i: Z/i at eps = 0.3", res[1j][0], "7.0457595358e-2", part='imag')
        R.agree("tau = i: open = closed at eps = 0.3", res[1j][0], res[1j][1], 1e-9, names=("open", "closed"))
    R.done()


if __name__ == "__main__":
    main()
