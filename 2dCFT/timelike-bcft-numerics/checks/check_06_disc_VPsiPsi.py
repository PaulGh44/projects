"""
CHECK 06 - the disc with one bulk and two boundary fields, V Psi Psi (Table II, Table S-vpp).
=============================================================================================

Bulk field V_{Pa}(z), z = e^{i phi}, boundary fields Psi_{Pb1}(0) and Psi_{Pb2}(infinity), boundary label s2 on the
positive and s1 on the negative half line.  Channel (i) expands the bulk field on the s2 segment, channel (ii) on
the s1 segment (paper Eqs. (S26)-(S27)):

  G_i  = int dP/i  Rhat(Pa, P | s2) (Chat)^{s2 s1 s2}_{P Pb1 Pb2}/Shat(P; s2, s2)  W_i(P),
  G_ii = int dP/i  Rhat(Pa, P | s1) (Chat)^{s1 s2 s1}_{P Pb2 Pb1}/Shat(P; s1, s1)  W_ii(P),
  [(Chat)/Shat = 2 RF:  dens_i = 2 Rhat(Pa,P|s2) RF_{s1,P}[Pb1 Pb2; s2 s2]],

  W_{i,ii}(P) = e^{-+ i pi/2 (2 ha + hb1 - hb2)} e^{-+ i pi/2 (h - 2 ha)} V(P; tau = +-1/2 + i/2),

V the four-point block of the doubled correlator (ha at 0, ha at eta, hb1 at 1, hb2 at infinity) continued from
tau = i oo to tau = +-1/2 + i/2, where eta = lambda(tau) = 2 +- i0 (phi = pi/2).  Both densities are odd; the
integrals are midpoint sums on eps + i[-4.5, 4.5].  Nothing can be adjusted: both channels scale alike under every
rescaling of the data.  At c >= 25 the same constraint (with R0, F, P in iR) is the torus two-point Moore-Seiberg
relation, Eq. (S28).

Parameters (Table S-vpp): Pa = 0.13i, Pb1 = 0.21i, Pb2 = 0.17i, s1 = 0.19i, s2 = 0.24i, level-12 elliptic blocks,
midpoint rule (N points).

Accuracy note.  The blocks are needed at |q| = e^{-pi/2} = 0.21, where the high-order coefficients of H(q) suffer
cancellations.  The paper's runs inverted the (badly conditioned) Gram matrices, which limited the blocks there to
1e-5 - 1e-3 relative for |P| = 0.6 - 2 (check_00, section 6, shows this on the exact c = 1 block); in the integrated
amplitudes it left errors of 3e-8 to 3e-6 relative in the printed values of Table S-vpp (the largest in the c = 28 row).
tlbcft.blocks solves the equilibrated Gram system instead: the two channels then agree to 1e-11 - 2e-10 (2e-9 at
11/35, limited by the modular kernel), and at chat = -2 the values change by 1.5e-10 from level 12 to 13, by 3e-11
from 800 to 1200 points and by 5e-11 from eps = 0.5 to 0.3 (G_i = G_ii = -3.8153907793).  The comparison with the printed values below therefore uses the
relative tolerance 5e-6, the accuracy those values actually have; the channel agreement is tested at 1e-8 - 1e-9.
Printed values (real parts):
   c = 28 (c >= 25 data)   N = 600          2.616050347e-2    2.616050391e-2
   chat = -2   eps = 0.5   N = 800         -3.815389907      -3.815389941
   chat = -2   eps = 0.3   N = 600         -3.815389170      -3.815386944
   chat = 1    eps = 0.5   N = 800        -12.27660319      -12.27660660
   chat = 0                                -8.363401364      -8.363400122
   chat = -22/5                            -1.426896173      -1.426896322
   chat = -7                               -0.4543253832     -0.4543252249
   chat = 11/35 (bhat^2 = 5/7)             -9.440668494      -9.440668401

Usage:  python checks/check_06_disc_VPsiPsi.py          (chat = -2 at eps = 0.5, a few seconds)
        python checks/check_06_disc_VPsiPsi.py --full   (all rows, ~8 min; the 11/35 row dominates)
"""
import time

import numpy as np

from _common import Report, options
from tlbcft.theory import Timelike, Spacelike
from tlbcft.blocks import SphereBlock
from tlbcft.numerics import midpoint

PA, PB1, PB2, S1, S2 = 0.13j, 0.21j, 0.17j, 0.19j, 0.24j


def channels(D, timelike, N=800, eps=0.5, Y=4.5, Nmax=12):
    if timelike:
        c, h = D.chat, D.hh
        y, P, w = midpoint(eps, Y, N)
        di = D.Rh2(PA, P, S2)*D.Ch3(P, PB1, PB2, S2, S1, S2)/D.Sh(P, S2, S2)
        dii = D.Rh2(PA, P, S1)*D.Ch3(P, PB2, PB1, S1, S2, S1)/D.Sh(P, S1, S1)
    else:
        c, h = D.c, D.h
        y, P, w = midpoint(0.0, Y, N)
        di = D.R2(PA, P, S2)*D.Fm(S1, P, PB1, PB2, S2, S2)
        dii = D.R2(PA, P, S1)*D.Fm(S2, P, PB2, PB1, S1, S1)
    ha, hb1, hb2 = h(PA), h(PB1), h(PB2)
    blk = SphereBlock(c, ha, ha, hb1, hb2, Nmax=Nmax)
    hP = h(P)
    H = blk.H_series(hP)
    WA = np.exp(-0.5j*np.pi*(2*ha + hb1 - hb2))*np.exp(-0.5j*np.pi*(hP - 2*ha))*blk.at_tau(0.5 + 0.5j, hP, H)
    WB = np.exp(+0.5j*np.pi*(2*ha + hb1 - hb2))*np.exp(+0.5j*np.pi*(hP - 2*ha))*blk.at_tau(-0.5 + 0.5j, hP, H)
    fi, fii = di*WA, dii*WB
    return w*np.sum(np.nan_to_num(fi)), w*np.sum(np.nan_to_num(fii)), max(abs(fi[0]), abs(fi[-1]), abs(fii[0]), abs(fii[-1]))


ROWS = [  # (data, label, N, eps, printed G_i, printed G_ii, tolerance for G_i = G_ii)
    (((1, 2), True), "chat = -2, eps = 0.5", 800, 0.5, "-3.815389907", "-3.815389941", 1e-9),
    (((1, 2), False), "c = 28 (c >= 25 data, P in iR)", 600, None, "2.616050347e-2", "2.616050391e-2", 1e-8),
    (((1, 2), True), "chat = -2, eps = 0.3", 600, 0.3, "-3.815389170", "-3.815386944", 1e-9),
    (((1, 1), True), "chat = 1", 800, 0.5, "-12.27660319", "-12.27660660", 1e-8),
    (((2, 3), True), "chat = 0", 800, 0.5, "-8.363401364", "-8.363400122", 1e-8),
    (((2, 5), True), "chat = -22/5", 800, 0.5, "-1.426896173", "-1.426896322", 1e-8),
    (((1, 3), True), "chat = -7", 800, 0.5, "-0.4543253832", "-0.4543252249", 1e-8),
    (((5, 7), True), "chat = 11/35 (bhat^2 = 5/7)", 800, 0.5, "-9.440668494", "-9.440668401", 1e-8),
]


def main():
    A = options(__doc__)
    R = Report(__file__, "disc with one bulk and two boundary fields (V Psi Psi): channel (i) = channel (ii)",
               "Eqs. (S26)-(S28); Table II row 'disc V Psi Psi'; Table S-vpp")
    for ((mn, tl), lab, N, eps, pi_, pii, tol) in (ROWS if A.full else ROWS[:1]):
        R.section(f"{lab}: midpoint rule with N = {N}, level-12 blocks at tau = +-1/2 + i/2")
        t0 = time.time()
        D = Timelike(mn) if tl else Spacelike(mn)
        Gi, Gii, ends = channels(D, tl, N=N, eps=eps if eps else 0.5)
        R.agree(f"G_i = G_ii (|integrands| at the ends {ends:.0e})", Gi, Gii, tol, names=("G_i", "G_ii"))
        R.paper("G_i (real part) vs paper, rel. tol 5e-6", Gi, pi_, rtol=5e-6)
        R.paper("G_ii (real part) vs paper, rel. tol 5e-6", Gii, pii, rtol=5e-6)
        R.note(f"[{time.time() - t0:.0f} s]")
    R.done()


if __name__ == "__main__":
    main()
