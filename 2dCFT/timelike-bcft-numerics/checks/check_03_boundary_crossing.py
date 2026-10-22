"""
CHECK 03 - boundary crossing: the boundary four-point function on the physical contour (Table II, Table S-4pt).
================================================================================================================

Four boundary fields Psi_{P1..P4} (P_j = i u_j) between boundary conditions s1..s4 (s_j = i v_j).  The s-channel
(paper Eq. (4), first line) and the t-channel are

  G_s(z) = int_{eps+iR} dP/i (Chat)^{s3 s2 s1}_{P P2 P1} (Chat)^{s4 s3 s1}_{P4 P3 P} / Shat(P; s1, s3) * V_z(P; P1 P2 P3 P4),
  G_t(z) = int_{eps+iR} dP/i (Chat)^{s4 s3 s2}_{P P3 P2} (Chat)^{s4 s2 s1}_{P4 P P1} / Shat(P; s2, s4) * V_{1-z}(P; P3 P2 P1 P4),

with the data of Table I (final normalization) and c <= 1 sphere blocks.  Crossing symmetry is G_s(z) = G_t(z).
Both densities are odd in P, so each integral is a pure residue sum on the label-shifted lattice (Eq. (5)); the
line integral does not depend on eps > 0 (for imaginary labels all poles lie on iR; checked in --full).

Parameters (paper Sec. S8): u = (0.13, 0.21, 0.17, 0.29), v = (0.19, 0.21, 0.24, 0.28), eps = 0.15, Simpson's rule
with 1200 points on [-4.5, 4.5], level-12 elliptic blocks.  Expected: Table S-4pt,
   chat = 1      z = 0.5   13.628318626              z = 0.35  13.079874102
   chat = -2     z = 0.5   0.42142571622
   chat = 0      z = 0.5   4.3546380859
   chat = -22/5  z = 0.5   2.1076975609e-2
   chat = -7     z = 0.35  3.3683382216e-4
(the irrational point chat = 0.272 is in check_11_irrational_b.py).

Usage:  python checks/check_03_boundary_crossing.py          (chat = 1 and -2, ~15 s)
        python checks/check_03_boundary_crossing.py --full   (all five central charges + eps-independence, ~1 min)
"""
import time

import numpy as np

from _common import Report, options
from tlbcft.theory import Timelike
from tlbcft.blocks import SphereBlock
from tlbcft.numerics import line, simpson

U = np.array([0.13, 0.21, 0.17, 0.29])
V = np.array([0.19, 0.21, 0.24, 0.28])


def four_point(T, zs, eps=0.15, Y=4.5, npts=1200, Nmax=12):
    """Returns {z: (G_s, G_t)} for the boundary four-point function with the data of T."""
    P1, P2, P3, P4 = 1j*U
    s1, s2, s3, s4 = 1j*V
    y, P = line(eps, Y, npts)
    dens_s = T.Ch3(P, P2, P1, s3, s2, s1)*T.Ch3(P4, P3, P, s4, s3, s1)/T.Sh(P, s1, s3)
    dens_t = T.Ch3(P, P3, P2, s4, s3, s2)*T.Ch3(P4, P, P1, s4, s2, s1)/T.Sh(P, s2, s4)
    h = T.hh(1j*U)
    bs = SphereBlock(T.chat, h[0], h[1], h[2], h[3], Nmax=Nmax)
    bt = SphereBlock(T.chat, h[2], h[1], h[0], h[3], Nmax=Nmax)
    Hs, Ht = bs.H_series(T.hh(P)), bt.H_series(T.hh(P))
    out = {}
    for z in zs:
        fs = dens_s*bs.value(z, T.hh(P), Hs)
        ft = dens_t*bt.value(1 - z, T.hh(P), Ht)
        out[z] = (simpson(fs, y), simpson(ft, y), max(abs(fs[0]), abs(fs[-1]), abs(ft[0]), abs(ft[-1])))
    return out


CASES = [((1, 1), "1", {0.5: "13.628318626", 0.35: "13.079874102"}),
         ((1, 2), "-2", {0.5: "0.42142571622"}),
         ((2, 3), "0", {0.5: "4.3546380859"}),
         ((2, 5), "-22/5", {0.5: "2.1076975609e-2"}),
         ((1, 3), "-7", {0.35: "3.3683382216e-4"})]


def main():
    A = options(__doc__)
    R = Report(__file__, "boundary crossing of the boundary four-point function on eps + iR",
               "Eq. (4) first line; Table II row 'boundary crossing'; Table S-4pt; Sec. S8 'Parameters'")
    cases = CASES if A.full else CASES[:2]
    for (mn, chat, printed) in cases:
        T = Timelike(mn)
        R.section(f"chat = {chat} (bhat^2 = {mn[0]}/{mn[1]}), eps = 0.15, 1200 points, level 12")
        t0 = time.time()
        res = four_point(T, sorted(printed, reverse=True))
        for z, (Gs, Gt, ends) in res.items():
            R.agree(f"z = {z}: crossing (|integrands| at the ends {ends:.0e})", Gs, Gt, 1e-9, names=("G_s", "G_t"))
            R.paper(f"z = {z}: G_s", Gs, printed[z])
        R.note(f"[{time.time() - t0:.0f} s]")
    if A.full:
        R.section("eps-independence: chat = -2 at eps = 0.3 (all poles of the densities lie on iR)")
        res = four_point(Timelike((1, 2)), [0.5], eps=0.3)
        R.paper("z = 0.5: G_s at eps = 0.3", res[0.5][0], "0.42142571622")
        R.agree("z = 0.5: crossing at eps = 0.3", res[0.5][0], res[0.5][1], 1e-9, names=("G_s", "G_t"))
    R.done()


if __name__ == "__main__":
    main()
