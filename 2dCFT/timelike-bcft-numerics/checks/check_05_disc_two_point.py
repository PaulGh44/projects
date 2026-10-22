"""
CHECK 05 - the disc two-point function on an FZZT-type boundary (Table II row "disc two-point", Table S-disc2).
=============================================================================================================

Two bulk fields V_{P1}, V_{P2} on a disc with boundary label s.  Bulk channel (bulk OPE, then the one-point function)
and boundary channel (each bulk field to the boundary, then the boundary two-point function), paper Eq. (4):

  G(z) = int_{eps+iR} dP/i  rhohat(P) Chat(P, P1, P2) Rhat(P | s) V_z(P; P2 P1 P1 P2)
       = int_{eps+iR} dP/i  Rhat(P1, P | s) Rhat(P2, P | s) / Shat(P; s, s) * V_{1-z}(P; P1 P1 P2 P2),

z the cross-ratio of the doubled points.  Both densities are odd in P.  This constraint fixes the constant kappa of
Shat (Eq. (S11)): with kappa four times larger the boundary channel becomes a quarter of the bulk channel.

Parameters (Table S-disc2): P1,2 = 0.13i, 0.21i, s = 0.19i, eps = 0.3, Simpson with 1001 points on [-4.5, 4.5],
level-12 blocks, z = 0.35 and 0.5.  Expected bulk-channel values:
   chat = 1       6.4716512430   5.5151927197
   chat = -2      7.5621654080   6.1958180167
   chat = 0       6.8377917235   5.7518016370
   chat = -22/5   8.4126775524   6.6747417620
   chat = -7      9.3045494869   7.1252876847
   chat = 11/35   6.7229558686   5.6784571994      (bhat^2 = 5/7; the closed-form modular kernel at b'^2 = 10/7
                                                      limits the channel agreement to ~3e-9)
Usage:  python checks/check_05_disc_two_point.py          (chat = -2 and 1, ~20 s)
        python checks/check_05_disc_two_point.py --full   (all six, ~10 min; the 11/35 row dominates)
"""
import time

import numpy as np

from _common import Report, options
from tlbcft.theory import Timelike
from tlbcft.blocks import SphereBlock
from tlbcft.numerics import line, simpson


def disc_two_point(T, zs=(0.35, 0.5), u=(0.13, 0.21), v=0.19, eps=0.3, Y=4.5, npts=1001, Nmax=12):
    P1, P2, s = 1j*u[0], 1j*u[1], 1j*v
    y, P = line(eps, Y, npts)
    dens_bulk = T.rhoh(P)*T.Ch(P, P1, P2)*T.Rh1(P, s)
    dens_bdy = T.Rh2(P1, P, s)*T.Rh2(P2, P, s)/T.Sh(P, s, s)
    h1, h2 = T.hh(P1), T.hh(P2)
    bs = SphereBlock(T.chat, h2, h1, h1, h2, Nmax=Nmax)
    bt = SphereBlock(T.chat, h1, h1, h2, h2, Nmax=Nmax)
    Hs, Ht = bs.H_series(T.hh(P)), bt.H_series(T.hh(P))
    out = {}
    for z in zs:
        out[z] = (simpson(dens_bulk*bs.value(z, T.hh(P), Hs), y), simpson(dens_bdy*bt.value(1 - z, T.hh(P), Ht), y))
    return out


CASES = [((1, 2), "-2", ("7.5621654080", "6.1958180167"), 1e-10),
         ((1, 1), "1", ("6.4716512430", "5.5151927197"), 1e-10),
         ((2, 3), "0", ("6.8377917235", "5.7518016370"), 1e-10),
         ((2, 5), "-22/5", ("8.4126775524", "6.6747417620"), 1e-9),
         ((1, 3), "-7", ("9.3045494869", "7.1252876847"), 1e-10),
         ((5, 7), "11/35", ("6.7229558686", "5.6784571994"), 1e-8)]


def main():
    A = options(__doc__)
    R = Report(__file__, "disc two-point function: bulk channel = boundary channel",
               "Eq. (4) second line; Eq. (S16); Table II row 'disc two-point'; Table S-disc2")
    for (mn, chat, printed, tol) in (CASES if A.full else CASES[:2]):
        T = Timelike(mn)
        R.section(f"chat = {chat} (bhat^2 = {mn[0]}/{mn[1]}), P1,2 = 0.13i, 0.21i, s = 0.19i, eps = 0.3, 1001 points")
        t0 = time.time()
        res = disc_two_point(T)
        for z, pr in zip((0.35, 0.5), printed):
            Gb, Gt = res[z]
            R.agree(f"z = {z}", Gb, Gt, tol, names=("bulk", "boundary"))
            R.paper(f"z = {z}: bulk channel", Gb, pr)
        if mn == (1, 2):
            T4 = Timelike(mn, scales=(T.k1, T.k2, 4*T.k3, 4*T.k4))       # Shat four times larger, same vertex RF
            res4 = disc_two_point(T4, zs=(0.5,))
            R.ratio("kappa four times larger: bulk/boundary at z = 0.5", res4[0.5][0], res4[0.5][1], 4.0, 1e-9)
        R.note(f"[{time.time() - t0:.0f} s]")
    R.done()


if __name__ == "__main__":
    main()
