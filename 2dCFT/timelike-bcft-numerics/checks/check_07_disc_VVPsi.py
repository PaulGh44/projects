"""
CHECK 07 - the disc with two bulk fields and one boundary field, V V Psi; it fixes k^2 = 2 (Table II, Table S-vvp).
===================================================================================================================

V_{P1}(z1) V_{P2}(z2) Psi_{Pb}(x) on a disc with boundary label s.  The doubled correlator is a chiral five-point
function, expanded in comb blocks in two channels (paper Eq. (4), third line, and Sec. S6):

  bulk channel      G_bulk = int dP/i  rhohat(P) Chat(P, P1, P2) Rhat(P, Pb | s)  W(P)
  boundary channel  G_bdy  = int int dP'/i dP''/i  Rhat(P1, P' | s) Rhat(P2, P'' | s) (Chat)^{sss}_{Pb P'' P'}
                                                  / (Shat(P'; s, s) Shat(P''; s, s))  W'(P', P'')

with W, W' the five-point comb blocks of tlbcft/comb5.py (symmetric frame, normalised by their OPE limits).
The bulk density is EVEN in P (a principal-value continuum), the boundary density is odd in P' and in P'' (a double
residue sum).  Under the one-parameter rescaling left free by all other constraints (boundary states x k, Shat x 1/k)
the bulk channel is invariant and the boundary channel scales as k^2; with the literal rotation (k = 1) the bulk
channel is exactly TWICE the boundary channel, so the constraint fixes k^2 = 2, the normalization of Table I
(paper, "The disc V V Psi fixes k").

Parameters (Table S-vvp): z_{1,2} = -+a + i y, boundary field at x; externals (P1, P2, Pb, s) = (0.13, 0.21, 0.17, 0.19)i
("A") or (0.11, 0.23, 0.29, 0.16)i ("B"); midpoint rule with step 0.1 on eps + i[-4.5, 4.5] (90 x 90 points for the
boundary channel), eps = 0.4; comb blocks to level N.  Values are imaginary; G/i is listed.  The agreement improves
with N at the rates of Sec. S6 ((sqrt2 - 1)^N at (a, y, x) = (1, 1, 0)).

Accuracy at bhat^2 = 5/7 (chat = 11/35).  The boundary channel needs the modular kernel RS_{s,Pa}[P] with P on the
contour; at 5/7 it comes from the closed-form fusion kernel at b'^2 = 10/7, which there agrees with the independent
Barnes-integral kernel (tlbcft.kernels.ModularKernelGeneral) only to 3e-9 - 3e-8.  Accordingly G_bdy has a spurious
real part ~2e-8, and the printed G_bdy values of the 11/35 rows (which differ from ours by 7.6e-9 relative) are
compared at 3e-8.  The bulk channel and the other rows agree with the printed digits.

Usage:  python checks/check_07_disc_VVPsi.py          (chat = -2, A, (1,1,0), N = 8, plus the literal ratio; ~20 s)
        python checks/check_07_disc_VVPsi.py --full   (all rows of Table S-vvp; ~1 hour, the 11/35 and level-12 rows dominate)
"""
import time

import numpy as np

from _common import Report, options
from tlbcft.theory import Timelike, Spacelike
from tlbcft.comb5 import partial_waves
from tlbcft.numerics import midpoint

EXT = {'A': (0.13j, 0.21j, 0.17j, 0.19j), 'B': (0.11j, 0.23j, 0.29j, 0.16j)}


def vvpsi_timelike(datasets, ext, geo, Nlist, eps=0.4, Y=4.5, hstep=0.1):
    """{N: [(G_bulk, G_bdy) for each data object]} for c <= 1 data objects sharing chat."""
    P1, P2, Pb, s = ext
    T0 = datasets[0]
    c, h = T0.chat, T0.hh
    y, P, w = midpoint(eps, Y, int(round(2*Y/hstep)))
    dens_bulk = [T.rhoh(P)*T.Ch(P1, P2, P)*T.Rh2(P, Pb, s) for T in datasets]
    dens_bdy = []
    for T in datasets:
        R1, R2, Sp = T.Rh2(P1, P, s), T.Rh2(P2, P, s), T.Sh(P, s, s)
        dens_bdy.append(np.array([R1*R2[l]*T.Ch3(Pb, P[l], P, s, s, s)/(Sp*Sp[l]) for l in range(len(P))]))  # [P'', P']
    out = {}
    for N in Nlist:
        Wb = partial_waves('bulk', c, h(P1), h(P2), h(Pb), h(P), h(P), N, *geo)
        Wd = np.array([partial_waves('bdy', c, h(P1), h(P2), h(Pb), h(P), np.full(len(P), h(P[l])), N, *geo)
                       for l in range(len(P))])
        out[N] = [(w*np.sum(db*Wb), w*w*np.sum(np.nan_to_num(dd*Wd))) for db, dd in zip(dens_bulk, dens_bdy)]
    return out


def vvpsi_spacelike(S, ext, geo, Nlist, Y=4.5, hstep=0.1):
    """c >= 25: both densities are even, so the integrals over iR are twice the half-line midpoint sums."""
    P1, P2, Pb, s = ext
    c, h = S.c, S.h
    y = (np.arange(int(round(Y/hstep))) + 0.5)*hstep
    P = 1j*y
    dens_bulk = S.C0(P1, P2, P)*S.S(s, P, Pb)                        # (rho0/2) C0 R0(P, Pb|s) = C0 S_{s,P}[Pb]
    out = {}
    for N in Nlist:
        Wb = partial_waves('bulk', c, h(P1), h(P2), h(Pb), h(P), h(P), N, *geo)
        Gb = 2*hstep*np.sum(dens_bulk*Wb)
        Gd = 0j
        for l, Ppp in enumerate(P):
            d2 = (S.R2(P1, P, s)*S.R2(P2, Ppp, s)*S.C3(Pb, Ppp, P, s, s, s)/(2*S.S0(P, s, s)*2*S.S0(Ppp, s, s)))
            Wd = partial_waves('bdy', c, h(P1), h(P2), h(Pb), h(P), np.full(len(P), h(Ppp)), N, *geo)
            Gd += (2*hstep)**2*np.sum(np.nan_to_num(d2*Wd))
        out[N] = (Gb, Gd)
    return out


#            (bhat^2, ext, geo, N, printed G_bulk/i, printed G_bdy/i, channel tolerance)
ROWS = [((1, 2), 'A', (1, 1, 0), 8, "5.998671366", "5.998664419", 3e-6),
        ((1, 2), 'A', (1, 1, 0), 10, "5.998693704", "5.998694667", 5e-7),
        ((1, 2), 'A', (1, 1, 0), 12, "5.998690511", "5.998690374", 1e-7),
        ((1, 2), 'B', (1, 1, 0), 8, "5.673007893", "5.673000815", 3e-6),
        ((1, 2), 'A', (1, 1.3, 0), 8, "7.563181674", "7.563000911", 1e-4),
        ((1, 2), 'A', (1, 1, 0.37), 12, "5.998079905", "5.997667705", 3e-4),
        ((1, 1), 'A', (1, 1, 0), 10, "5.550342680", "5.550343405", 5e-7),
        ((2, 3), 'A', (1, 1, 0), 10, "5.730089968", "5.730090794", 5e-7),
        ((2, 5), 'A', (1, 1, 0), 10, "6.161474704", "6.161475584", 5e-7),
        ((1, 3), 'A', (1, 1, 0), 10, "6.145932206", "6.145932714", 5e-7),
        ((5, 7), 'A', (1, 1, 0), 8, "5.676853951", "5.676848487", 3e-6),
        ((5, 7), 'A', (1, 1, 0), 10, "5.676853867", "5.676854630", 5e-7),
        ((5, 7), 'A', (1, 1, 0), 12, "5.676853879", "5.676853730", 1e-7)]
LITERAL = {8: "2.0000023", 10: "1.9999997", 12: "2.0000000457"}


def main():
    A = options(__doc__)
    R = Report(__file__, "disc with two bulk fields and one boundary field (V V Psi); normalization k^2 = 2",
               "Eq. (4); Sec. S6; 'The disc V V Psi fixes k'; Table II row 'disc V V Psi'; Table S-vvp")
    rows = ROWS if A.full else ROWS[:1]
    groups = {}
    for row in rows:
        groups.setdefault(row[:3], []).append(row)
    for (mn, ext, geo), rr in groups.items():
        Ns = [r[3] for r in rr]
        t0 = time.time()
        T = Timelike(mn)
        Tlit = Timelike(mn, normalization='literal')
        R.section(f"chat = {T.chat:.4f} (bhat^2 = {mn[0]}/{mn[1]}), externals {ext}, (a, y, x) = {geo}, levels {Ns}")
        res = vvpsi_timelike([T, Tlit], EXT[ext], geo, Ns)
        for (_, _, _, N, pb, pd, tol) in rr:
            (Gb, Gd), (Gbl, Gdl) = res[N]
            R.agree(f"N = {N}: bulk = boundary (Table I data)", Gb, Gd, tol, names=("G_bulk", "G_bdy"))
            R.paper(f"N = {N}: G_bulk/i", Gb, pb, part='imag')
            # at bhat^2 = 5/7 the boundary channel needs RS_{s,Pa}[P] with P on the contour, from the closed-form
            # fusion kernel at b'^2 = 10/7 (mn = 70); there it agrees with the Barnes-integral kernel only to
            # 3e-9 - 3e-8 (and G_bdy acquires a spurious real part ~2e-8), so the printed digits are compared at 3e-8
            R.paper(f"N = {N}: G_bdy/i", Gd, pd, part='imag', rtol=3e-8 if mn == (5, 7) else None)
            q = Gbl/Gdl
            R.ratio(f"N = {N}: literal rotation (k = 1): G_bulk/G_bdy", Gbl, Gdl, 2.0, 10*tol)
            if mn == (1, 2) and ext == 'A' and geo == (1, 1, 0):
                R.paper(f"N = {N}: literal ratio (paper)", q, LITERAL[N], slack=3)
            R.ratio(f"N = {N}: bulk channel, Table I / literal = sqrt2", Gb, Gbl, np.sqrt(2), 1e-12)
            R.ratio(f"N = {N}: boundary channel, Table I / literal = 2 sqrt2", Gd, Gdl, 2*np.sqrt(2), 1e-12)
        R.note(f"[{time.time() - t0:.0f} s]")
    if A.full:
        for geo, pb, pd, tol in (((1, 1, 0), "1.126183453e-5", "1.126183807e-5", 1e-6),
                                 ((1, 1.3, 0), "5.824547027e-6", "5.823738591e-6", 3e-4)):
            R.section(f"c = 28 (c >= 25 data, P in iR), externals A, (a, y, x) = {geo}, N = 10")
            t0 = time.time()
            Gb, Gd = vvpsi_spacelike(Spacelike((1, 2)), EXT['A'], geo, [10])[10]
            R.agree("N = 10: bulk = boundary", Gb, Gd, tol, names=("G_bulk", "G_bdy"))
            R.paper("N = 10: G_bulk", Gb, pb)
            R.paper("N = 10: G_bdy", Gd, pd)
            R.note(f"[{time.time() - t0:.0f} s]")
    R.done()


if __name__ == "__main__":
    main()
