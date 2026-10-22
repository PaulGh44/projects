"""
CHECK 11 - irrational bhat: kernels from Barnes integrals (Sec. S8 "Kernels at arbitrary b"; Tables S-4pt, S-cyl, S-kac).
=========================================================================================================================

At irrational bhat^2 there is no closed form; F is computed from the Teschner-Vartanov Barnes integral.  On the physical
contour (P = i Phat, Phat in eps + iR) the pole families of the integrand interleave across the straight contour, and the
kernel is the line integral plus -+2 pi times the residues of the poles that crossed (tlbcft.kernels.GeneralFusionKernel;
validated against the closed form at bhat^2 = 5/7 in check_00).  Gamma_b is the integral representation (GeneralB).

What is verified at bhat^2 = 1/sqrt2 (chat = 0.272), where no Kac momenta coincide:
 1. the Kac cancellation in the s-channel of boundary crossing (Table S-kac, last row:
    6e-12, 2e-9, 1e-11, 1e-9, 2e-8, 4e-9 for (r,s) = (1,1),(2,1),(1,2),(3,1),(1,3),(2,2)), and at the golden-ratio point
    bhat^2 = (3 - sqrt5)/2 (chat = -5) for the lowest pairs;
 2. the residues of the Psi_0 trace density P K_ab(P) at the label-shifted lattice: e_a v_a/(4 pi) (multiplicity one at
    irrational bhat), and no residue at the Kac momenta;
 [--full]
 3. boundary crossing on eps + iR with the data of Table I (Table S-4pt, chat = 0.272, z = 0.5: 5.947008604731);
 4. the cylinder with a boundary insertion (Table S-cyl, chat = 0.272: Z(i)/i = 1.4795212693e-1, Z(1.3i)/i = 8.4748066682e-2).
 The Barnes-integral kernel is accurate to ~1e-10 (check_00, section 5).  Its error is common to both channels, which
 agree much better, but it shifts the values themselves; the printed digits of items 3-4 are compared at 3e-10 relative.

Runtime: each kernel takes ~0.5-1 s.  Default (items 1-2): ~3 min.  --full adds items 3-4 with 801-point grids:
about 3200 + 1600 kernel evaluations, ~40 min.

Usage:  python checks/check_11_irrational_b.py [--full]
"""
import time

import numpy as np

from _common import Report, options
from tlbcft.theory import Timelike
from tlbcft.blocks import SphereBlock
from tlbcft.numerics import circle_residue
from check_10_kac_cancellation import pi_rs, block_residue
from check_03_boundary_crossing import four_point
from check_04_cylinder import cylinder

BH2 = 2**-0.5


def kac_s_channel(R, T, pairs, zs=(0.3, 0.5), u=(0.13, 0.21, 0.17, 0.29), v=(0.19, 0.21, 0.24, 0.28)):
    K, bh, c = T.K, T.bh, T.chat
    u1, u2, u3, u4 = u
    v1, v2, v3, v4 = v
    hh = T.hh(1j*np.array(u))
    A = lambda P: K.F(1j*P, -v2, -u1, -u2, -v3, -v1)
    B = lambda P: K.F(-u4, -v3, 1j*P, -u3, -v4, -v1)
    blk = SphereBlock(c, hh[0], hh[1], hh[2], hh[3], Nmax=12)
    for (r, s) in pairs:
        t0 = time.time()
        Prs, Prms = T.Pdeg(r, s), T.Pdeg(r, -s)
        A1, B1, B2 = A(Prs), B(Prs), B(Prms)
        ResA = circle_residue(A, Prms, rad=0.01, npts=12)
        Vres = block_residue(blk, T.hh, Prs, zs, rad=0.01)
        R.agree(f"(r,s) = ({r},{s}): (b) B(P_(r,-s))/B(P_(r,s))", B2/B1,
                (-1)**(r*s)*pi_rs(r, s, -u4, -u3, bh)/pi_rs(r, s, -v3, -v1, bh), 1e-7)
        for z in zs:
            T1 = Prs*A1*B1*Vres[z]
            T2 = Prms*ResA*B2*blk.value(z, T.hh(Prms))
            r_ = abs(T1 + T2)/abs(T1)
            R.true(f"(r,s) = ({r},{s}), z = {z}: |T1+T2|/|T1|", r_ < 1e-5, f"{r_:.1e}")
        R.note(f"[{time.time() - t0:.0f} s]")


def main():
    A = options(__doc__)
    R = Report(__file__, "irrational bhat (Barnes-integral kernels)",
               "Sec. S8 'Kernels at arbitrary b'; Tables S-4pt, S-cyl, S-kac (bhat^2 = 1/sqrt2)")
    T = Timelike(BH2)
    R.note(f"bhat^2 = 1/sqrt2 = {BH2:.8f}, chat = {T.chat:.6f}")

    R.section("1. Kac cancellation, s-channel of boundary crossing, bhat^2 = 1/sqrt2")
    kac_s_channel(R, T, ((1, 1), (2, 1), (1, 2), (3, 1), (1, 3), (2, 2)))
    Tg = Timelike((3 - np.sqrt(5))/2)
    R.section(f"1b. Kac cancellation at the golden-ratio point bhat^2 = (3-sqrt5)/2, chat = {Tg.chat:.6f}")
    kac_s_channel(R, Tg, ((1, 1), (2, 1), (1, 2)))

    R.section("2. Psi_0 trace density P K_ab(P), K_ab = F_{iP, i s_a}[iP0 i s_a; iP i s_b] at P0 = 0hat, bhat^2 = 1/sqrt2")
    va, vb = 0.21, 0.085
    sa, sb = 1j*va, 1j*vb
    D = lambda P: P*T.K.F(1j*P, 1j*sa, 1j*P, 1j*T.zero, 1j*sa, 1j*sb)
    t0 = time.time()
    for (r, s) in ((1, 1), (3, 1), (1, 3)):
        for ea in (1, -1):
            for eb in (1, -1):
                Pst = 1j*(ea*va + eb*vb + 0.5*(r*T.bh + s/T.bh))
                res = circle_residue(D, Pst, rad=0.006, npts=8)
                R.agree(f"(r,s) = ({r},{s}), signs ({ea:+d},{eb:+d}): residue = e_a v_a/(4 pi)", res, ea*va/(4*np.pi), 1e-7)
    # no pole at the Kac momenta: the circle integral must vanish on the scale of the residues above (~1.7e-2);
    # the 8-point circle on Barnes-integral kernels leaves ~1e-9 there
    for (r, s) in ((1, 1), (2, 1), (1, 2)):
        Pk = T.Pdeg(r, -s)
        R.small(f"Kac momentum P_(r,-s), (r,s) = ({r},{s}): |residue|", circle_residue(D, Pk, rad=0.006, npts=8), 1e-8)
    R.note(f"[{time.time() - t0:.0f} s]")

    if A.full:
        R.section("3. boundary crossing at bhat^2 = 1/sqrt2 (801 points, eps = 0.15, level-12 blocks)")
        t0 = time.time()
        res = four_point(T, [0.5], npts=801)
        Gs, Gt, ends = res[0.5]
        R.agree(f"z = 0.5: crossing (|integrands| at the ends {ends:.0e})", Gs, Gt, 1e-10, names=("G_s", "G_t"))
        # the Barnes-integral kernel is accurate to ~1e-10 (check_00, section 5); the 13 printed digits are compared at
        # 3e-10 relative (the error is common to both channels, which agree far better)
        R.paper("z = 0.5: G_s", Gs, "5.947008604731", rtol=3e-10)
        R.note(f"[{time.time() - t0:.0f} s]")
        R.section("4. cylinder at bhat^2 = 1/sqrt2 (801 points, eps = 0.15, level-8 torus blocks)")
        t0 = time.time()
        res, _, _ = cylinder(T, (1j, 1.3j), npts=801)
        for tau, pr in zip((1j, 1.3j), ("1.4795212693e-1", "8.4748066682e-2")):
            Zo, Zc, ends = res[tau]
            R.agree(f"tau = {tau}: open = closed", Zo, Zc, 1e-9, names=("open", "closed"))
            R.paper(f"tau = {tau}: Z/i", Zo, pr, part='imag', rtol=3e-10)     # kernel accuracy, as in item 3
        R.note(f"[{time.time() - t0:.0f} s]")
    R.done()


if __name__ == "__main__":
    main()
