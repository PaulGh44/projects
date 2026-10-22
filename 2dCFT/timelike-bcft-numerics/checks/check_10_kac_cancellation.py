"""
CHECK 10 - the Kac cancellation: no open-string states at the Kac momenta (Sec. S4, Table S-kac).
=================================================================================================

An open-string density of boundary crossing is (internal-slot kernel A) x (external-slot kernel B) x (block).  Besides
the label-shifted lattice it has two families of simple poles that both multiply the block of the null descendant
V_z(P_(r,-s)) [P_(r,s) = (i/2)(r bhat - s/bhat), P_(r,-s) = (i/2)(r bhat + s/bhat)]:
    the block poles at P_(r,s):            T1 = P_(r,s) A(P_(r,s)) B(P_(r,s)) Res_{P_(r,s)} V_z,
    the poles of A at P_(r,-s):             T2 = P_(r,-s) Res_{P_(r,-s)} A  B(P_(r,-s)) V_z(P_(r,-s)).
The claim (proved in Sec. S4) is T1 + T2 = 0, so the open-string spectrum is exactly the label-shifted lattice.
A, B are the source kernels with the VWR ratios stripped (RF_A = (P/s2) A, RF_B = (P4/s3) B); normalizations play no role.
The three ingredients of the proof are checked separately:
  (a) Res_{Ps=P_(r,s)} F_{Ps,Pt}[...] = rho_rs/(-2 P_(r,s)) F_{P_(r,-s),Pt}[...]: the internal-slot residue of a kernel is
      the Zamolodchikov residue of the source block with the kernel's own external momenta, Eq. (S13);
  (b) the null-vector relation in the external slot: B(P_(r,-s))/B(P_(r,s)) = (-1)^{rs} pi_rs(Ps,P2)/pi_rs(Pt,P4),
      pi_rs the fusion polynomial;
  (c) R_rs(i bhat; Phat_i) = R_rs(bhat; i Phat_i)  (the timelike and source Zamolodchikov residues coincide).
Needs effectively irrational bhat: bhat^2 = 5/7 (first coincidence of Kac momenta at level 32), 5/8.

Channels: s- and t-channel of the boundary four-point function (externals u = (0.13,0.21,0.17,0.29), labels
v = (0.19,0.21,0.24,0.28)); the open channel of the cylinder (labels 0.21, 0.085, insertion P0 = 0.31i, torus blocks);
the boundary channel of the disc two-point function (P1,2 = 0.13i, 0.21i, label 0.1056i), where for r, s both odd the two
terms vanish separately ("0/0").  In the bulk channel of the disc there is no cancellation:
(T1+T2)/T1 = 1 + cos(2 pi v (r bhat + s/bhat))/cos(2 pi v (r bhat - s/bhat)) (closed-string states at the Kac momenta).

Expected (Table S-kac, |T1+T2|/|T1|, (r,s) = (1,1),(2,1),(1,2),(3,1),(1,3),(2,2)), all far below 1:
   5/7 s: 2e-11 1e-8 1e-10 1e-8 4e-8 2e-7      5/7 t: 1e-10 2e-7 2e-10 2e-8 7e-8 3e-8
   5/8 t: 3e-11 1e-7 2e-11 5e-10 5e-8 2e-8     5/7 cylinder: 5e-10 4e-8 2e-10 2e-6 1e-6 7e-6
   5/7 disc: 0/0 2e-8 2e-8 0/0 0/0 8e-7         (1/sqrt2, irrational: check_11_irrational_b.py)
Residues are circle integrals; level-12 blocks (level-8 torus blocks); the pass criterion is |T1+T2|/|T1| < 1e-5.

Usage:  python checks/check_10_kac_cancellation.py          (5/7 s-channel with ingredients, 5/7 cylinder; ~30 s)
        python checks/check_10_kac_cancellation.py --full   (all rows of Table S-kac except 1/sqrt2; ~2 min)
"""
import time

import numpy as np

from _common import Report, options
from tlbcft.theory import Timelike
from tlbcft.blocks import SphereBlock, TorusBlock
from tlbcft.numerics import circle_residue

PAIRS = ((1, 1), (2, 1), (1, 2), (3, 1), (1, 3), (2, 2))


def lam(p, q, b):
    return 0.5*(p*b + q/b)


def pi_rs(r, s, P, Pp, b):
    """fusion polynomial prod_{p,q} (P + P' - lam_pq)(P - P' - lam_pq), p in {1-r,...,r-1}, q in {1-s,...,s-1}."""
    val = 1.0 + 0j
    for p in range(1 - r, r, 2):
        for q in range(1 - s, s, 2):
            val *= (P + Pp - lam(p, q, b))*(P - Pp - lam(p, q, b))
    return val


def R_rs(r, s, b, P1, P2, P3, P4):
    """Zamolodchikov residue (up to a convention constant): A_rs pi_rs(P2, P1) pi_rs(P3, P4)."""
    A = 0.5 + 0j
    for k in range(1 - r, r + 1):
        for l in range(1 - s, s + 1):
            if (k, l) not in ((0, 0), (r, s)):
                A /= lam(k, l, b)
    return A*pi_rs(r, s, P2, P1, b)*pi_rs(r, s, P3, P4, b)


def block_residue(blk, h, z0, zvals, rad=0.02, npts=16):
    """Res_{P=z0} of the block V(P) (variable P, weight h(P)), for each z (real) in zvals."""
    th = 2*np.pi*(np.arange(npts) + 0.5)/npts
    pts = z0 + rad*np.exp(1j*th)
    H = blk.H_series(h(pts))
    return {z: rad*np.mean(blk.value(z, h(pts), H)*np.exp(1j*th)) for z in zvals}


def four_point(R, mn, channel, zs=(0.3, 0.5), ingredients=True, u=(0.13, 0.21, 0.17, 0.29), v=(0.19, 0.21, 0.24, 0.28)):
    T = Timelike(mn)
    K, bh, c = T.K, T.bh, T.chat
    u1, u2, u3, u4 = u
    v1, v2, v3, v4 = v
    hh = T.hh(1j*np.array(u))
    if channel == 's':
        A = lambda P: K.F(1j*P, -v2, -u1, -u2, -v3, -v1)
        B = lambda P: K.F(-u4, -v3, 1j*P, -u3, -v4, -v1)
        blk = SphereBlock(c, hh[0], hh[1], hh[2], hh[3], Nmax=12)
        A_ext, Bs = (-u1, -u2, -v3, -v1), dict(Ps=-u4, Pt=-v3, P2=-u3, P4=-v1)
        uz = (u1, u2, u3, u4)
    else:
        A = lambda P: K.F(1j*P, -v3, -u2, -u3, -v4, -v2)
        B = lambda P: K.F(-u4, -v2, -u1, 1j*P, -v4, -v1)
        blk = SphereBlock(c, hh[2], hh[1], hh[0], hh[3], Nmax=12)
        A_ext, Bs = (-u2, -u3, -v4, -v2), dict(Ps=-u4, Pt=-v2, P2=-u1, P4=-v4)
        uz = (u3, u2, u1, u4)
    zmap = (lambda z: z) if channel == 's' else (lambda z: 1 - z)
    Q = bh + 1/bh
    cs = 1 + 6*Q*Q
    hsrc = lambda P: Q*Q/4 - P**2
    blk_A = SphereBlock(cs, *[hsrc(p) for p in A_ext], Nmax=12)     # source block with the kernel A's externals
    for (r, s) in PAIRS:
        Prs, Prms = T.Pdeg(r, s), T.Pdeg(r, -s)
        t0 = time.time()
        A1, B1, B2 = A(Prs), B(Prs), B(Prms)
        ResA = circle_residue(A, Prms)
        zz = [zmap(z) for z in zs]
        Vres = block_residue(blk, T.hh, Prs, zz)
        V2 = {z: blk.value(z, T.hh(Prms)) for z in zz}
        for z in zz:
            T1 = Prs*A1*B1*Vres[z]
            T2 = Prms*ResA*B2*V2[z]
            r_ = abs(T1 + T2)/abs(T1)
            R.true(f"{mn[0]}/{mn[1]} {channel}-channel, (r,s) = ({r},{s}), z = {z:.2f}: |T1+T2|/|T1|", r_ < 1e-5,
                   f"{r_:.1e}   (T1 = {T1:.4e})")
        if ingredients:
            pred_b = (-1)**(r*s)*pi_rs(r, s, Bs['Ps'], Bs['P2'], bh)/pi_rs(r, s, Bs['Pt'], Bs['P4'], bh)
            R.agree(f"   (b) null-vector ratio B(P_(r,-s))/B(P_(r,s))", B2/B1, pred_b, 1e-7)
            Psrc, Psrc2 = 0.5*(r*bh + s/bh), 0.5*(r*bh - s/bh)
            Vs_res = block_residue(blk_A, hsrc, Psrc, (0.3,))[0.3]
            rho_A = Vs_res/blk_A.value(0.3, hsrc(Psrc2))
            R.agree(f"   (a) Res A / A(P_(r,s)) = -rho_rs/i (Zamolodchikov residue of the source block)", ResA/A1, -rho_A/1j, 1e-6)
            Rt = R_rs(r, s, 1j*bh, *[1j*x for x in uz])
            Rs = R_rs(r, s, bh, *[-x for x in uz])
            R.agree(f"   (c) R_rs(i bhat; Phat) = R_rs(bhat; i Phat)", Rt, Rs, 1e-12)
        R.note(f"[{time.time() - t0:.0f} s]")


def cylinder(R, mn, va=0.21, vb=0.085, u0=0.31, taus=(1j, 1.3j)):
    """open channel of the cylinder: trace density P K(P) V_tau(P; P0), K = F_{iP,i sa}[iP0 i sa; iP i sb]."""
    T = Timelike(mn)
    K, c = T.K, T.chat
    sa, sb, P0 = 1j*va, 1j*vb, 1j*u0
    Kab = lambda P: K.F(1j*P, 1j*sa, 1j*P, 1j*P0, 1j*sa, 1j*sb)
    TB = TorusBlock(c, P0, Nmax=8)
    for (r, s) in PAIRS:
        Prs, Prms = T.Pdeg(r, s), T.Pdeg(r, -s)
        K1 = Kab(Prs)
        ResK = circle_residue(Kab, Prms, rad=0.01)
        for tau in taus:
            if tau != taus[0] and r*s > 3:
                continue        # at tau = 1.3i the level-4 residue (~q^4) is below the truncation error of the blocks
            Vres = circle_residue(lambda P: TB.value(tau, P), Prs, rad=0.01, npts=12)
            V2 = TB.value(tau, Prms)
            T1, T2 = Prs*K1*Vres, Prms*ResK*V2
            r_ = abs(T1 + T2)/abs(T1)
            R.true(f"{mn[0]}/{mn[1]} cylinder, (r,s) = ({r},{s}), tau = {tau}: |T1+T2|/|T1|", r_ < 3e-5,
                   f"{r_:.1e}   (T1 = {T1:.4e})")


def disc(R, mn, u=(0.13, 0.21), v=0.1056, zs=(0.3, 0.5)):
    """both channels of the disc two-point function (data of Table I up to normalization)."""
    T = Timelike(mn)
    bh, c = T.bh, T.chat
    P1, P2, s = 1j*u[0], 1j*u[1], 1j*v
    Db = lambda P: T.rhoh(P)*T.Ch(P, P1, P2)*T.Rh1(P, s)
    Dt = lambda P: T.Rh2(P1, P, s)*T.Rh2(P2, P, s)/T.Sh(P, s, s)
    h1, h2 = T.hh(P1), T.hh(P2)
    bs = SphereBlock(c, h2, h1, h1, h2, Nmax=12)
    bt = SphereBlock(c, h1, h1, h2, h2, Nmax=12)
    for (r, s_) in PAIRS:
        Prs, Prms = T.Pdeg(r, s_), T.Pdeg(r, -s_)
        pred_bulk = 1 + np.cos(2*np.pi*v*(r*bh + s_/bh))/np.cos(2*np.pi*v*(r*bh - s_/bh))
        both_odd = (r % 2 == 1 and s_ % 2 == 1)
        for (name, D, blk, zmap) in (("boundary", Dt, bt, lambda z: 1 - z), ("bulk", Db, bs, lambda z: z)):
            th8 = 2*np.pi*(np.arange(8) + 0.5)/8
            D1 = np.mean(D(Prs + 0.002*np.exp(1j*th8)))        # value at P_(r,s): mean over a tiny circle
            ResD = circle_residue(D, Prms, rad=0.01)
            zz = [zmap(z) for z in zs]
            Vres = block_residue(blk, T.hh, Prs, zz, rad=0.01)
            for z in zz:
                T1, T2 = D1*Vres[z], ResD*blk.value(z, T.hh(Prms))
                if name == "boundary" and both_odd:
                    R.true(f"{mn[0]}/{mn[1]} disc boundary channel, (r,s) = ({r},{s_}), z = {z:.2f}: 0/0 (both terms vanish)",
                           abs(T1) < 1e-8 and abs(T2) < 1e-8, f"|T1| = {abs(T1):.1e}, |T2| = {abs(T2):.1e}")
                elif name == "boundary":
                    r_ = abs(T1 + T2)/abs(T1)
                    R.true(f"{mn[0]}/{mn[1]} disc boundary channel, (r,s) = ({r},{s_}), z = {z:.2f}: |T1+T2|/|T1|", r_ < 1e-5,
                           f"{r_:.1e}")
                else:
                    R.agree(f"{mn[0]}/{mn[1]} disc bulk channel (no cancellation), (r,s) = ({r},{s_}), z = {z:.2f}: (T1+T2)/T1",
                            (T1 + T2)/T1, pred_bulk, 1e-5, names=("computed", "1 + cos/cos"))


def main():
    A = options(__doc__)
    R = Report(__file__, "Kac cancellation: T1 + T2 = 0 in every open-string channel",
               "'Parity and a discrete spectrum'; Sec. S4 'Kac cancellation', Eq. (S13); Table S-kac")
    t0 = time.time()
    R.section("boundary four-point function, s-channel, bhat^2 = 5/7, with the ingredients (a)-(c)")
    four_point(R, (5, 7), 's')
    R.section("cylinder, open channel, bhat^2 = 5/7")
    cylinder(R, (5, 7))
    if A.full:
        R.section("boundary four-point function, t-channel, bhat^2 = 5/7")
        four_point(R, (5, 7), 't', ingredients=False)
        R.section("boundary four-point function, t-channel, bhat^2 = 5/8")
        four_point(R, (5, 8), 't', ingredients=False)
        R.section("disc two-point function, both channels, bhat^2 = 5/7")
        disc(R, (5, 7))
    R.note(f"[{time.time() - t0:.0f} s]")
    R.done()


if __name__ == "__main__":
    main()
