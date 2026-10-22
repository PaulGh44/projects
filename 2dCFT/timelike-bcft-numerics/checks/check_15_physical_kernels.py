"""
CHECK 15 - parity, and why the physical (even) kernels do not give a boundary sewing (main text; Sec. S7).
==========================================================================================================

Parity.  For f meromorphic near iR with Gaussian decay, int_{eps+iR} dP/i f = PV int dy f(iy) + pi sum(residues on iR):
for ODD f the principal value vanishes and the integral is a pure residue sum; for EVEN f the residues cancel and it is a
continuum (independent of eps).  The VWR kernels RF, RS are odd in their internal momenta, which is why every open-string
channel of Table I is a residue sum.

The physical c <= 1 kernel  Fhat = -i (Pt/Ps)(T1 + T2)  (the rotated sum of the two root terms; even, crosses the c <= 1
blocks) depends on the sheet of sqrt(Delta), and used with a reflection-even two-point function it does not give a
boundary sewing:
 1. bhat = 1, all momenta and labels i/4 (paper Eq. (S4)): the boundary four-point density is elementary,
       VWR:     sinh^2(pi P/2)/(8 sinh 2 pi P)  -> residue weights (0, 1/2, -1, 1/2), period four;
       physical kernel on a fixed sheet:  cosh^2(pi P/2)/(8 sinh 2 pi P) -> weights (1, -1/2, 0, -1/2).
    Crossing of the residue sums  J(T) = pi Res_0 + 2 pi sum_{n>0} Res_{P = in/2} e^{-pi T n^2/4}  is J(T) = T^{-1/2} J(1/T); it holds for the
    VWR weights (self-dual under the finite Fourier transform) and fails at order one for the fixed-sheet weights;
    an even density is a principal value, independent of eps;
 2. chat = -2 with the generic data of check_03 (u = (0.13,0.21,0.17,0.29), v = (0.19,0.21,0.24,0.28)): with the physical
    kernels on a sheet tracked along the contour ('sheet-fixed', no momentum factor) or flipped for Im P < 0
    ('reflection-even', Roussillon-Tsiares' physical kernel), the two channels of the boundary four-point function do not
    agree (paper: misses of 8-47% and 24-48%), and the reflection-even amplitudes change between eps = 0.15 and 0.3
    (paper: by a factor ~3); Route 2 (Table I) crosses to 1e-12 on the same grid (check_03).

Usage:  python checks/check_15_physical_kernels.py      (~15 s)
"""
import numpy as np
from scipy import integrate

from _common import Report, options
from tlbcft.theory import Timelike
from tlbcft.blocks import SphereBlock
from tlbcft.numerics import line, simpson


def right_bank(f, T, eps=0.13, L=8.0, N=200001):
    """int_{eps+iR} dP/i f(P) e^{pi T P^2}."""
    y = np.linspace(-L, L, N)
    P = eps + 1j*y
    return integrate.simpson(f(P)*np.exp(np.pi*T*P**2), x=y)


def residue_sum(res, T, nmax=80):
    """pi Res_0 + 2 pi sum_{n >= 1} Res_{in/2} e^{-pi T n^2/4}  (odd density: the poles at +-in/2 have equal residues)."""
    return np.pi*res(0) + 2*np.pi*sum(res(n)*np.exp(-np.pi*T*n*n/4) for n in range(1, nmax))


def track(sq):
    """continuous branch of sqrt(Delta) along the contour."""
    out = np.array(sq, dtype=complex)
    for j in range(1, len(out)):
        if abs(out[j] - out[j-1]) > abs(out[j] + out[j-1]):
            out[j] = -out[j]
    return out


def main():
    options(__doc__)
    R = Report(__file__, "parity on the physical contour; the physical kernels do not sew",
               "main text 'Parity and a discrete spectrum'; Sec. S4 'Parity and poles'; Sec. S7")
    kV = lambda P: np.sinh(np.pi*P/2)**2/(8*np.sinh(2*np.pi*P))
    kC = lambda P: np.cosh(np.pi*P/2)**2/(8*np.sinh(2*np.pi*P))
    kE = lambda P: np.sinh(np.pi*P/2)**2/(8*P*np.sinh(2*np.pi*P))
    resV = lambda n: np.sinh(1j*np.pi*n/4)**2/(16*np.pi*(-1)**n)
    resC = lambda n: np.cosh(1j*np.pi*n/4)**2/(16*np.pi*(-1)**n)

    R.section("1a. odd densities: the contour integral is the residue sum")
    for T in (0.5, 1.0, 2.0):
        R.agree(f"VWR density, T = {T}", right_bank(kV, T), residue_sum(resV, T), 1e-11, names=("contour", "residues"))
        R.agree(f"cosine (fixed-sheet physical) density, T = {T}", right_bank(kC, T), residue_sum(resC, T), 1e-11,
                names=("contour", "residues"))
    R.section("1b. even density: a principal value, independent of eps")
    ref = right_bank(kE, 1.0, eps=0.13)
    for eps in (0.05, 0.3):
        R.agree(f"eps = {eps} vs 0.13", right_bank(kE, 1.0, eps=eps), ref, 1e-11)
    R.section("1c. crossing of the residue sums, J(T) = T^{-1/2} J(1/T)")
    for T in (0.5, 0.7, 2.0, 3.0):
        R.agree(f"VWR weights, T = {T}", residue_sum(resV, T), residue_sum(resV, 1/T)/np.sqrt(T), 1e-12)
        d = abs(residue_sum(resC, T) - residue_sum(resC, 1/T)/np.sqrt(T))/abs(residue_sum(resC, T))
        R.true(f"fixed-sheet weights, T = {T}: crossing FAILS at order one", d > 0.05, f"relative defect {d:.2f}")
    dft = lambda w: [0.5*sum(w[r]*np.exp(2j*np.pi*m*r/4) for r in range(4)).real for m in range(4)]
    wV, wC = [0, 0.5, -1, 0.5], [1, -0.5, 0, -0.5]
    R.true("VWR weights (0, 1/2, -1, 1/2) are self-dual under the finite Fourier transform",
           np.allclose(dft(wV), wV), f"DFT = {np.round(dft(wV), 6)}")
    R.true("fixed-sheet weights (1, -1/2, 0, -1/2) are not", not np.allclose(dft(wC), wC), f"DFT = {np.round(dft(wC), 6)}")

    R.section("2. chat = -2, generic data: boundary crossing with the physical kernels (Route 1)")
    T = Timelike((1, 2))
    K = T.K
    u = np.array([0.13, 0.21, 0.17, 0.29])
    v = np.array([0.19, 0.21, 0.24, 0.28])
    u1, u2, u3, u4 = u
    v1, v2, v3, v4 = v
    h = T.hh(1j*u)
    bs = SphereBlock(T.chat, h[0], h[1], h[2], h[3], Nmax=12)
    bt = SphereBlock(T.chat, h[2], h[1], h[0], h[3], Nmax=12)
    # physical vertex kernels: -i (Pt/Ps) (T1 + T2) of the rotated arguments, as in RF (paper Eq. (2))
    args = {'A': lambda P: ((1j*P, -v2, -u1, -u2, -v3, -v1), P/(1j*v2)),
            'B': lambda P: ((-u4, -v3, 1j*P, -u3, -v4, -v1), (1j*u4)/(1j*v3)),
            'At': lambda P: ((1j*P, -v3, -u2, -u3, -v4, -v2), P/(1j*v3)),
            'Bt': lambda P: ((-u4, -v2, -u1, 1j*P, -v4, -v1), (1j*u4)/(1j*v2))}
    results = {}
    for eps in (0.15, 0.3):
        y, P = line(eps, 4.5, 1201)
        F = {}
        for key, af in args.items():
            a, pref = af(P)
            T1, T2, sq = K.root_terms(*a, return_sqrt=True)
            flip = track(sq) != sq                         # continuous sheet differs from the principal branch
            F[key] = -1j*pref*np.where(flip, -(T1 + T2), T1 + T2)
        sgn = np.where(y < 0, -1.0, 1.0)
        Hs, Ht = bs.H_series(T.hh(P)), bt.H_series(T.hh(P))
        for z in (0.5, 0.35):
            Vs, Vt = bs.value(z, T.hh(P), Hs), bt.value(1 - z, T.hh(P), Ht)
            for name, Ds, Dt in (("sheet-fixed", F['A']*F['B'], F['At']*F['Bt']),
                                 ("reflection-even", F['A']*F['B']*sgn, F['At']*F['Bt']*sgn)):
                Gs, Gt = simpson(Ds*Vs, y), simpson(Dt*Vt, y)
                results[(eps, z, name)] = Gs
                d = abs(Gs - Gt)/abs(Gs)
                R.true(f"eps = {eps}, z = {z}, {name}: G_s != G_t", d > 0.05, f"|G_s - G_t|/|G_s| = {d:.2f}")
    for z in (0.5, 0.35):
        ch = abs(results[(0.3, z, 'reflection-even')]/results[(0.15, z, 'reflection-even')])
        R.true(f"z = {z}: reflection-even amplitude depends on the contour", abs(ch - 1) > 0.5,
               f"|G(eps=0.3)/G(eps=0.15)| = {ch:.2f}")
    R.done()


if __name__ == "__main__":
    main()
