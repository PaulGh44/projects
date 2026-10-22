"""
CHECK 08 - the annulus and Cardy's condition (paper "Annulus and Cardy's condition", Sec. S4, Table II row "annulus").
=====================================================================================================================

Two boundary states s_a = i v_a, s_b = i v_b.  The closed channel (paper Eq. (4), last line) is

   Z_ab = int_{eps+iR} dP/i rhohat(P) Rhat(P|s_a) Rhat(P|s_b) chi_P(qt),     chi_P(q) = q^{-P^2}/eta(q),

with rhohat Rhat Rhat = sqrt2 cos(4 pi s_a P) cos(4 pi s_b P)/(sinh(2 pi bhat P) sinh(2 pi P/bhat)) in the normalization
of Table I.  The density is EVEN in P with a double pole at P = 0 without residue: the integral is a finite part and does
not depend on eps.  With tau = iT, q = e^{-2 pi T}, qt = e^{-2 pi/T}, the Abel-summed Fourier expansion
   FP 1/(sin theta sin phi) = -4 sum_{r,s odd} cos(4 pi p_{r,-s} y),   p_{r,-s} = (r bhat + s/bhat)/2,   Eq. (S21)
and int dy cos(4 pi k y) e^{-2 pi y^2/T} = sqrt(T/2) q^{k^2} give the open channel

   Z_ab = sum_{r,s odd} sum_{+-,+-'} chi_{P(r,-s) +- (s_a +-' s_b)}(q),    i.e. weights h = (chat-1)/24 + k^2,
          k = (r bhat + s/bhat)/2 +- (v_a +-' v_b),

one Verma character per point of the label-shifted odd-odd Kac lattice (Eq. (5)) with coefficient +1: the annulus is a
positive INTEGER combination of characters, with multiplicity m_{r,s} where several lattice points share a weight.
The literal rotation (k = 1) would give 1/2 per point.

What is verified:
 1. closed channel (line integral on eps + iR, three values of eps) = open channel (lattice sum), for bhat^2 = 1, 1/2,
    2/3, 5/7, 0.49, 1/sqrt2 and several labels, T = 0.7, 1.3 (paper: to 1e-16; here to double precision ~1e-13);
 2. positivity and integrality of the coefficients on the first 200 weights, four bhat, six label pairs;
 3. the quantized examples: chat = 1, v = 1/4: Z = sum_n n chi_{n^2/4};  chat = -2, v = bhat/4:
    Z = chi_{3/8} + 2 chi_1 + 2 chi_{15/8} + 2 chi_3 + 3 chi_{35/8} + 4 chi_6 + ...;
 4. the comparison annuli of Eq. (S22) with the normalization-invariant kernels U_a(P) U_b(-P):
       Bautista-Bawane  sin sin/(sinh sinh): same support, sign -1 for the shifts +-(v_a + v_b) (not a trace);
       Giribet-Sivilotti  cos cos sinh(2 pi P/bhat)/sinh(2 pi bhat P): support (r bhat +- 1/bhat)/2 +- (v_a +- v_b);
       a positive combination of degenerate characters chi^deg_(r,1) only at quantized labels s = i k bhat/4.

Usage:  python checks/check_08_annulus.py      (~10 s)
"""
from collections import defaultdict

import numpy as np
from scipy import integrate

from _common import Report, options


def kernel(kind, bh, va, vb, P):
    """Closed-channel densities (normalization of Table I for 'R2'; normalization-invariant U U for 'BB', 'GS')."""
    ca, cb = np.cos(4*np.pi*(1j*va)*P), np.cos(4*np.pi*(1j*vb)*P)
    sa, sb = np.sin(4*np.pi*(1j*va)*P), np.sin(4*np.pi*(1j*vb)*P)
    sh = np.sinh(2*np.pi*bh*P)*np.sinh(2*np.pi*P/bh)
    if kind == 'R2':
        return np.sqrt(2)*ca*cb/sh
    if kind == 'R2literal':
        return ca*cb/(np.sqrt(2)*sh)
    if kind == 'BB':
        return np.sqrt(2)*sa*sb/sh
    if kind == 'GS':
        return np.sqrt(2)*np.sinh(2*np.pi*P/bh)/np.sinh(2*np.pi*bh*P)*ca*cb
    raise ValueError(kind)


def Z_closed(kind, bh, va, vb, T, eps=0.11, L=10.0, N=400001):
    """eta(q) * Z = T^{-1/2} int dy density(eps + i y) e^{2 pi P^2/T}   (chi_P(qt) = qt^{-P^2}/eta(qt), eta(qt) = sqrt(T) eta(q))."""
    y = np.linspace(-L, L, N)
    P = eps + 1j*y
    return integrate.simpson(kernel(kind, bh, va, vb, P)*np.exp(2*np.pi*P**2/T), x=y)/np.sqrt(T)


def spectrum(kind, bh, va, vb, R=60):
    """Open channel as a list (k, coefficient): eta(q) Z = sum c q^{k^2}."""
    p = lambda r, s: 0.5*(r*bh - s/bh)
    out = []
    if kind in ('R2', 'R2literal', 'BB'):
        coef = {'R2': 1.0, 'R2literal': 0.5, 'BB': None}[kind]
        for r in range(1, R, 2):
            for s in range(1, R, 2):
                for d, sd in ((va - vb, +1), (va + vb, -1)):
                    for pm in (+1, -1):
                        out.append((p(r, -s) + pm*d, coef if coef is not None else sd*1.0))
    else:   # GS: sin(phi)/sin(theta) = sum_{r odd} [cos(4 pi p_{r,1} y) - cos(4 pi p_{r,-1} y)]
        for r in range(1, R, 2):
            for mom, sg in ((p(r, 1), +1), (p(r, -1), -1)):
                for d in (va - vb, va + vb):
                    for pm in (+1, -1):
                        out.append((mom + pm*d, sg*0.25))
    return out


def Z_open(kind, bh, va, vb, T):
    q = np.exp(-2*np.pi*T)
    return sum(c*q**(k*k) for k, c in spectrum(kind, bh, va, vb))


def content(kind, bh, va, vb, R=60, tol=1e-9):
    """Verma content: {k^2: total coefficient} (coefficients of coinciding weights added)."""
    acc = defaultdict(float)
    for k, c in spectrum(kind, bh, va, vb, R):
        acc[round(k*k, 9)] += c
    return sorted((w, m) for w, m in acc.items() if abs(m) > tol)


def degenerate_decomposition(bh, v, rmax=20):
    """Write the GS self-annulus (labels v_a = v_b = v) as sum_r n_r chi^deg_(r,1) + remainder, where
    chi^deg_(r,1) = q^{p(r,1)^2} - q^{p(r,-1)^2}, p(r,s) = (r bhat - s/bhat)/2.  Returns ({r: n_r}, |remainder|) over the
    weights below p(rmax/2, -1)^2."""
    acc = defaultdict(float)
    for k, c in spectrum('GS', bh, v, v, R=2*rmax + 1):
        acc[round(k*k, 9)] += c
    p = lambda r, s: 0.5*(r*bh - s/bh)
    n = {}
    for r in range(1, rmax + 1):
        w1, w2 = round(p(r, 1)**2, 9), round(p(r, -1)**2, 9)
        n[r] = acc.get(w1, 0.0)
        acc[w1] = acc.get(w1, 0.0) - n[r]
        acc[w2] = acc.get(w2, 0.0) + n[r]
    wmax = p(rmax//2, -1)**2
    resid = max([abs(m) for w, m in acc.items() if w < wmax] + [0.0])
    return n, resid


def main():
    options(__doc__)
    R = Report(__file__, "annulus: closed channel = positive integer sum of characters (Cardy)",
               "Eq. (4) last line; 'Annulus and Cardy's condition'; Sec. S4 Eqs. (S21)-(S22); Table II row 'annulus'")
    R.section("1. closed channel (finite part on eps + iR) = open channel (lattice sum), Table I normalization")
    cases = [(1.0, 0.19, 0.24, "chat = 1"), (1/np.sqrt(2), 0.19, 0.24, "chat = -2"), (np.sqrt(2/3), 0.19, 0.24, "chat = 0"),
             (np.sqrt(5/7), 0.13, 0.21, "bhat^2 = 5/7"), (0.7, 0.13, 0.21, "bhat^2 = 0.49"),
             (2**-0.25, 0.19, 0.24, "bhat^2 = 1/sqrt2"), (1/np.sqrt(2), 1/(4*np.sqrt(2)), 1/(4*np.sqrt(2)), "chat = -2, v = bhat/4")]
    for (bh, va, vb, lab) in cases:
        for T in (0.7, 1.3):
            Zo = Z_open('R2', bh, va, vb, T)
            for eps in (0.05, 0.11, 0.25):
                R.agree(f"{lab}, v = ({va:.4f}, {vb:.4f}), T = {T}, eps = {eps}", Z_closed('R2', bh, va, vb, T, eps=eps), Zo,
                        1e-12, names=("closed", "open"))
    R.section("2. positivity and integrality of the open-channel coefficients (first 200 weights)")
    for (bh, lab) in [(1.0, "chat = 1"), (1/np.sqrt(2), "chat = -2"), (np.sqrt(2/3), "chat = 0"), (0.7, "bhat = 0.7")]:
        for (va, vb) in [(0.19, 0.24), (0.13, 0.21), (bh/4, bh/4), (bh/2, bh/2), (bh/4, bh/2), (0.31, 0.05)]:
            cc = [m for w, m in content('R2', bh, va, vb)[:200]]
            ok = min(cc) > 0 and all(abs(m - round(m)) < 1e-9 for m in cc)
            R.true(f"{lab}, v = ({va:.3f}, {vb:.3f}): coefficients positive integers", ok,
                   f"min {min(cc):.0f}, max {max(cc):.0f}")
    R.section("3. quantized labels")
    cont = content('R2', 1.0, 0.25, 0.25)
    expected = [(n*n/4, n) for n in range(1, 13)]
    got = [(w, m) for w, m in cont if w < 36.5]
    R.true("chat = 1, v = 1/4: Z = sum_n n chi_{n^2/4} (h = n^2/4, n = 1..12)",
           all(abs(w - we) < 1e-9 and abs(m - me) < 1e-9 for (w, m), (we, me) in zip(got, expected)),
           "  ".join(f"{m:.0f}*chi_{w:g}" for w, m in got[:6]) + " ...")
    bh = 1/np.sqrt(2)
    e = (1 - 6*(bh - 1/bh)**2 - 1)/24
    cont = content('R2', bh, bh/4, bh/4)
    hs = [(round(e + w, 6), m) for w, m in cont[:6]]
    exp = [(3/8, 1), (1.0, 2), (15/8, 2), (3.0, 2), (35/8, 3), (6.0, 4)]
    R.true("chat = -2, v = bhat/4: Z = chi_{3/8} + 2 chi_1 + 2 chi_{15/8} + 2 chi_3 + 3 chi_{35/8} + 4 chi_6 + ...",
           all(abs(a - c) < 1e-6 and abs(b - d) < 1e-9 for (a, b), (c, d) in zip(hs, exp)),
           "  ".join(f"{m:.0f}*chi_{h:g}" for h, m in hs))
    R.section("4. literal rotation: one half per lattice point")
    for T in (0.7, 1.3):
        R.agree(f"chat = -2, literal normalization, T = {T}: closed = (1/2) sum chi", Z_closed('R2literal', bh, 0.19, 0.24, T),
                Z_open('R2literal', bh, 0.19, 0.24, T), 1e-12, names=("closed", "open"))
        R.ratio(f"   Table I / literal = k^2 = 2", Z_closed('R2', bh, 0.19, 0.24, T), Z_closed('R2literal', bh, 0.19, 0.24, T),
                2.0, 1e-13)
    R.section("5. comparison annuli: Bautista-Bawane and Giribet-Sivilotti one-point functions, Eq. (S22)")
    for kind in ('BB', 'GS'):
        for (bh_, va, vb, lab) in [(1/np.sqrt(2), 0.19, 0.24, "chat = -2"), (0.7, 0.13, 0.21, "bhat = 0.7"),
                                   (np.sqrt(2/3), np.sqrt(2/3)/4, np.sqrt(2/3)/4, "chat = 0, v = bhat/4")]:
            for T in (0.7, 1.3):
                R.agree(f"{kind}, {lab}, T = {T}", Z_closed(kind, bh_, va, vb, T), Z_open(kind, bh_, va, vb, T), 1e-12,
                        names=("closed", "open"))
    cc = [m for w, m in content('BB', 1/np.sqrt(2), 0.19, 0.24)[:40]]
    R.true("BB at generic labels has negative coefficients (shifts +-(v_a+v_b)): not a trace", min(cc) < 0,
           f"min coefficient {min(cc):+.0f}")
    # GS in terms of degenerate characters chi^deg_(r,1) = chi_{P(r,1)} - chi_{P(r,-1)} (null vector at level r)
    bh_ = 0.7
    for (va, lab, quantized) in ((0.13, "generic label 0.13i", False), (bh_/4, "quantized label i bhat/4", True),
                                 (3*bh_/4, "quantized label 3i bhat/4", True)):
        n, resid = degenerate_decomposition(bh_, va)
        ok = (resid < 1e-9 and min(n.values()) >= -1e-12) if quantized else (resid > 1e-3)
        R.true(f"GS, bhat = 0.7, {lab}: positive combination of degenerate characters = {quantized}", ok,
               f"coefficients of chi^deg_(r,1), r = 1..6: {[round(n[r], 3) for r in range(1, 7)]}, residual {resid:.1e}")
    R.done()


if __name__ == "__main__":
    main()
