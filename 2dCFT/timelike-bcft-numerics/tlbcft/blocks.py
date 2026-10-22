r"""
Virasoro conformal blocks at arbitrary central charge (c <= 1 and c >= 25 alike).
================================================================================

Sphere four-point blocks (paper Sec. S1, "Blocks and kernels")
---------------------------------------------------------------
  V_z(h; h1, h2, h3, h4) = z^{h - h1 - h2} (1 + O(z)),   operators h1, h2, h3, h4 at 0, z, 1, infinity.

  Series in z from the Verma module of weight h (c generic):
     V_z = z^{h-h1-h2} sum_N z^N sum_{|lam|=|mu|=N} gamma_lam(h; h4, h3) (G_N^{-1})_{lam mu} gamma_mu(h; h1, h2),
     gamma_lam(h; ha, hb) = prod_i ( h + sum_{l>i} lam_l + lam_i hb - ha ),   G_N the Gram matrix at level N
     in the basis L_{-lam_1} ... L_{-lam_k}|h>,  lam_1 >= lam_2 >= ...

  Elliptic (Zamolodchikov) form, used for evaluation because it converges much faster:
     V_z = (16 q)^{h - e} z^{e - h1 - h2} (1 - z)^{e - h2 - h3} theta3(q)^{(c-1)/2 - 4(h1+h2+h3+h4)} H(q),
     e = (c-1)/24,  q = e^{i pi tau},  z = theta2(q)^4/theta3(q)^4.
  H(q) = 1 + O(q) is obtained as a truncated q-series by re-expanding the z-series ("level Nmax" = order in q).

  Everything is vectorised over the internal weight h: SphereBlock(c, h1..h4, Nmax).H_series(h_array) builds
  the Gram matrices once for all weights (class VermaModule), which is ~100x faster than one weight at a time.

Torus one-point blocks
----------------------
  V_tau(h; h0) = q^{h - c/24} (1 + O(q)),  q = e^{2 pi i tau}.  Two implementations:
   * TorusBlock: brute force from the Verma module (Ward identity for <lam| phi_0(1) |mu>), vectorised over h;
   * poghossian_values: the Poghossian / Fateev-Litvinov-Neveu-Onofri / Hadasz-Jaskolski-Suchanek relation
         prod_n (1 - q^n) q^{-(h - (c-1)/24)} V_tau(P; P0) = H^{sphere}_{b', sqrt2 P}[P0/sqrt2, b'/4, b'/4, b'/4](q'),
     b'^2 = 2 b^2, q = q'^2, used to cross-check TorusBlock in the self-test.

Exact blocks used as references
-------------------------------
  * c = 1, all external weights 1/16:  V_z(P) = 2^{-4P^2} e^{-i pi P^2 tau} / ( [z(1-z)]^{1/8} theta3(q) ),
    with weights h = -P^2 (paper Sec. S1, below Eq. (S4)).

Self-test:  python -m tlbcft.blocks
"""
from functools import lru_cache
from math import factorial

import numpy as np
from scipy.special import ellipk


# ---------------------------------------------------------------------------------------------
# partitions and normal ordering of Virasoro creation operators
# ---------------------------------------------------------------------------------------------
def partitions(n, maxpart=None):
    """Partitions of n as non-increasing tuples."""
    if maxpart is None:
        maxpart = n
    if n == 0:
        yield ()
        return
    for k in range(min(n, maxpart), 0, -1):
        for rest in partitions(n - k, k):
            yield (k,) + rest


@lru_cache(maxsize=None)
def parts_of(n):
    return tuple(partitions(n))


@lru_cache(maxsize=None)
def insert(m, mu):
    """L_{-m} L_{-mu_1} ... L_{-mu_k} |h>  re-ordered into the PBW basis:  dict {partition: integer coefficient}."""
    if not mu or m >= mu[0]:
        return {(m,) + mu: 1}
    out = {}
    for part, cf in insert(m, mu[1:]).items():
        for part2, cf2 in insert(mu[0], part).items():
            out[part2] = out.get(part2, 0) + cf*cf2
    for part, cf in insert(m + mu[0], mu[1:]).items():          # [L_{-m}, L_{-mu_1}] = (mu_1 - m) L_{-m-mu_1}
        out[part] = out.get(part, 0) + (mu[0] - m)*cf
    return out


class VermaModule:
    """Verma module(s) of central charge c and weights h (array of shape (K,)), levels <= Nmax.

    apply_L(n, mu): L_n L_{-mu}|h> (n >= 1) as a dict {partition: coefficient array of shape (K,)}.
    G[N]:           Gram matrices, shape (K, p(N), p(N)).
    solve(N, rhs):  G[N]^{-1} rhs by an equilibrated linear solve (rhs of shape (K, p(N)) or (K, p(N), m)).
                    The Gram matrices are badly conditioned (entries ~ h^N); solving instead of inverting, after
                    scaling rows and columns by 1/sqrt|G_ii| (by 1/sqrt(max_j |G_ij|) where an accidental G_ii = 0),
                    gains 4-6 digits in the blocks at large nome (|q| = e^{-pi/2}, check_00 section 6) and a few
                    digits at large |h| everywhere.
    Ginv[N]:        explicit inverses (computed on first use; used by the five-point comb blocks).
    """
    def __init__(self, c, h, Nmax, gram=True, symmetric_fill=False):
        self.c = c
        self.h = np.atleast_1d(np.asarray(h, dtype=complex))
        self.K = self.h.size
        self.Nmax = Nmax
        self._cache = {}
        self.G = {}
        self._Ginv = {}
        if gram:
            for N in range(Nmax + 1):
                P = parts_of(N)
                Gm = np.zeros((self.K, len(P), len(P)), dtype=complex)
                for i, lam in enumerate(P):
                    for j, mu in enumerate(P):
                        if symmetric_fill and j < i:      # G is symmetric; copying halves the cost
                            Gm[:, i, j] = Gm[:, j, i]
                        else:
                            Gm[:, i, j] = self.inner(lam, mu)
                self.G[N] = Gm

    @property
    def Ginv(self):
        for N in self.G:
            if N not in self._Ginv:
                self._Ginv[N] = np.linalg.inv(self.G[N]) if N > 0 else np.ones((self.K, 1, 1), dtype=complex)
        return self._Ginv

    def solve(self, N, rhs):
        """G[N]^{-1} rhs (equilibrated solve)."""
        rhs = np.asarray(rhs, dtype=complex)
        vec = (rhs.ndim == 2)
        r = rhs[..., None] if vec else rhs
        diag = np.abs(np.einsum('kii->ki', self.G[N]))
        rowmax = np.max(np.abs(self.G[N]), axis=2)
        d = 1/np.sqrt(np.where(diag > 1e-12*rowmax, diag, rowmax))      # symmetric diagonal scaling (row max if G_ii ~ 0)
        Gs = self.G[N]*d[:, :, None]*d[:, None, :]
        x = np.linalg.solve(Gs, r*d[:, :, None])*d[:, :, None]
        return x[..., 0] if vec else x

    def apply_L(self, n, mu):
        key = (n, mu)
        if key in self._cache:
            return self._cache[key]
        out = {}
        if mu:
            mu1, rest = mu[0], mu[1:]
            k = n - mu1
            coef = n + mu1
            if k > 0:
                for p, cf in self.apply_L(k, rest).items():
                    out[p] = out.get(p, 0) + coef*cf
            elif k == 0:
                out[rest] = out.get(rest, 0) + coef*(self.h + sum(rest)) + self.c/12*n*(n*n - 1)
            else:
                for p, cf in insert(-k, rest).items():
                    out[p] = out.get(p, 0) + coef*cf
            for p, cf in self.apply_L(n, rest).items():
                for p2, cf2 in insert(mu1, p).items():
                    out[p2] = out.get(p2, 0) + cf*cf2
        self._cache[key] = out
        return out

    def inner(self, lam, mu):
        """<h| L_{lam_k} ... L_{lam_1} L_{-mu_1} ... L_{-mu_l} |h>  (array over the K weights)."""
        state = {mu: np.ones(self.K, dtype=complex)}
        for n in lam:
            new = {}
            for p, cf in state.items():
                for p2, cf2 in self.apply_L(n, p).items():
                    new[p2] = new.get(p2, 0) + cf*cf2
            state = new
        return state.get((), 0.0)*np.ones(self.K, dtype=complex)


def gamma_vertex(lam, h, ha, hb):
    """<h; lam| V_hb(1) |ha> / <h|V_hb(1)|ha> = prod_i (h + sum_{l>i} lam_l + lam_i hb - ha)."""
    val = 1.0 + 0j
    tail = 0
    for i in range(len(lam) - 1, -1, -1):
        val = val*(h + tail + lam[i]*hb - ha)
        tail += lam[i]
    return val


# ---------------------------------------------------------------------------------------------
# truncated q-series arithmetic (rows = different weights)
# ---------------------------------------------------------------------------------------------
def ser_mul(a, b):
    """Product of truncated series (length of a); a, b 1-d, or 2-d arrays whose rows are multiplied pairwise."""
    a = np.asarray(a, dtype=complex); b = np.asarray(b, dtype=complex)
    n = a.shape[-1]
    if a.ndim == 1 and b.ndim == 1:
        return np.convolve(a, b)[:n]
    a2, b2 = np.broadcast_arrays(np.atleast_2d(a), np.atleast_2d(b))
    return np.array([np.convolve(x, y)[:n] for x, y in zip(a2, b2)])


def ser_pow(s, alpha):
    """s(q)^alpha for a series with s[0] = 1 (J.C.P. Miller recursion); alpha may be an array -> rows."""
    s = np.asarray(s, dtype=complex)
    alpha = np.asarray(alpha, dtype=complex)
    n = s.shape[-1]
    p = np.zeros(alpha.shape + (n,), dtype=complex)
    p[..., 0] = 1.0
    for k in range(1, n):
        acc = 0
        for j in range(1, k + 1):
            acc = acc + (alpha*j - (k - j))*s[j]*p[..., k - j]
        p[..., k] = acc/k
    return p


def theta_series(n):
    th3 = np.zeros(n, dtype=complex); th4 = np.zeros(n, dtype=complex); t2 = np.zeros(n, dtype=complex)
    th3[0] = 1; th4[0] = 1
    for m in range(1, n):
        if m*m < n:
            th3[m*m] += 2
            th4[m*m] += 2*(-1)**m
    for m in range(0, n):
        if m*(m + 1) < n:
            t2[m*(m + 1)] += 1                       # theta2 = 2 q^{1/4} t2(q)
    return th3, th4, t2


def z_series(n):
    """z/(16 q) and 1 - z as q-series (z = theta2^4/theta3^4), and theta3."""
    th3, th4, t2 = theta_series(n)
    ith3_4 = ser_pow(ser_pow(th3, 4), -1)
    return ser_mul(ser_pow(t2, 4), ith3_4), ser_mul(ser_pow(th4, 4), ith3_4), th3


def elliptic_nome(z):
    """q = e^{i pi tau} with z = lambda(tau), for real z in (0, 1)."""
    z = complex(z)
    if abs(z.imag) < 1e-15 and 0 < z.real < 1:
        return np.exp(-np.pi*ellipk(1 - z.real)/ellipk(z.real))
    raise ValueError("elliptic_nome: real z in (0,1) only; use SphereBlock.at_tau for other points")


def theta_logs(tau, nterms=60):
    """q = e^{i pi tau} and log theta2, log theta3, log theta4 from the product formulas (holomorphic in tau)."""
    q = np.exp(1j*np.pi*tau)
    n = np.arange(1, nterms + 1)
    q2n, q2n1 = q**(2*n), q**(2*n - 1)
    l1q = np.sum(np.log(1 - q2n))
    lth2 = np.log(2) + 1j*np.pi*tau/4 + l1q + 2*np.sum(np.log(1 + q2n))
    lth3 = l1q + 2*np.sum(np.log(1 + q2n1))
    lth4 = l1q + 2*np.sum(np.log(1 - q2n1))
    return q, lth2, lth3, lth4


# ---------------------------------------------------------------------------------------------
# sphere four-point blocks
# ---------------------------------------------------------------------------------------------
class SphereBlock:
    """Sphere four-point blocks at fixed (c; h1, h2, h3, h4), vectorised over the internal weight h.

    H_series(h)      -> array (K, Nmax+1): coefficients of H(q) for each weight in h
    value(z, h, H)   -> V_z(h) for real z in (0, 1)
    at_tau(tau, h, H)-> V at a point tau of the upper half plane, continued holomorphically from tau = i oo
                        (used for the disc with one bulk and two boundary fields, where z = lambda(tau) = 2 +- i0).
    gram = 'solve' (default) or 'inverse': how G[N]^{-1} is applied.  'inverse' (explicit inverse, as in the project's
    original vblock.py) is kept only so that check_00 can show the loss of accuracy it causes.
    """
    def __init__(self, c, h1, h2, h3, h4, Nmax=12, chunk=400, gram='solve'):
        self.c, self.hs, self.Nmax, self.chunk = c, (h1, h2, h3, h4), Nmax, chunk
        self.gram = gram
        self.z16, self.omz, self.th3 = z_series(Nmax + 1)
        e = (c - 1)/24
        n = Nmax + 1
        # h-independent factors of H(q)
        self.fixed = ser_mul(ser_pow(self.omz, -(e - h2 - h3)), ser_pow(self.th3, -((c - 1)/2 - 4*(h1 + h2 + h3 + h4))))
        # (q z16)^N as shifted series (the factor 16^N is applied to the coefficients a_N)
        self.zpows = []
        zp = np.zeros(n, dtype=complex); zp[0] = 1
        for N in range(n):
            self.zpows.append(np.concatenate([np.zeros(N), zp[:n - N]]))
            zp = ser_mul(zp, self.z16)

    def z_coeffs(self, h):
        """a_N(h), N = 0..Nmax, of V_z = z^{h-h1-h2} sum_N a_N z^N  (shape (K, Nmax+1))."""
        h1, h2, h3, h4 = self.hs
        h = np.atleast_1d(np.asarray(h, dtype=complex))
        V = VermaModule(self.c, h, self.Nmax)
        a = np.zeros((h.size, self.Nmax + 1), dtype=complex)
        for N in range(self.Nmax + 1):
            P = parts_of(N)
            vL = np.stack([gamma_vertex(lam, h, h4, h3)*np.ones(h.size) for lam in P], axis=1)
            vR = np.stack([gamma_vertex(mu, h, h1, h2)*np.ones(h.size) for mu in P], axis=1)
            x = np.einsum('kij,kj->ki', V.Ginv[N], vR) if self.gram == 'inverse' else V.solve(N, vR)
            a[:, N] = np.sum(vL*x, axis=1)
        return a

    def H_series(self, h):
        h = np.atleast_1d(np.asarray(h, dtype=complex))
        out = np.zeros((h.size, self.Nmax + 1), dtype=complex)
        for i0 in range(0, h.size, self.chunk):
            hh = h[i0:i0 + self.chunk]
            a = self.z_coeffs(hh)
            acc = np.zeros((hh.size, self.Nmax + 1), dtype=complex)
            for N in range(self.Nmax + 1):
                acc += (a[:, N]*16**N)[:, None]*self.zpows[N][None, :]
            e = (self.c - 1)/24
            H = ser_mul(acc, ser_pow(self.z16, hh - e))
            out[i0:i0 + self.chunk] = ser_mul(H, self.fixed[None, :])
        return out

    def value(self, z, h, H=None):
        c = self.c
        h1, h2, h3, h4 = self.hs
        e = (c - 1)/24
        h = np.asarray(h, dtype=complex)
        if H is None:
            H = self.H_series(h)
        q = elliptic_nome(z)
        Hq = np.atleast_2d(H) @ (q**np.arange(self.Nmax + 1))
        th3 = 1 + 2*sum(q**(m*m) for m in range(1, 40))
        val = (16*q)**(h - e)*z**(e - h1 - h2)*(1 - z)**(e - h2 - h3)*th3**((c - 1)/2 - 4*(h1 + h2 + h3 + h4))*Hq
        return val if h.ndim else val[0]

    def at_tau(self, tau, h, H=None):
        c = self.c
        h1, h2, h3, h4 = self.hs
        e = (c - 1)/24
        h = np.asarray(h, dtype=complex)
        if H is None:
            H = self.H_series(h)
        q, lth2, lth3, lth4 = theta_logs(tau)
        llam, l1mlam = 4*(lth2 - lth3), 4*(lth4 - lth3)
        Hq = np.atleast_2d(H) @ (q**np.arange(np.atleast_2d(H).shape[1]))
        expo = ((h - e)*(np.log(16) + 1j*np.pi*tau) + (e - h1 - h2)*llam + (e - h2 - h3)*l1mlam
                + ((c - 1)/2 - 4*(h1 + h2 + h3 + h4))*lth3)
        val = np.exp(expo)*Hq
        return val if h.ndim else val[0]


def block_c1_sixteenth(z, P):
    """Exact c = 1 block with all external weights 1/16 and internal weight h = -P^2."""
    K, Kp = ellipk(z), ellipk(1 - z)
    tau = 1j*Kp/K
    q = np.exp(1j*np.pi*tau)
    th3 = 1 + 2*sum(q**(n*n) for n in range(1, 60))
    return 2**(-4*P**2)*np.exp(-1j*np.pi*P**2*tau)/(z**0.125*(1 - z)**0.125*th3)


# ---------------------------------------------------------------------------------------------
# torus one-point blocks
# ---------------------------------------------------------------------------------------------
def eta_prod_series(n):
    """prod_{k>=1} (1 - q^k) as a q-series of length n."""
    s = np.zeros(n, dtype=complex); s[0] = 1
    for k in range(1, n):
        s2 = s.copy()
        s2[k:] -= s[:n - k]
        s = s2
    return s


class TorusBlock:
    """Torus one-point blocks V_tau(P; P0) = q^{h - c/24} sum_{N<=Nmax} T_N(h) q^N,  T_0 = 1,  q = e^{2 pi i tau},
    at central charge c and external weight h0 = (c-1)/24 - P0^2, vectorised over the internal momenta P
    (h = (c-1)/24 - P^2).  Brute force from the Verma module:
        T_N = sum_{|lam|=|mu|=N} (G_N^{-1})_{mu lam} <h; lam| phi_0(1) |h; mu>,
    with the matrix elements from the Ward identity [L_n, phi(z)] = z^n (z d/dz + (n+1) h0) phi(z).
    Valid for any complex c (only c and h0 enter)."""
    def __init__(self, c, P0, Nmax=8):
        self.c, self.P0, self.Nmax = c, P0, Nmax
        self.h0 = (c - 1)/24 - P0**2

    def coeffs(self, P):
        """T_N(P), N = 0..Nmax; rows = internal momenta."""
        P = np.atleast_1d(np.asarray(P, dtype=complex))
        h = (self.c - 1)/24 - P**2
        h0 = self.h0
        V = VermaModule(self.c, h, self.Nmax)
        memo = {}

        def E(lam, nu):
            key = (lam, nu)
            if key in memo:
                return memo[key]
            if not lam:
                val = np.ones(h.size, dtype=complex)
                tail = 0
                for i in range(len(nu) - 1, -1, -1):
                    val = val*(tail + nu[i]*h0)
                    tail += nu[i]
            else:
                n, lam2 = lam[0], lam[1:]
                val = ((h + sum(lam2) - h0 - (h + sum(nu))) + (n + 1)*h0)*E(lam2, nu)
                for nu2, cf in V.apply_L(n, nu).items():
                    val = val + cf*E(lam2, nu2)
            memo[key] = val
            return val

        T = np.zeros((h.size, self.Nmax + 1), dtype=complex)
        for N in range(self.Nmax + 1):
            Pp = parts_of(N)
            Em = np.zeros((h.size, len(Pp), len(Pp)), dtype=complex)        # Em[:, i, j] = <lam_i| phi_0 |mu_j>
            for i in range(len(Pp)):
                for j in range(len(Pp)):
                    Em[:, i, j] = E(Pp[i], Pp[j])
            T[:, N] = np.einsum('kii->k', V.solve(N, Em))                   # T_N = tr(G^{-1} E)
        return T

    def value(self, tau, P, T=None):
        P = np.asarray(P, dtype=complex)
        if T is None:
            T = self.coeffs(P)
        q = np.exp(2j*np.pi*tau)
        h = (self.c - 1)/24 - P**2
        val = q**(h - self.c/24)*(np.atleast_2d(T) @ (q**np.arange(self.Nmax + 1)))
        return val if P.ndim else val[0]


def poghossian_values(c, P0, P, tau, order=6):
    """Cross-check of TorusBlock against sphere blocks (Poghossian relation): returns the two sides of
         prod_n (1-q^n) q^{-(h-(c-1)/24)} V_tau(P; P0)  =  H^{sphere}_{b', sqrt2 P}[P0/sqrt2, b'/4, b'/4, b'/4](q'),
       b'^2 = 2 b^2,  q = e^{2 pi i tau} = q'^2,  each truncated (torus to q^order, sphere to q'^{2 order})."""
    q, qp = np.exp(2j*np.pi*tau), np.exp(1j*np.pi*tau)
    T = TorusBlock(c, P0, Nmax=order).coeffs(P)[0]
    lhs = ser_mul(eta_prod_series(order + 1), T) @ q**np.arange(order + 1)
    x = (c - 13)/6
    b2 = (x + np.sqrt(x*x - 4 + 0j))/2
    cp = 13 + 6*(2*b2 + 1/(2*b2))
    ep = (cp - 1)/24
    sb = SphereBlock(cp, ep - b2/8, ep - P0**2/2, ep - b2/8, ep - b2/8, Nmax=2*order)
    rhs = sb.H_series(ep - 2*P**2)[0] @ qp**np.arange(2*order + 1)
    return lhs, rhs


# ---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    print("1. level-1 and level-2 coefficients against the exact formulas (c = -2):")
    c, h = -2.0, 0.37
    h1, h2, h3, h4 = 0.13, 0.21, 0.17, 0.29
    a = SphereBlock(c, h1, h2, h3, h4, Nmax=3).z_coeffs(h)[0]
    print(f"   a_1 = {a[1].real:.15f}   exact (h+h2-h1)(h+h3-h4)/(2h) = {(h + h2 - h1)*(h + h3 - h4)/(2*h):.15f}")
    Gm = np.array([[4*h*(2*h + 1), 6*h], [6*h, 4*h + c/2]])
    vL = np.array([(h + h3 - h4)*(h + h3 - h4 + 1), h + 2*h3 - h4])
    vR = np.array([(h + h2 - h1)*(h + h2 - h1 + 1), h + 2*h2 - h1])
    print(f"   a_2 = {a[2].real:.15f}   exact = {vL @ np.linalg.solve(Gm, vR):.15f}")
    print("2. c = 1, external weights 1/16: elliptic block vs the exact block:")
    sb = SphereBlock(1.0, 1/16, 1/16, 1/16, 1/16, Nmax=12)
    Ps = np.array([0.31j, 0.15 + 0.6j])
    H = sb.H_series(-Ps**2)
    for z in [0.3, 0.5, 0.7]:
        v = sb.value(z, -Ps**2, H)
        ex = np.array([block_c1_sixteenth(z, P) for P in Ps])
        print(f"   z={z}: max rel. error {np.max(np.abs(v/ex - 1)):.1e}")
    print("3. at_tau: lambda(+-1/2 + i/2) = 2 -+ i0, and agreement with value() at tau = i (z = 1/2):")
    for tau in [0.5 + 0.5j, -0.5 + 0.5j]:
        q, l2, l3, l4 = theta_logs(tau)
        print(f"   tau={tau}: lambda = {np.exp(4*(l2 - l3)):.12f}")
    sb = SphereBlock(28.0, 1.1, 1.2, 1.15, 1.3, Nmax=12)
    print(f"   value(1/2)/at_tau(i) = {sb.value(0.5, 1.215)/sb.at_tau(1j, 1.215):.14f}")
    print("4. torus blocks (brute force) vs sphere blocks through the Poghossian relation, tau = i:")
    for (c, P0, P, lab) in [(1 + 6*(0.83 + 0.11j + 1/(0.83 + 0.11j))**2, 0.19 - 0.08j, 0.27 + 0.05j, "complex c"),
                            (28.0, 0.21j, 0.3j, "c = 28"), (-2.0, 0.17j, 0.15 + 0.4j, "c = -2")]:
        lhs, rhs = poghossian_values(c, P0, P, 1j)
        print(f"   {lab}: |torus/sphere - 1| = {abs(lhs/rhs - 1):.1e}")
    print("5. speed: 1200 internal weights at level 12 ...", end=" ", flush=True)
    t0 = time.time()
    sb = SphereBlock(-2.0, 0.1, 0.2, 0.15, 0.3, Nmax=12)
    HH = sb.H_series(-0.15**2 + np.linspace(-4.5, 4.5, 1200)**2 + 0j)
    print(f"{time.time() - t0:.1f} s")
