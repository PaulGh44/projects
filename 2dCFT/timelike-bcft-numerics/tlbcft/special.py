r"""
Special functions: the Barnes G function and the Barnes double Gamma function Gamma_b.
=====================================================================================

Everything in this package is built from three special functions:

  * the Barnes G function  G(z),  G(z+1) = Gamma(z) G(z),  G(1) = 1;
  * the Barnes double Gamma function  Gamma_b(x)  in the normalisation of the paper (Supplemental
    Material, "Barnes functions and bulk data"):

        Gamma_b(Q/2) = 1,      Gamma_b(x + b) = sqrt(2 pi) b^{b x - 1/2} Gamma_b(x) / Gamma(b x),
        Q = b + 1/b,           Gamma_b = Gamma_{1/b};

  * the double sine  S_b(x) = Gamma_b(x) / Gamma_b(Q - x).

Three independent implementations of log Gamma_b are provided, so that every number in the checks
can be cross-validated:

  1. RationalB(m, n):  exact reduction at rational b^2 = m/n to a finite product of Barnes G
     functions (fast; used for all closed-form kernels).  With s = sqrt(m n),

        Gamma_b(z) = exp(phi(z)) / prod_{r=0}^{m-1} prod_{l=0}^{n-1} G(z/s + r/m + l/n),
        phi(z)     = -(log s / 2) z^2 + (s/2 log 2pi + Q/2 log s) z + c0,

     where the constant c0 is fixed by Gamma_b(Q/2) = 1.

  2. GeneralB(b):  the integral representation, valid for any real b > 0 (irrational b included),

        log Gamma_b(x) = int_0^oo dt/t [ (e^{-x t} - e^{-Q t/2}) / ((1 - e^{-b t})(1 - e^{-t/b}))
                                         - (a^2/2) e^{-t} - a/t ],      a = Q/2 - x,

     evaluated with Gauss-Legendre panels on [0, T1] (vectorised over x) and the analytic tail on
     [T1, oo); Re x is first shifted to >= X0 with the b-shift relation above.

  3. lgamma_b_int(x, b):  the same integral with adaptive scipy.quad (slow, scalar, used only as an
     independent reference).

Logs are returned (not values) because the kernels are products of dozens of Gamma_b factors.
The functions are analytic; the imaginary part of a log is defined only modulo 2 pi i, which is
harmless because only exp(log) is ever used.

Self-test:  python -m tlbcft.special
"""
import warnings

import numpy as np
from scipy import integrate
from scipy.special import loggamma, exp1

# ---------------------------------------------------------------------------------------------
# Barnes G function
# ---------------------------------------------------------------------------------------------
ZETA_PRIME_M1 = -0.1654211437004509292          # zeta'(-1)
# Bernoulli numbers B_{2k+2}, k = 1..8, for the Stirling-type expansion of log G
_BERN = {4: -1/30, 6: 1/42, 8: -1/30, 10: 5/66, 12: -691/2730, 14: 7/6, 16: -3617/510, 18: 43867/798}


def logG(z, shift_to=12.0):
    """log G(z) for complex z (vectorised), up to multiples of 2 pi i (only exp(logG) is ever used).

    Method: shift z -> w + 1 = z + N with Re(z + N) >= shift_to, using G(z+1) = Gamma(z) G(z):
        log G(z) = log G(z + N) - sum_{k=0}^{N-1} log Gamma(z + k),
        sum_{k=0}^{N-1} log Gamma(z + k) = N log Gamma(z) + sum_{j=0}^{N-2} (N - 1 - j) log(z + j),
    then the asymptotic expansion
        log G(w+1) = w^2/2 log w - 3 w^2/4 + w/2 log 2pi - log(w)/12 + zeta'(-1) + sum_{k=1}^{8} B_{2k+2}/(4k(k+1) w^{2k}),
    whose first omitted term is < 1e-17 for |w| >= 12.  The functional equation holds to ~1e-13 relative
    (python -m tlbcft.special).
    """
    z = np.asarray(z, dtype=complex)
    Nk = np.maximum(0, np.ceil(shift_to - z.real)).astype(int)
    Nmax = int(Nk.max()) if Nk.size else 0
    acc = Nk*loggamma(z)
    with np.errstate(divide='ignore', invalid='ignore'):     # G has zeros at 0, -1, -2, ...: log -> -inf there
        for j in range(Nmax - 1):
            acc = acc + np.where(j <= Nk - 2, (Nk - 1 - j)*np.log(z + j), 0.0)
    w = z + Nk - 1
    lg = (w*w/2)*np.log(w) - 0.75*w*w + (w/2)*np.log(2*np.pi) - np.log(w)/12 + ZETA_PRIME_M1
    for k in range(1, 9):
        lg = lg + _BERN[2*k+2]/(4*k*(k+1)*w**(2*k))
    return lg - acc


def G(z):
    """Barnes G function."""
    return np.exp(logG(z))


# ---------------------------------------------------------------------------------------------
# Gamma_b at rational b^2 = m/n: reduction to Barnes G
# ---------------------------------------------------------------------------------------------
class RationalB:
    """log Gamma_b and log S_b at b^2 = m/n (exact reduction to Barnes G functions).

    Attributes: m, n, b = sqrt(m/n), Q = b + 1/b.
    """
    def __init__(self, m, n):
        self.m, self.n = m, n
        self.s = np.sqrt(m*n)
        self.b = np.sqrt(m/n)
        self.Q = self.b + 1/self.b
        self.shifts = np.array([r/m + l/n for r in range(m) for l in range(n)])
        s, Q = self.s, self.Q
        self.A = 0.5*s*np.log(2*np.pi) + 0.5*Q*np.log(s)
        # c0 from the normalisation Gamma_b(Q/2) = 1
        self.c0 = (0.5*np.log(s)*(Q/2)**2 - self.A*(Q/2)) + np.sum(logG(Q/(2*s) + self.shifts))

    def lgamma_b(self, z):
        """log Gamma_b(z), vectorised over z."""
        z = np.asarray(z, dtype=complex)
        phi = -0.5*np.log(self.s)*z**2 + self.A*z + self.c0
        return phi - np.sum(logG(z[..., None]/self.s + self.shifts), axis=-1)

    def lS_b(self, z):
        """log S_b(z) = log Gamma_b(z) - log Gamma_b(Q - z)."""
        return self.lgamma_b(z) - self.lgamma_b(self.Q - np.asarray(z, dtype=complex))


# ---------------------------------------------------------------------------------------------
# Gamma_b at arbitrary real b: vectorised integral representation
# ---------------------------------------------------------------------------------------------
class GeneralB:
    """log Gamma_b and log S_b at arbitrary real b > 0 (integral representation, vectorised).

    Parameters X0 (shift target for Re x), T1 (end of the numerical t-range), panel and ngl (Gauss-Legendre
    panels) control the accuracy; the defaults give ~1e-13 absolute accuracy for |Im x| <~ 10.
    """
    def __init__(self, b, X0=2.0, T1=18.0, panel=1.0, ngl=16):
        self.b = float(b)
        self.Q = self.b + 1/self.b
        self.X0, self.T1 = X0, T1
        xg, wg = np.polynomial.legendre.leggauss(ngl)
        edges = np.arange(0.0, T1 + 1e-12, panel)
        t, w = [], []
        for k in range(len(edges) - 1):
            a, c = edges[k], edges[k+1]
            t.append(0.5*(c - a)*xg + 0.5*(c + a))
            w.append(0.5*(c - a)*wg)
        self.t = np.concatenate(t)
        self.w = np.concatenate(w)
        b, Q, tt = self.b, self.Q, self.t
        self.eQ = np.exp(-Q*tt/2)
        self.E = np.expm1(-b*tt)*np.expm1(-tt/b)
        self.et = np.exp(-tt)
        # tail of int_{T1}^oo dt/t e^{-Qt/2} (1 - e^{-bt})^{-1}(1 - e^{-t/b})^{-1} (e^{a t} - 1), leading terms
        self.tail_g = -exp1(Q*T1/2) - exp1((Q/2 + b)*T1) - exp1((Q/2 + 1/b)*T1)

    def lgamma_b(self, z):
        """log Gamma_b(z), vectorised over z."""
        z = np.asarray(z, dtype=complex)
        scalar = (z.ndim == 0)
        z = np.atleast_1d(z)
        b, Q = self.b, self.Q
        # shift Re z up to >= X0:  log Gamma_b(z) = log Gamma_b(z + k b) + sum_j [log Gamma(b z_j) - ...]
        k = np.maximum(0, np.ceil((self.X0 - z.real)/b)).astype(int)
        corr = np.zeros_like(z)
        for j in range(int(k.max()) if k.size else 0):
            zj = z + j*b
            corr = corr + np.where(j < k, loggamma(b*zj) - 0.5*np.log(2*np.pi) - (b*zj - 0.5)*np.log(b), 0.0)
        x = z + k*b
        a = Q/2 - x
        tt = self.t[None, :]
        g = self.eQ[None, :]*np.expm1(a[:, None]*tt)/self.E[None, :]
        integrand = (g - 0.5*a[:, None]**2*self.et[None, :] - a[:, None]/tt)/tt
        I = integrand @ self.w
        I = I - a/self.T1 - 0.5*a*a*exp1(self.T1) + self.tail_g
        out = I + corr
        return out[0] if scalar else out

    def lS_b(self, z):
        """log S_b(z)."""
        return self.lgamma_b(z) - self.lgamma_b(self.Q - np.asarray(z, dtype=complex))


def lgamma_b_int(x, b):
    """log Gamma_b(x) by adaptive quadrature of the integral representation (scalar; reference only)."""
    x = complex(x)
    Q = b + 1/b
    corr = 0.0 + 0.0j
    while x.real < 0.5:
        corr += loggamma(b*x) - 0.5*np.log(2*np.pi) - (b*x - 0.5)*np.log(b)
        x = x + b
    a = Q/2 - x

    def integrand(t):
        if t < 1e-3:       # rewrite with expm1 to avoid cancellation at small t
            e1, e2 = np.expm1(-x*t), np.expm1(-Q*t/2)
            d1, d2 = -np.expm1(-b*t), -np.expm1(-t/b)
            return ((e1 - e2)/(d1*d2) - a*a/2*np.exp(-t) - a/t)/t
        return ((np.exp(-x*t) - np.exp(-Q*t/2))/((1 - np.exp(-b*t))*(1 - np.exp(-t/b)))
                - a*a/2*np.exp(-t) - a/t)/t
    with warnings.catch_warnings():      # quad warns about round-off at the 1e-13 level; harmless here
        warnings.simplefilter("ignore")
        re = integrate.quad(lambda t: integrand(t).real, 0, np.inf, limit=400, epsabs=1e-13, epsrel=1e-13)[0]
        im = integrate.quad(lambda t: integrand(t).imag, 0, np.inf, limit=400, epsabs=1e-13, epsrel=1e-13)[0]
    return re + 1j*im + corr


# ---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import math
    print("Barnes G at integers and at 1/2 (Glaisher constant A):")
    A = 1.2824271291006226
    for z, v in {1: 1.0, 4: 2.0, 6: 288.0, 0.5: 2**(1/24)*math.exp(1/8)*math.pi**(-1/4)*A**(-1.5)}.items():
        print(f"   G({z}) = {G(z).real:.15g}   exact {v:.15g}")
    z = -2.3 + 1.1j
    print(f"   functional equation G(z+1)/(Gamma(z)G(z)) at z={z}: {G(z+1)/(np.exp(loggamma(z))*G(z)):.15f}")
    print("Gamma_b: rational reduction vs integral representations:")
    for (m, n) in [(1, 1), (1, 2), (2, 3), (5, 7)]:
        rb = RationalB(m, n); gb = GeneralB(rb.b)
        worst1 = worst2 = 0.0
        for z in [0.7, 1.3 + 0.4j, 0.35 - 0.9j, 2.1 + 0.2j, -0.4 + 1.7j]:
            a = rb.lgamma_b(z)
            worst1 = max(worst1, abs(a - lgamma_b_int(z, rb.b)))
            worst2 = max(worst2, abs(np.exp(a - gb.lgamma_b(z)) - 1))
        print(f"   b^2={m}/{n}: |reduction - quad| <= {worst1:.1e},  |reduction/GeneralB - 1| <= {worst2:.1e}")
    rb = RationalB(1, 2); b, Q = rb.b, rb.Q
    x = 0.4 + 0.3j
    print(f"   Gamma_b(Q/2) = {np.exp(rb.lgamma_b(Q/2)):.15f}")
    print(f"   shift relation: {np.exp(rb.lgamma_b(x + b) - rb.lgamma_b(x)):.15f}"
          f"  vs  {np.sqrt(2*np.pi)*b**(b*x - 0.5)/np.exp(loggamma(b*x)):.15f}")
    P = 0.23j
    rho0 = -4*np.sqrt(2)*np.sin(2*np.pi*b*P)*np.sin(2*np.pi*P/b)
    lhs = np.exp(rb.lgamma_b(2*P) + rb.lgamma_b(-2*P) - rb.lgamma_b(Q + 2*P) - rb.lgamma_b(Q - 2*P))
    print(f"   Gamma_b(2P)Gamma_b(-2P)/Gamma_b(Q+-2P) = {lhs:.15f}   sqrt2/rho0(P) = {np.sqrt(2)/rho0:.15f}")
