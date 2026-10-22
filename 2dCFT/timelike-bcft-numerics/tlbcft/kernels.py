r"""
Virasoro crossing kernels at c >= 25 and their Virasoro-Wick rotations.
======================================================================

Conventions (paper, Eq. (1) and Supplemental Material Sec. S1)
----------------------------------------------------------------
  c = 1 + 6 Q^2,  Q = b + 1/b,  h = Q^2/4 - P^2,  physical spectrum P in iR.
  Sphere four-point blocks V_z(P_s; P1 P2 P3 P4) have the operators P1, P2, P3, P4 at 0, z, 1, infinity and
  leading coefficient one.  The fusion kernel is defined by

      V_z(P_s; P_1234) = int_{iR} dP_t/i  F_{P_s,P_t}[P2 P3; P1 P4]  V_{1-z}(P_t; P_3214),

  i.e. the s-channel fuses (P1,P2) and (P3,P4), the t-channel (P2,P3) and (P1,P4).  F is even in all six
  momenta.  The modular kernel S_{P_s,P_t}[P0] is defined by the torus one-point blocks:

      (-i tau)^{h0} V_tau(P_s; P0) = int_{iR} dP_t/i  S_{P_s,P_t}[P0]  V_{-1/tau}(P_t; P0).

  ** Argument order used throughout the code **:  F(Ps, Pt, P1, P2, P3, P4) = F_{Ps,Pt}[P2 P3; P1 P4].
  The helper Fm(Ps, Pt, A, B, C, D) = F_{Ps,Pt}[A B; C D] follows the "matrix" order of the paper.

The Virasoro-Wick rotation (VWR), paper Eq. (2)
-----------------------------------------------
  RF_{Ps,Pt}[P2 P3; P1 P4] = (Pt/Ps) F_{iPt,iPs}[iP2 iP1; iP3 iP4],     RS_{Ps,Pt}[P0] = (Pt/Ps) S_{iPt,iPs}[iP0],

  with F, S the c >= 25 kernels at the real parameter b = bhat.  RF and RS are odd in their internal momenta.

Implementations
---------------
  FusionKernel(m, n)      closed form of Roussillon(-Tsiares) at rational b^2 = m/n [arXiv:2512.03172]
                          (vectorised over all six momenta; coincident-root limit handled);
  tv_kernel(...)          Teschner-Vartanov Barnes integral on a straight contour (valid for imaginary momenta;
                          used to validate the closed form);
  GeneralFusionKernel(b)  Teschner-Vartanov integral at ANY real b, with the residues of the poles that cross the
                          contour when the momenta have real parts (paper Sec. S8, "Kernels at arbitrary b");
  ModularKernel(m, n)     S from F at b' = sqrt2 b (Roussillon [arXiv:2405.09325], paper Eq. (S5)):
                            S^{(b)}_{Ps,Pt}[P0] = sqrt2 256^{Ps^2-Pt^2} F^{(sqrt2 b)}_{sqrt2 Ps, sqrt2 Pt}[P0/sqrt2  e; e  e],
                            e = b/(2 sqrt2);
  ModularKernelGeneral(b) the same with GeneralFusionKernel;
  mod_kernel_tv(...)      Teschner's Barnes integral for S (validation only).

Closed form at b^2 = m/n (s = sqrt(mn), kappa = 1/m + 1/n = Q/s)
-----------------------------------------------------------------
  F = F_b/(2 sqrt(Delta)) sum_{k=0}^{s^2-1} [ calF(u_1 + k/s^2) - calF(u_2 + k/s^2) ],
  u_j = log(z_j)/(2 pi i s^2),  z_{1,2} the roots of  alpha z^2 + beta z + gamma = 0,  Delta = beta^2 - 4 alpha gamma,
  alpha = sum_{sigma} sigma_E exp(2 pi i s^2 [sigma_E kappa/4 + sigma.P/(2s)]),  gamma = (same with -2 pi i),
  beta  = 4 sum_{(ij) in {(s,t),(1,3),(2,4)}} sin(2 pi s P_i) sin(2 pi s P_j),
  calF(u) = prod_{sigma_E=-1} Gt_mn(u - 1 + kappa/4 - sigma.P/(2s)) / prod_{sigma_E=+1} Gt_mn(u - kappa/4 - sigma.P/(2s)),
  F_b = C_mn G_mn(2Pt/s) G_mn(-2Pt/s)/[G_mn(kappa + 2Ps/s) G_mn(kappa - 2Ps/s)] prod_faces prod_{sigma_f} G_mn(kappa/2 + sigma.P_f/s)^{-eta_f},
  G_mn(x) = prod_{r<m,l<n} G(x + r/m + l/n),   Gt_mn(x) = prod_{r<m,l<n} Gt(x + (r+1)/m + (l+1)/n - 1 - kappa/2),
  Gt(x) = G(1+x)/G(1-x),   C_mn = i (2 pi)^{4mn-m-n}/s.
  The sum over sigma runs over the 8 sign vectors (sigma_s, sigma_t, sigma_1..sigma_4) with product +1 at every vertex
  of the tetrahedron; sigma_E is the product of all six signs.  The result does not depend on the branch of sqrt(Delta).

Self-test:  python -m tlbcft.kernels
"""
import itertools
from math import gcd

import numpy as np
from scipy import integrate

from .special import logG, RationalB, GeneralB

# ---------------------------------------------------------------------------------------------
# tetrahedral index sets: edges (s, t, 1, 2, 3, 4) -> 0..5
# ---------------------------------------------------------------------------------------------
S_, T_, E1, E2, E3, E4 = 0, 1, 2, 3, 4, 5
FACES = [(S_, E1, E2), (S_, E3, E4), (T_, E2, E3), (T_, E1, E4)]   # three-point vertices of the two channels
VERTS = [(T_, E3, E4), (T_, E1, E2), (S_, E1, E4), (S_, E2, E3)]   # triples whose sign product must be +1
SIGMAS = [np.array(sig) for sig in itertools.product([1, -1], repeat=6)
          if all(sig[a]*sig[b]*sig[c] == 1 for (a, b, c) in VERTS)]
assert len(SIGMAS) == 8
_SIG = np.array(SIGMAS, dtype=float)
_SIGE = np.prod(_SIG, axis=1)                                         # sigma_E = product of the six signs


def _face_terms():
    out = []
    for f in FACES:
        eta = 1 if T_ in f else -1
        for sf in itertools.product([1, -1], repeat=3):
            if np.prod(sf) == eta:
                out.append((eta, f, sf))
    return out


_FACE_TERMS = _face_terms()


# ---------------------------------------------------------------------------------------------
# closed-form fusion kernel at rational b^2
# ---------------------------------------------------------------------------------------------
class FusionKernel:
    """c >= 25 fusion kernel F_{Ps,Pt}[P2 P3; P1 P4] at b^2 = m/n (closed form), vectorised.

    F(Ps, Pt, P1, P2, P3, P4): arguments broadcast as numpy arrays.
    RF(...): the VWR image at bhat^2 = m/n.
    """
    def __init__(self, m, n, degenerate_tol=1e-9):
        self.m, self.n = m, n
        self.s = np.sqrt(m*n)
        self.b = np.sqrt(m/n)
        self.kappa = 1/m + 1/n
        self.shG = np.array([r/m + l/n for r in range(m) for l in range(n)])
        self.shGt = np.array([(r+1)/m + (l+1)/n - 1 - self.kappa/2 for r in range(m) for l in range(n)])
        self.lconst = 0.5j*np.pi + (4*m*n - m - n)*np.log(2*np.pi) - np.log(self.s)   # log of C_mn = i (2pi)^{4mn-m-n}/s
        self.k_shifts = np.arange(m*n)/(m*n)
        self.degenerate_tol = degenerate_tol

    def _lG(self, x):          # log G_mn(x)
        return np.sum(logG(np.asarray(x, dtype=complex)[..., None] + self.shG), axis=-1)

    def _lGt(self, x):         # log Gt_mn(x)
        a = np.asarray(x, dtype=complex)[..., None] + self.shGt
        return np.sum(logG(1 + a) - logG(1 - a), axis=-1)

    def _lcalF(self, u, Pf):
        """log calF(u); u has shape (K, N), Pf has shape (6, N)."""
        s, kap = self.s, self.kappa
        lf = np.zeros_like(u)
        for j in range(_SIG.shape[0]):
            arg = (_SIG[j] @ Pf)/(2*s)
            if _SIGE[j] == -1:
                lf = lf + self._lGt(u - 1 + kap/4 - arg[None, :])
            else:
                lf = lf - self._lGt(u - kap/4 - arg[None, :])
        return lf

    def _pieces(self, Pf):
        with np.errstate(divide='ignore', invalid='ignore'):
            return self._pieces_raw(Pf)

    def _pieces_raw(self, Pf):
        s, kap = self.s, self.kappa
        sn = np.sin(2*np.pi*s*Pf)
        ex = 2j*np.pi*s*s*(_SIGE[:, None]*kap/4 + (_SIG @ Pf)/(2*s))
        alpha = np.sum(_SIGE[:, None]*np.exp(ex), axis=0)
        gamma = np.sum(_SIGE[:, None]*np.exp(-ex), axis=0)
        beta = 4*(sn[S_]*sn[T_] + sn[E1]*sn[E3] + sn[E2]*sn[E4])
        Delta = beta*beta - 4*alpha*gamma
        sq = np.sqrt(Delta)
        # roots; the smaller-magnitude numerator suffers cancellation -> use Vieta z1 z2 = gamma/alpha
        r1 = (-beta + sq)/(2*alpha)
        r2 = (-beta - sq)/(2*alpha)
        big1 = np.abs(-beta + sq) >= np.abs(-beta - sq)
        z1 = np.where(big1, r1, gamma/(alpha*r2))
        z2 = np.where(big1, gamma/(alpha*r1), r2)
        lpref = (self.lconst + self._lG(2*Pf[T_]/s) + self._lG(-2*Pf[T_]/s)
                 - self._lG(kap + 2*Pf[S_]/s) - self._lG(kap - 2*Pf[S_]/s))
        for (eta, f, sf) in _FACE_TERMS:
            lpref = lpref - eta*self._lG(kap/2 + (sf[0]*Pf[f[0]] + sf[1]*Pf[f[1]] + sf[2]*Pf[f[2]])/s)
        return alpha, beta, gamma, Delta, sq, z1, z2, lpref

    def root_terms(self, Ps, Pt, P1, P2, P3, P4, return_sqrt=False):
        """(T1, T2): F = T1 - T2, the two root terms of the closed form, computed with the principal branch of
        sqrt(Delta) (returned as a third value if return_sqrt).  Under sqrt(Delta) -> -sqrt(Delta): T1 -> -T2, T2 -> -T1,
        so F is branch-independent while T1 + T2 (the physical kernel, paper Sec. S1/S7) changes sign."""
        P = np.stack(np.broadcast_arrays(*[np.asarray(x, dtype=complex) for x in (Ps, Pt, P1, P2, P3, P4)]))
        shp = P.shape[1:]
        Pf = P.reshape(6, -1)
        alpha, beta, gamma, Delta, sq, z1, z2, lpref = self._pieces(Pf)
        out = []
        for z in (z1, z2):
            uu = (np.log(z)/(2j*np.pi*self.s**2))[None, :] + self.k_shifts[:, None]
            out.append((np.exp(lpref)/(2*sq)*np.sum(np.exp(self._lcalF(uu, Pf)), axis=0)).reshape(shp))
        if return_sqrt:
            return out[0], out[1], sq.reshape(shp)
        return out[0], out[1]

    def F(self, Ps, Pt, P1, P2, P3, P4):
        """F_{Ps,Pt}[P2 P3; P1 P4]."""
        P = np.stack(np.broadcast_arrays(*[np.asarray(x, dtype=complex) for x in (Ps, Pt, P1, P2, P3, P4)]))
        shp = P.shape[1:]
        Pf = P.reshape(6, -1)
        alpha, beta, gamma, Delta, sq, z1, z2, lpref = self._pieces(Pf)
        s2 = self.s**2
        out = []
        for z in (z1, z2):
            uu = (np.log(z)/(2j*np.pi*s2))[None, :] + self.k_shifts[:, None]
            out.append(np.sum(np.exp(self._lcalF(uu, Pf)), axis=0))
        with np.errstate(divide='ignore', invalid='ignore'):
            val = np.exp(lpref)/(2*sq)*(out[0] - out[1])
        # coincident roots (Delta -> 0): F = -F_b/(2 pi i s^2 beta) sum_k calF'(u0 + k/s^2)
        degen = np.abs(Delta) <= self.degenerate_tol*(np.abs(beta*beta) + np.abs(4*alpha*gamma))
        if np.any(degen):
            idx = np.where(degen)[0]
            Pd = Pf[:, idx]
            u0 = np.log(-beta[idx]/(2*alpha[idx]))/(2j*np.pi*s2)
            uu = u0[None, :] + self.k_shifts[:, None]
            hstep = 1e-5
            lf0 = self._lcalF(uu, Pd)
            dF = np.exp(lf0)*(self._lcalF(uu + hstep, Pd) - self._lcalF(uu - hstep, Pd))/(2*hstep)
            val[idx] = -np.exp(lpref[idx])/(2j*np.pi*s2*beta[idx])*np.sum(dF, axis=0)
        return val.reshape(shp)

    def Fm(self, Ps, Pt, A, B, C, D):
        """F_{Ps,Pt}[A B; C D] (matrix order of the paper)."""
        return self.F(Ps, Pt, C, A, B, D)

    def RF(self, Ps, Pt, P1, P2, P3, P4):
        """VWR kernel RF_{Ps,Pt}[P2 P3; P1 P4] = (Pt/Ps) F_{iPt,iPs}[iP2 iP1; iP3 iP4]."""
        Ps = np.asarray(Ps, dtype=complex)
        Pt = np.asarray(Pt, dtype=complex)
        return (Pt/Ps)*self.F(1j*Pt, 1j*Ps, 1j*np.asarray(P3), 1j*np.asarray(P2), 1j*np.asarray(P1), 1j*np.asarray(P4))

    def RFm(self, Ps, Pt, A, B, C, D):
        """RF_{Ps,Pt}[A B; C D] (matrix order)."""
        return self.RF(Ps, Pt, C, A, B, D)


# ---------------------------------------------------------------------------------------------
# Teschner-Vartanov integral (straight contour) -- validation of the closed form
# ---------------------------------------------------------------------------------------------
def _tv_prefactor_log(lgamma_b, Q, P):
    lp = lgamma_b(Q + 2*P[S_]) + lgamma_b(Q - 2*P[S_]) - lgamma_b(2*P[T_]) - lgamma_b(-2*P[T_]) - np.log(2)
    for (eta, f, sf) in _FACE_TERMS:
        lp = lp + eta*lgamma_b(Q/2 + sf[0]*P[f[0]] + sf[1]*P[f[1]] + sf[2]*P[f[2]])
    return lp


def tv_kernel(m, n, Ps, Pt, P1, P2, P3, P4, ymax=12.0, ny=6001, c=0.0):
    """Teschner-Vartanov fusion kernel at b^2 = m/n on the straight contour Re u = c:

        F = F_b(P) int_{c+iR} du/i  prod_{sigma} S_b(u + Q/2 + (Q/4) sigma_E + sigma.P/2)^{-sigma_E},
        F_b = Gamma_b(Q +- 2Ps)/(2 Gamma_b(+-2Pt)) prod_faces prod_{sigma_f} Gamma_b(Q/2 + sigma.P_f)^{eta_f}.

    Valid as is when the line separates the two pole families (e.g. all momenta imaginary, c = 0).
    """
    rb = RationalB(m, n)
    P = np.array([Ps, Pt, P1, P2, P3, P4], dtype=complex)
    Q = rb.Q
    lpref = _tv_prefactor_log(rb.lgamma_b, Q, P)
    y = np.linspace(-ymax, ymax, ny)
    u = c + 1j*y
    lf = np.zeros_like(u)
    for sig in SIGMAS:
        sE = np.prod(sig)
        lf = lf - sE*rb.lS_b(u + Q/2 + Q/4*sE + 0.5*np.dot(sig, P))
    return np.exp(lpref)*integrate.simpson(np.exp(lf), x=y)


# ---------------------------------------------------------------------------------------------
# fusion kernel at arbitrary real b (Teschner-Vartanov integral with residue corrections)
# ---------------------------------------------------------------------------------------------
def _all_poles(gb, P, c, window=0.6, listing=4.0):
    """Poles of the TV integrand with Re u within `listing` of the line Re u = c.
    Returns (u0, kind, crossed, near): kind = +1 (right family, u = Q/4 - sigma.P/2 + m b + n/b, sigma_E=+1) or
    -1 (left family, u = -Q/4 - sigma.P/2 - m b - n/b, sigma_E=-1); crossed = lies on the wrong side of the line;
    near = within `window` of the line (its principal part is subtracted from the line integrand)."""
    b, Q = gb.b, gb.Q
    out = []
    for sig in SIGMAS:
        sE = int(np.prod(sig))
        sP = 0.5*np.dot(sig, P)
        if sE == +1:
            base = Q/4 - sP
            depth = c - base.real + listing
            if depth <= 0:
                continue
            for mm in range(0, int(depth/b) + 2):
                for nn in range(0, int(depth*b) + 2):
                    if mm*b + nn/b < depth:
                        u0 = base + mm*b + nn/b
                        out.append((u0, +1, u0.real < c, abs(u0.real - c) < window))
        else:
            base = -Q/4 - sP
            depth = base.real - c + listing
            if depth <= 0:
                continue
            for mm in range(0, int(depth/b) + 2):
                for nn in range(0, int(depth*b) + 2):
                    if mm*b + nn/b < depth:
                        u0 = base - mm*b - nn/b
                        out.append((u0, -1, u0.real > c, abs(u0.real - c) < window))
    return out


class GeneralFusionKernel:
    """F_{Ps,Pt}[P2 P3; P1 P4] at arbitrary real b (paper Sec. S8).

    For momenta with real parts the two pole families of the TV integrand interleave across the line Re u = c; the
    kernel is the straight-line integral plus
        -2 pi sum(residues of right-family poles with Re u0 < c) + 2 pi sum(residues of left-family poles with Re u0 > c).
    Poles within `window` of the line have their principal parts subtracted and integrated in closed form; residues are
    circle integrals.  Coincident poles (trace configurations) are grouped into higher-order poles.
    Scalar interface (one kernel per call); ~0.05-0.2 s per call.
    """
    def __init__(self, b, ymax=5.0, ny=801, c=0.0137, rad=0.004, npts=32):
        self.gb = GeneralB(b)
        self.b = float(b)
        self.Q = self.gb.Q
        self.y = np.linspace(-ymax, ymax, ny)
        self.c, self.rad, self.npts = c, rad, npts
        w = np.ones(ny); w[1:-1:2] = 4; w[2:-1:2] = 2
        self.wS = w/3*(self.y[1] - self.y[0])

    def _log_integrand(self, u, P):
        u = np.asarray(u, dtype=complex)
        lf = np.zeros_like(u)
        for sig in SIGMAS:
            sE = np.prod(sig)
            lf = lf - sE*self.gb.lS_b(u + self.Q/2 + self.Q/4*sE + 0.5*np.dot(sig, P))
        return lf

    def _J(self, P):
        c = self.c
        u = c + 1j*self.y
        Y = self.y[-1]
        fv = np.exp(self._log_integrand(u, P))
        groups = []
        for (u0, kind, crossed, near) in _all_poles(self.gb, P, c):
            for g in groups:
                if abs(g['u0'] - u0) < 1e-8:
                    g['mult'] += 1
                    g['crossed'] = g['crossed'] or crossed
                    break
            else:
                groups.append(dict(u0=u0, kind=kind, crossed=crossed, near=near, mult=1))
        pos = np.array([g['u0'] for g in groups]) if groups else np.zeros(0, dtype=complex)
        th = 2*np.pi*(np.arange(self.npts) + 0.5)/self.npts
        # Laurent coefficients a_{-j} of the poles that are crossed or near the line, from circle integrals
        # (all circles evaluated in one vectorised call)
        need = [k for k, g in enumerate(groups) if g['crossed'] or g['near']]
        rads = []
        for k in need:
            d = np.abs(pos - groups[k]['u0']); d[k] = np.inf
            rads.append(min(self.rad, 0.3*d.min()) if d.size > 1 else self.rad)
        if need:
            pts = np.concatenate([groups[k]['u0'] + r*np.exp(1j*th) for k, r in zip(need, rads)])
            fc_all = np.exp(self._log_integrand(pts, P)).reshape(len(need), self.npts)
        for g in groups:
            g['a'] = []
        for i, (k, rad) in enumerate(zip(need, rads)):
            g = groups[k]
            g['a'] = [rad**j*np.mean(fc_all[i]*np.exp(1j*j*th)) for j in range(1, g['mult'] + 1)]   # a_{-1}, a_{-2}, ...
        sub = np.zeros_like(fv)
        exact = 0.0 + 0.0j
        for g in groups:
            if not g['near']:
                continue
            u0 = g['u0']; x0, y0 = u0.real, u0.imag
            for j, a in enumerate(g['a'], start=1):
                sub += a/(u - u0)**j
                if j == 1:
                    exact += a*((np.arctan((Y - y0)/(c - x0)) - np.arctan((-Y - y0)/(c - x0)))
                                - 0.5j*(np.log((c - x0)**2 + (Y - y0)**2) - np.log((c - x0)**2 + (Y + y0)**2)))
                else:
                    exact += a*((c + 1j*Y - u0)**(1 - j) - (c - 1j*Y - u0)**(1 - j))/(1j*(1 - j))
        line = (fv - sub) @ self.wS + exact
        corr = 0.0 + 0.0j
        for g in groups:
            if g['crossed']:
                corr += (-2*np.pi if g['kind'] == +1 else 2*np.pi)*g['a'][0]
        return line + corr

    def F(self, Ps, Pt, P1, P2, P3, P4):
        P = np.array([Ps, Pt, P1, P2, P3, P4], dtype=complex)
        return np.exp(_tv_prefactor_log(self.gb.lgamma_b, self.Q, P))*self._J(P)

    def Fm(self, Ps, Pt, A, B, C, D):
        return self.F(Ps, Pt, C, A, B, D)

    def RF(self, Ps, Pt, P1, P2, P3, P4):
        return (Pt/Ps)*self.F(1j*Pt, 1j*Ps, 1j*P3, 1j*P2, 1j*P1, 1j*P4)

    def RFm(self, Ps, Pt, A, B, C, D):
        return self.RF(Ps, Pt, C, A, B, D)


# ---------------------------------------------------------------------------------------------
# modular kernel
# ---------------------------------------------------------------------------------------------
class ModularKernel:
    """Modular kernel S^{(b)}_{Ps,Pt}[P0] at b^2 = m/n via the fusion kernel at b'^2 = 2 m/n (paper Eq. (S5)):

        S_{Ps,Pt}[P0] = sqrt2 * 256^{Ps^2 - Pt^2} * F^{(sqrt2 b)}_{sqrt2 Ps, sqrt2 Pt}[P0/sqrt2  e; e  e],  e = b/(2 sqrt2).

    (The P0 slot is P2 in the paper; by the symmetry F[P2 P3; P1 P4] = F[P1 P4; P2 P3] with P3 = P4 = e it may equally
    be put in the P1 slot, which is what the code does.)  Vectorised.
    """
    def __init__(self, m, n):
        g = gcd(2*m, n)
        self.b = np.sqrt(m/n)
        self.K = FusionKernel((2*m)//g, n//g)
        self.e = self.b/(2*np.sqrt(2))

    def S(self, Ps, Pt, P0):
        s2, e = np.sqrt(2), self.e
        Ps = np.asarray(Ps, dtype=complex)
        Pt = np.asarray(Pt, dtype=complex)
        return s2*256**(Ps**2 - Pt**2)*self.K.F(s2*Ps, s2*Pt, np.asarray(P0)/s2, e, e, e)

    def RS(self, Ps, Pt, P0):
        """VWR modular kernel RS_{Ps,Pt}[P0] = (Pt/Ps) S_{iPt,iPs}[iP0]."""
        Ps = np.asarray(Ps, dtype=complex)
        Pt = np.asarray(Pt, dtype=complex)
        return (Pt/Ps)*self.S(1j*Pt, 1j*Ps, 1j*np.asarray(P0))


class ModularKernelGeneral:
    """Modular kernel at arbitrary real b, from GeneralFusionKernel at b' = sqrt2 b (scalar)."""
    def __init__(self, b, **kw):
        self.b = float(b)
        self.K = GeneralFusionKernel(np.sqrt(2)*self.b, **kw)
        self.e = self.b/(2*np.sqrt(2))

    def S(self, Ps, Pt, P0):
        s2, e = np.sqrt(2), self.e
        return s2*256**(Ps**2 - Pt**2)*self.K.F(s2*Ps, s2*Pt, e, P0/s2, e, e)

    def RS(self, Ps, Pt, P0):
        return (Pt/Ps)*self.S(1j*Pt, 1j*Ps, 1j*P0)


def mod_kernel_tv(m, n, Ps, Pt, P0, ymax=14.0, ny=8001, c=0.0):
    """Teschner's Barnes integral for S_{Ps,Pt}[P0] at b^2 = m/n (validation only; imaginary momenta):

      S = rho0(Pt)/(2 S_b(Q/2 + P0)) prod_{+-} Gamma_b(Q +- 2Ps) Gamma_b(Q/2 - P0 +- 2Pt) / [Gamma_b(Q +- 2Pt) Gamma_b(Q/2 - P0 +- 2Ps)]
          * int_{c+iR} du/i e^{4 pi i Ps u} prod_{+-} S_b(u + Q/4 + (+-2Pt + P0)/2) / S_b(u + 3Q/4 + (+-2Pt - P0)/2).
    """
    rb = RationalB(m, n)
    Q, b = rb.Q, rb.b
    lpref = np.log(-4*np.sqrt(2)*np.sin(2*np.pi*b*Pt)*np.sin(2*np.pi*Pt/b)) - np.log(2) - rb.lS_b(Q/2 + P0)
    for sgn in (1, -1):
        lpref += (rb.lgamma_b(Q + sgn*2*Ps) - rb.lgamma_b(Q + sgn*2*Pt)
                  + rb.lgamma_b(Q/2 - P0 + sgn*2*Pt) - rb.lgamma_b(Q/2 - P0 + sgn*2*Ps))
    y = np.linspace(-ymax, ymax, ny)
    u = c + 1j*y
    lf = 4j*np.pi*Ps*u
    for sgn in (1, -1):
        lf = lf + rb.lS_b(u + Q/4 + 0.5*(sgn*2*Pt + P0)) - rb.lS_b(u + 3*Q/4 + 0.5*(sgn*2*Pt - P0))
    return np.exp(lpref)*integrate.simpson(np.exp(lf), x=y)


# ---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("1. c = 25 (b = 1), all externals -1/4 (source weight 15/16): F = -2^{4Pt^2-4Ps^2} (Pt/Ps) sin(2 pi Ps Pt)")
    K11 = FusionKernel(1, 1)
    for Ps, Pt in [(0.12j, 0.21j), (0.15 + 0.19j, 0.3j), (0.4, 0.27)]:
        num = K11.F(Ps, Pt, -0.25, -0.25, -0.25, -0.25)
        ex = -2**(-4*Ps**2 + 4*Pt**2)*(Pt/Ps)*np.sin(2*np.pi*Ps*Pt)
        print(f"   Ps={Ps}, Pt={Pt}: rel. error {abs(num - ex)/abs(ex):.1e}")
    print("2. c_hat = 1, all externals i/4 (h_hat = 1/16): RF = 2^{4Pt^2-4Ps^2} sin(2 pi Ps Pt)   [paper Eq. (3)]")
    for Ps, Pt in [(0.12j, 0.21j), (0.15 + 0.19j, 0.3j), (0.4, 0.27)]:
        rf = K11.RF(Ps, Pt, 0.25j, 0.25j, 0.25j, 0.25j)
        ex = 2**(-4*Ps**2 + 4*Pt**2)*np.sin(2*np.pi*Ps*Pt)
        print(f"   Ps={Ps}, Pt={Pt}: RF/exact = {rf/ex:.12f}")
    print("3. closed form vs Teschner-Vartanov integral (imaginary momenta):")
    for (m, n) in [(1, 1), (1, 2), (2, 3)]:
        K = FusionKernel(m, n)
        for args in [(0.13j, 0.29j, 0.07j, 0.11j, 0.17j, 0.23j), (0.31j, 0.12j, 0.21j, 0.05j, 0.16j, 0.27j)]:
            print(f"   b^2={m}/{n}: closed/TV = {K.F(*args)/tv_kernel(m, n, *args):.12f}")
    print("4. symmetries F[P2 P3;P1 P4] = F[P1 P4;P2 P3] = F[P3 P2;P4 P1], evenness (b^2 = 1/2):")
    K = FusionKernel(1, 2)
    Ps, Pt, P1, P2, P3, P4 = 0.13j, 0.29j + 0.05, 0.07j, 0.11j, 0.17j - 0.02, 0.23j
    a = K.F(Ps, Pt, P1, P2, P3, P4)
    for name, v in [("swap 1", K.F(Ps, Pt, P2, P1, P4, P3)), ("swap 2", K.F(Ps, Pt, P4, P3, P2, P1)),
                    ("Ps->-Ps", K.F(-Ps, Pt, P1, P2, P3, P4)), ("P3->-P3", K.F(Ps, Pt, P1, P2, -P3, P4))]:
        print(f"   {name}: rel. diff {abs(v - a)/abs(a):.1e}")
    print("5. modular kernel from F (Roussillon) vs Teschner's integral (b^2 = 1/2 and 1):")
    for (m, n) in [(1, 2), (1, 1)]:
        MK = ModularKernel(m, n)
        for (Ps, Pt, P0) in [(0.23j, 0.31j, 0.17j), (0.11j, 0.42j, 0.29j)]:
            print(f"   b^2={m}/{n}: S_F/S_Teschner = {MK.S(Ps, Pt, P0)/mod_kernel_tv(m, n, Ps, Pt, P0):.12f}")
    print("6. general-b kernel vs closed form at b^2 = 5/7, momenta with real parts (Route-2 regime):")
    K57 = FusionKernel(5, 7)
    G57 = GeneralFusionKernel(np.sqrt(5/7))
    u = np.array([0.13, 0.21, 0.17, 0.29]); v = np.array([0.19, 0.21, 0.24, 0.28])
    for y in (0.3, 1.7, 3.8):
        Ph = 0.15 + 1j*y
        args = (1j*Ph, -v[1], -u[0], -u[1], -v[2], -v[0])
        g = G57.F(*args); cf = K57.F(*args)
        print(f"   Phat = 0.15 + {y}i: rel. diff {abs(g - cf)/abs(cf):.1e}")
