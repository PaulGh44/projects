r"""
The bulk and boundary data of Liouville theory: c >= 25 (baseline) and c <= 1 (this work, paper Table I).
=========================================================================================================

This module is the single place where the data of Table I are defined.  Every check script imports them from here.

Notation (paper, "Kernels and the Virasoro-Wick rotation" and Sec. S1)
---------------------------------------------------------------------
  c <= 1:  b = i bhat, bhat > 0,  chat = 1 - 6 (bhat - 1/bhat)^2,  hhat(P) = (chat - 1)/24 - P^2,
           spectrum P in eps + iR (eps > 0).  Degenerate momenta P_(r,s) = (i/2)(r bhat - s/bhat);
           weight zero at 0hat = (i/2)(bhat - 1/bhat), weight one at (i/2)(bhat + 1/bhat).
  "Source" functions rho0, C0, F, S are the c >= 25 functions WITH PARAMETER bhat (Q = bhat + 1/bhat),
  evaluated at rotated arguments (i P).

c >= 25 data (Table I, middle column; class Spacelike), spectrum P in iR:
  rho0(P)                 = -4 sqrt2 sin(2 pi b P) sin(2 pi P/b)          bulk measure rho0 dP/(2i)
  C0(P1,P2,P3)            = Gamma_b(2Q)/(sqrt2 Gamma_b(Q)^3) prod_{+-+-+-} Gamma_b(Q/2 +- P1 +- P2 +- P3)
                            / prod_j Gamma_b(Q +- 2 P_j)                     (DOZZ, paper Eq. (S3))
  S0(P; s1, s2)           = 1/(rho0(P) C0(P, s1, s2))                       boundary two-point
  (C0)^{s3 s2 s1}_{P3 P2 P1} = 2 S0(P3; s1, s3) F_{s2,P3}[P2 P1; s3 s1]        boundary three-point
  R0(Pa, Pb | s)          = 2 S_{s,Pa}[Pb]/rho0(Pa)                         bulk-boundary
  R0(Pa | s)              = 2 sqrt2 cos(4 pi s Pa)/rho0(Pa)                 one-point (FZZT)

c <= 1 data (Table I, right column; class Timelike), spectrum P in eps + iR:
  rhoh(P)                 = P^2/(2 rho0(iP))                                 bulk measure rhoh dP/i
  Ch(P1,P2,P3)            = 1/C0(iP1, iP2, iP3)                              bulk three-point (timelike DOZZ)
  Sh(P; s1, s2)           = 2 sqrt2 rho0(i s1) rho0(i s2) C0(iP, i s1, i s2)/(s1 s2 P)      two-point; open measure dP/(i Sh)
  (Ch)^{s3 s2 s1}_{P3 P2 P1} = 2 Sh(P3; s1, s3) RF_{s2,P3}[P2 P1; s3 s1]                     three-point
  Rh(Pa, Pb | s)          = 2 sqrt2 RS_{s,Pa}[Pb]/Pa                                         bulk-boundary
  Rh(Pa | s)              = 4 cos(4 pi s Pa)/Pa                                              one-point (boundary state rel. to rhoh)
  Rh(Pa, 0hat | s)        = 4 sin(4 pi s Pa)/Pa                                              coefficient of Psi_0 (weight zero)

Normalisation.  The rotation leaves one constant free (paper, "The constants left free"): relative to the
"literal rotation" (legs rho0(Pa) -> Pa, kappa = 1/2) the one-point function, bulk-boundary coefficient, two- and
three-point coefficients are multiplied by (k1, k2, k3, k4).  Table I is
      normalization='final'   (k1, k2, k3, k4) = (sqrt2, sqrt2, sqrt2, 2 sqrt2),
and the literal rotation is normalization='literal', (1, 1, 1, 1).  Any other tuple may be passed as `scales`.
In the final normalization (Ch)/Sh = 2 RF and every VWR ratio enters as P/sqrt2.
"""
import numpy as np

from .special import RationalB, GeneralB
from .kernels import FusionKernel, ModularKernel, GeneralFusionKernel, ModularKernelGeneral

SQRT2 = np.sqrt(2.0)
NORMALIZATIONS = {'final': (SQRT2, SQRT2, SQRT2, 2*SQRT2), 'literal': (1.0, 1.0, 1.0, 1.0)}


class _Source:
    """c >= 25 functions with parameter b (rho0, C0, F, S) shared by Spacelike and Timelike."""
    def _init_source(self, b2):
        if isinstance(b2, tuple):                       # rational b^2 = m/n: closed-form kernels
            m, n = b2
            self.m, self.n = m, n
            self.gb = RationalB(m, n)
            self.K = FusionKernel(m, n)
            self.MK = ModularKernel(m, n)
            self.vectorised = True
        else:                                           # arbitrary b^2: Barnes integrals (scalar kernels)
            self.m = self.n = None
            self.gb = GeneralB(np.sqrt(b2))
            self.K = GeneralFusionKernel(np.sqrt(b2))
            self.MK = ModularKernelGeneral(np.sqrt(b2))
            self.vectorised = False
        self.b = self.gb.b
        self.Q = self.gb.Q
        Q = self.Q
        self._lC0const = complex(np.asarray(self.gb.lgamma_b(np.array(2*Q)) - np.log(SQRT2)
                                            - 3*self.gb.lgamma_b(np.array(Q))))

    def _vec(self, f, *args):
        if self.vectorised:
            return f(*args)
        return np.vectorize(lambda *a: complex(f(*a)), otypes=[complex])(*args)

    # ---- source bulk data
    def rho0(self, P):
        """rho0(P) = -4 sqrt2 sin(2 pi b P) sin(2 pi P/b)."""
        P = np.asarray(P, dtype=complex)
        return -4*SQRT2*np.sin(2*np.pi*self.b*P)*np.sin(2*np.pi*P/self.b)

    def lC0(self, P1, P2, P3):
        """log of the DOZZ coefficient C0(P1, P2, P3) (paper Eq. (S3))."""
        P1, P2, P3 = np.broadcast_arrays(*[np.asarray(x, dtype=complex) for x in (P1, P2, P3)])
        Q, lg = self.Q, self.gb.lgamma_b
        out = self._lC0const + np.zeros(P1.shape, dtype=complex)
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    out = out + lg(Q/2 + s1*P1 + s2*P2 + s3*P3)
        for P in (P1, P2, P3):
            for s in (1, -1):
                out = out - lg(Q + 2*s*P)
        return out

    def C0(self, P1, P2, P3):
        return np.exp(self.lC0(P1, P2, P3))

    # ---- source kernels
    def F(self, Ps, Pt, P1, P2, P3, P4):
        """F_{Ps,Pt}[P2 P3; P1 P4]."""
        return self._vec(self.K.F, Ps, Pt, P1, P2, P3, P4)

    def Fm(self, Ps, Pt, A, B, C, D):
        """F_{Ps,Pt}[A B; C D]."""
        return self.F(Ps, Pt, C, A, B, D)

    def S(self, Ps, Pt, P0):
        """S_{Ps,Pt}[P0]."""
        return self._vec(self.MK.S, Ps, Pt, P0)


class Spacelike(_Source):
    """c >= 25 Liouville boundary data in the bootstrap normalization (Table I, middle column).

    Spacelike((m, n)) uses the closed-form kernels at b^2 = m/n; Spacelike(b2) with a float uses Barnes integrals.
    """
    def __init__(self, b2):
        self._init_source(b2)
        self.c = 1 + 6*self.Q**2

    def h(self, P):
        return self.Q**2/4 - np.asarray(P, dtype=complex)**2

    def S0(self, P, s1, s2):
        return 1/(self.rho0(P)*self.C0(P, s1, s2))

    def C3(self, P3, P2, P1, s3, s2, s1):
        """(C0)^{s3 s2 s1}_{P3 P2 P1} = 2 S0(P3; s1, s3) F_{s2,P3}[P2 P1; s3 s1]."""
        return 2*self.S0(P3, s1, s3)*self.Fm(s2, P3, P2, P1, s3, s1)

    def R2(self, Pa, Pb, s):
        """R0(Pa, Pb | s) = 2 S_{s,Pa}[Pb]/rho0(Pa)."""
        return 2*self.S(s, Pa, Pb)/self.rho0(Pa)

    def R1(self, Pa, s):
        """R0(Pa | s) = 2 sqrt2 cos(4 pi s Pa)/rho0(Pa)."""
        return 2*SQRT2*np.cos(4*np.pi*s*np.asarray(Pa))/self.rho0(Pa)


class Timelike(_Source):
    """c <= 1 Liouville bulk and boundary data (Table I, right column).

    Timelike((m, n))              bhat^2 = m/n, closed-form kernels (vectorised)
    Timelike(0.7071...)           bhat^2 given as a float: Barnes integrals (slow, scalar kernels)
    normalization='final'         Table I;  'literal' = literal rotation (k = 1);  or scales=(k1, k2, k3, k4).
    """
    def __init__(self, bh2, normalization='final', scales=None):
        self._init_source(bh2)
        self.bh = self.b
        self.chat = 1 - 6*(self.bh - 1/self.bh)**2
        self.normalization = normalization if scales is None else 'custom'
        self.k1, self.k2, self.k3, self.k4 = NORMALIZATIONS[normalization] if scales is None else scales
        self.zero = 0.5j*(self.bh - 1/self.bh)          # 0hat: chat-weight zero
        self.one = 0.5j*(self.bh + 1/self.bh)           # chat-weight one (image of the c >= 25 identity)

    # ---- weights and special momenta
    def hh(self, P):
        """hhat(P) = (chat - 1)/24 - P^2."""
        return (self.chat - 1)/24 - np.asarray(P, dtype=complex)**2

    def Pdeg(self, r, s):
        """P_(r,s) = (i/2)(r bhat - s/bhat); P_(r,-s) = (i/2)(r bhat + s/bhat) are the null-descendant momenta."""
        return 0.5j*(r*self.bh - s/self.bh)

    # ---- VWR kernels, paper Eq. (2)
    def RF(self, Ps, Pt, P1, P2, P3, P4):
        """RF_{Ps,Pt}[P2 P3; P1 P4] = (Pt/Ps) F_{iPt,iPs}[iP2 iP1; iP3 iP4]."""
        return self._vec(self.K.RF, Ps, Pt, P1, P2, P3, P4)

    def RFm(self, Ps, Pt, A, B, C, D):
        """RF_{Ps,Pt}[A B; C D]."""
        return self.RF(Ps, Pt, C, A, B, D)

    def RS(self, Ps, Pt, P0):
        """RS_{Ps,Pt}[P0] = (Pt/Ps) S_{iPt,iPs}[iP0]."""
        return self._vec(self.MK.RS, Ps, Pt, P0)

    # ---- bulk data
    def rhoh(self, P):
        """rhohat(P) = P^2/(2 rho0(iP))."""
        P = np.asarray(P, dtype=complex)
        return P**2/(2*self.rho0(1j*P))

    def Ch(self, P1, P2, P3):
        """Chat(P1, P2, P3) = 1/C0(iP1, iP2, iP3)."""
        return np.exp(-self.lC0(1j*np.asarray(P1), 1j*np.asarray(P2), 1j*np.asarray(P3)))

    # ---- boundary data (Table I)
    def Sh(self, P, s1, s2):
        """Shat(P; s1, s2) = k3 * 2 rho0(i s1) rho0(i s2) C0(iP, i s1, i s2)/(s1 s2 P)."""
        P = np.asarray(P, dtype=complex)
        return self.k3*2*self.rho0(1j*s1)*self.rho0(1j*s2)*self.C0(1j*P, 1j*s1, 1j*s2)/(s1*s2*P)

    def Ch3(self, P3, P2, P1, s3, s2, s1):
        """(Chat)^{s3 s2 s1}_{P3 P2 P1} = k4 * Shat_literal(P3; s1, s3) RF_{s2,P3}[P2 P1; s3 s1]
        (final normalization: 2 Shat(P3; s1, s3) RF_{s2,P3}[P2 P1; s3 s1])."""
        return (self.k4/self.k3)*self.Sh(P3, s1, s3)*self.RFm(s2, P3, P2, P1, s3, s1)

    def Rh2(self, Pa, Pb, s):
        """Rhat(Pa, Pb | s) = k2 * 2 RS_{s,Pa}[Pb]/Pa   (final: 2 sqrt2 RS/Pa)."""
        Pa = np.asarray(Pa, dtype=complex)
        return self.k2*2*self.RS(s, Pa, Pb)/Pa

    def Rh1(self, Pa, s):
        """Rhat(Pa | s) = k1 * 2 sqrt2 cos(4 pi s Pa)/Pa   (final: 4 cos/Pa).  Boundary state relative to rhohat."""
        Pa = np.asarray(Pa, dtype=complex)
        return self.k1*2*SQRT2*np.cos(4*np.pi*s*Pa)/Pa

    def Rh_psi0(self, Pa, s):
        """Coefficient of the weight-zero boundary field Psi_0:  Rhat(Pa, 0hat | s) = k2 * 2 sqrt2 sin(4 pi s Pa)/Pa
        (final: 4 sin/Pa), the closed form of k2 * 2 RS_{s,Pa}[0hat]/Pa (paper Eq. (S20))."""
        Pa = np.asarray(Pa, dtype=complex)
        return self.k2*2*SQRT2*np.sin(4*np.pi*s*Pa)/Pa

    # ---- the even ("odd-pairing") form of the data, paper Eq. (S18)  [final normalization only]
    def R1_even(self, Pa, s):
        """R'(Pa | s) = 4 cos(4 pi s Pa)  [Rhat(Pa|s) = R'/Pa]."""
        return 4*np.cos(4*np.pi*s*np.asarray(Pa, dtype=complex))

    def R2_even(self, Pa, Pb, s):
        """R'(Pa, Pb | s) = 2 sqrt2 S_{iPa, is}[iPb]  [Rhat(Pa,Pb|s) = R'/s]."""
        return 2*SQRT2*self.S(1j*np.asarray(Pa), 1j*s, 1j*np.asarray(Pb))

    def S_even(self, P, s1, s2):
        """S'(P; s1, s2) = rho0(i s1) rho0(i s2) C0(iP, i s1, i s2)  [Shat = 2 sqrt2 S'/(s1 s2 P)]."""
        return self.rho0(1j*s1)*self.rho0(1j*s2)*self.C0(1j*np.asarray(P), 1j*s1, 1j*s2)

    def C3_even(self, P3, P2, P1, s3, s2, s1):
        """C' = 4 sqrt2 S'(P3; s1, s3) F_{iP3, i s2}[iP2 i s3; iP1 i s1]  [(Chat) = C'/(s1 s2 s3)]."""
        return 4*SQRT2*self.S_even(P3, s1, s3)*self.Fm(1j*np.asarray(P3), 1j*s2, 1j*np.asarray(P2), 1j*s3,
                                                         1j*np.asarray(P1), 1j*s1)
