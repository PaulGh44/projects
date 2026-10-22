r"""
Five-point comb blocks for the disc with two bulk fields and one boundary field (paper Sec. S6).
================================================================================================

The disc  < V_{P1}(z1) V_{P2}(z2) Psi_{Pbeta}(x) >  (z1, z2 in the upper half plane, x on the real line) doubles to a
chiral five-point function with operators at z1, z2, x, zb1, zb2 (zb = complex conjugate).  Its two channels are
expanded in five-point "comb" blocks:

  bulk channel      (z1 z2) -> P  ;  x  ;  P <- (zb1 zb2)      [bulk OPE, then the bulk field P hits the boundary]
  boundary channel  (z1 zb1) -> P' ;  x  ;  P'' <- (z2 zb2)   [each bulk field goes to the boundary separately]

Symmetric frame.  For each channel a Moebius map M takes the involution exchanging both fused pairs to w -> -w:
the first pair sits at -d, +d, the boundary point at t, the second pair at -R, +R with d R = 1, and the comb block is

   B = <0| V5(R) V4(-R) Pi_b V3(t) Pi_a V_R1(d) V_L1(-d) |0>
     = f_b * sum_{I,J} (-1)^{|J|} E_J t^{h_b + |J| - h3 - h_a - |I|} M_{JI} C_I,

   C = G_a^{-1} c,  c_I = <a; I| V_R1(d) V_L1(-d) |0> = (2d)^{h_a - hL - hR} d^{|I|} chat_I(h_a; hL, hR),
   E = G_b^{-1} e,  e_J = (1/R)^{|J|} chat_J(h_b; h5, h4),
   M_{JI} = <b; J| V3(1) |a; I>  (Ward identity),     f_b = <0| V5(R) V4(-R) V_b(0) |0>,

a double sum over the two intermediate Verma modules truncated at level Nmax.  At z_{1,2} = -+1 + i, x = 0 both channels
converge as (sqrt2 - 1)^N.

Normalisation of the partial waves (paper Eq. (S30)).  The frame factors are multivalued, so each partial wave W is
normalised by continuing all logarithms along a path to its OPE limit and matching the physical leading behaviour
   bulk:      W ~ |z12|^{2(h-h1-h2)} |z2 - zb2|^{hb - 2h} |z2 - x|^{-2 hb}                  (z1 -> z2),
   boundary:  W ~ (2y1)^{h'-2h1} (2y2)^{h''-2h2} |x2-x1|^{hb-h'-h''} / (|x-x1|^{h'-h''+hb} |x2-x|^{h''-h'+hb})   (y -> 0),
for z_k = x_k + i y_k, x1 < x < x2.  Paths: z_{1,2} = -+ s a + i y (bulk) and -+ a + i s y (boundary), s: 1 -> 0.

Main entry point:  partial_waves(channel, c, h1, h2, hbt, ha, hb, Nmax, a, y, x)  -> W for paired arrays (ha, hb).
"""
from math import factorial

import numpy as np

from .blocks import VermaModule, parts_of


# ---------------------------------------------------------------------------------------------
# comb blocks in the symmetric frame
# ---------------------------------------------------------------------------------------------
def _gamma(lam, h, ha, hb):
    """<h; lam| V_hb(1) |ha>  (h may be an array)."""
    val = 1.0 + 0j
    tail = 0
    for i in range(len(lam) - 1, -1, -1):
        val = val*(h + tail + lam[i]*hb - ha)
        tail += lam[i]
    return val


def pair_coeffs(V, hL, hR):
    """chat_I(h; hL, hR) per level N: dict N -> array (K, p(N)), defined by
       <h; I| V_R(d) V_L(-d) |0> = (2d)^{h - hL - hR} d^{|I|} chat_I = <h; I| e^{-d L_{-1}} V_R(2d) |hL>."""
    out = {}
    for N in range(V.Nmax + 1):
        P = parts_of(N)
        arr = np.zeros((V.K, len(P)), dtype=complex)
        for i, I in enumerate(P):
            state = {I: np.ones(V.K, dtype=complex)}
            tot = np.zeros(V.K, dtype=complex)
            for k in range(N + 1):
                s = sum(cf*_gamma(K, V.h, hL, hR) for K, cf in state.items()) if state else 0
                tot = tot + ((-1)**k*2.0**(N - k)/factorial(k))*s
                if k < N:
                    new = {}
                    for p, cf in state.items():
                        for p2, cf2 in V.apply_L(1, p).items():
                            new[p2] = new.get(p2, 0) + cf*cf2
                    state = new
            arr[:, i] = tot
        out[N] = arr
    return out


class _Vertex:
    """M_{JI} = <h_b; J| V3(1) |h_a; I> for paired arrays (h_a, h_b)."""
    def __init__(self, Va, hb, h3):
        self.Va = Va
        self.hb = np.atleast_1d(np.asarray(hb, dtype=complex))
        self.h3 = h3
        self.ha = Va.h
        self._M = {}

    def M(self, J, I):
        key = (J, I)
        if key in self._M:
            return self._M[key]
        if not J:
            v = _gamma(I, self.ha, self.hb, self.h3)*np.ones(self.Va.K, dtype=complex)
        else:
            J1, Jp = J[0], J[1:]
            v = (self.hb + sum(Jp) - self.ha - sum(I) + J1*self.h3)*self.M(Jp, I)
            for K, cf in self.Va.apply_L(J1, I).items():
                v = v + cf*self.M(Jp, K)
        self._M[key] = v
        return v


def comb_block(c, ha, hb, hL, hR, h3, h4, h5, d, t, dp, Nmax):
    """sum_{I,J} (-1)^{|J|} E_J t^{h_b+|J|-h3-h_a-|I|} M_{JI} C_I, without f_b and (2d)^{h_a-hL-hR}.
       ha, hb: arrays of equal length (paired internal weights)."""
    ha = np.atleast_1d(np.asarray(ha, dtype=complex))
    hb = np.atleast_1d(np.asarray(hb, dtype=complex))
    Va = VermaModule(c, ha, Nmax, symmetric_fill=True)
    Vb = VermaModule(c, hb, Nmax, symmetric_fill=True)
    ca, cb = pair_coeffs(Va, hL, hR), pair_coeffs(Vb, h5, h4)
    Cv = {N: np.einsum('kij,kj->ki', Va.Ginv[N], ca[N])*d**N for N in range(Nmax + 1)}
    Ev = {N: np.einsum('kij,kj->ki', Vb.Ginv[N], cb[N])*(-dp)**N for N in range(Nmax + 1)}
    Vx = _Vertex(Va, hb, h3)
    tot = np.zeros(Va.K, dtype=complex)
    for NJ in range(Nmax + 1):
        PJ = parts_of(NJ)
        for NI in range(Nmax + 1):
            PI = parts_of(NI)
            Mm = np.zeros((Va.K, len(PJ), len(PI)), dtype=complex)
            for j, J in enumerate(PJ):
                for i, I in enumerate(PI):
                    Mm[:, j, i] = Vx.M(J, I)
            tot = tot + np.einsum('kj,kji,ki->k', Ev[NJ], Mm, Cv[NI])*t**(NJ - NI)
    return tot*t**(hb - h3 - ha)


# ---------------------------------------------------------------------------------------------
# geometry: Moebius map to the symmetric frame
# ---------------------------------------------------------------------------------------------
def _involution(a1, a2, b1, b2):
    """Moebius involution iota(w) = (A w + B)/(C w - A) with iota(a1) = a2, iota(b1) = b2."""
    M = np.array([[a1 + a2, 1.0, -a1*a2], [b1 + b2, 1.0, -b1*b2]], dtype=complex)
    _, _, vh = np.linalg.svd(M)
    A, B, C = vh[-1].conj()
    return A, B, C


def sym_frame(pair_in, vertex, pair_out):
    """Images of the five points in the symmetric frame, and M, M'."""
    wL, wR = pair_in
    wo1, wo2 = pair_out
    A, B, C = _involution(wL, wR, wo1, wo2)
    disc = np.sqrt(A*A + B*C + 0j)
    p, q = (A + disc)/C, (A - disc)/C                     # fixed points of the involution

    def m(w, p=p, q=q):
        return (w - p)/(w - q)
    if abs(m(wL)) > abs(m(wo1)):                          # the in-pair must be inside
        p, q = q, p
    mR, mo = (wR - p)/(wR - q), (wo1 - p)/(wo1 - q)
    kap = np.exp(-1j*np.angle(mR))/np.sqrt(abs(mR)*abs(mo))   # d = kap m(wR) > 0 and d R = 1
    M = lambda w: kap*(w - p)/(w - q)
    dM = lambda w: kap*(p - q)/(w - q)**2
    return dict(p=p, q=q, kap=kap, d=M(wR), dL=M(wL), o1=M(wo1), o2=M(wo2), t=M(vertex), M=M, dM=dM)


def configs(channel, z1, z2, x):
    zb1, zb2 = np.conj(z1), np.conj(z2)
    if channel == 'bulk':
        F = sym_frame((z2, z1), x, (zb1, zb2))
        F['pts'] = dict(L=z2, R=z1, o1=zb1, o2=zb2, x=x)
    else:
        F = sym_frame((z1, zb1), x, (z2, zb2))
        F['pts'] = dict(L=z1, R=zb1, o1=z2, o2=zb2, x=x)
    return F


# ---------------------------------------------------------------------------------------------
# physical partial waves
# ---------------------------------------------------------------------------------------------
def _frame_logs(channel, z1, z2, x):
    F = configs(channel, z1, z2, x)
    P = F['pts']
    lM = {k: np.log(F['dM'](P[k])) for k in ['L', 'R', 'x', 'o1', 'o2']}
    if F['o1'].real > 0:
        v5c, v4c = 'o1', 'o2'
    else:
        v5c, v4c = 'o2', 'o1'
    return F, lM, v4c, v5c, abs(F['o1'])


def _lead_log(channel, z1, z2, x, h1, h2, hbt, ha, hb):
    if channel == 'bulk':
        h = ha
        return (2*(h - h1 - h2)*np.log(abs(z1 - z2)) + (hbt - 2*h)*np.log(abs(z2 - np.conj(z2)))
                - 2*hbt*np.log(abs(z2 - x)))
    hp, hpp = ha, hb
    y1, y2, x1, x2 = z1.imag, z2.imag, z1.real, z2.real
    return ((hp - 2*h1)*np.log(2*y1) + (hpp - 2*h2)*np.log(2*y2) + (hpp - hp - hbt)*np.log(abs(x - x1))
            + (hp - hpp - hbt)*np.log(abs(x2 - x)) + (hbt - hp - hpp)*np.log(abs(x2 - x1)))


def _weights(channel, h1, h2, hbt):
    if channel == 'bulk':
        return dict(L=h2, R=h1, x=hbt, o1=h1, o2=h2)
    return dict(L=h1, R=h1, x=hbt, o1=h2, o2=h2)


def _path_point(channel, s, a=1.0, y=1.0, x=0.0):
    if channel == 'bulk':
        return -s*a + 1j*y, s*a + 1j*y, x          # s -> 0: z1 -> z2
    return -a + 1j*s*y, a + 1j*s*y, x              # s -> 0: both bulk fields to the boundary


def _normalisation(channel, h1, h2, hbt, ha, hb, a=1.0, y=1.0, x=0.0):
    """log N = lim_{s->0} [log W_lead - log F_level0], all logarithms continued from s = 1 along the path
    (Richardson extrapolation on the last points)."""
    w = _weights(channel, h1, h2, hbt)
    ss = np.concatenate([np.linspace(1.0, 0.02, 200), 0.02*2.0**(-np.arange(1, 12))])
    traj = {k: [] for k in ['L', 'R', 'x', 'o1', 'o2']}
    lt, ld, lR, v5s = [], [], [], []
    for s in ss:
        z1, z2, xx = _path_point(channel, s, a, y, x)
        F, lM, v4c, v5c, Rs = _frame_logs(channel, z1, z2, xx)
        for k in traj:
            traj[k].append(lM[k])
        lt.append(np.log(F['t'])); ld.append(np.log(2*F['d'].real)); lR.append(Rs); v5s.append(v5c)
    assert len(set(v5s)) == 1, "out-pair assignment changes along the path"
    v5c = v5s[0]
    v4c = 'o2' if v5c == 'o1' else 'o1'
    for k in traj:
        arr = np.array(traj[k])
        traj[k] = arr.real + 1j*np.unwrap(arr.imag)
    lt = np.array(lt)
    lt = lt.real + 1j*np.unwrap(lt.imag)

    def lF(i, ha, hb):
        out = sum(w[k]*traj[k][i] for k in traj)
        Rs = lR[i]
        h4c, h5c = w[v4c], w[v5c]
        lfb = (hb - h5c - h4c)*np.log(2*Rs) + (h4c - h5c - hb)*np.log(Rs) + (h5c - h4c - hb)*(np.log(Rs) + 1j*np.pi)
        return out + lfb + (ha - w['L'] - w['R'])*ld[i] + (hb - w['x'] - ha)*lt[i]

    D = []
    for i in range(len(ss) - 6, len(ss)):
        z1, z2, xx = _path_point(channel, ss[i], a, y, x)
        D.append(_lead_log(channel, z1, z2, xx, h1, h2, hbt, ha, hb) - lF(i, ha, hb))
    D = np.array(D)
    sv = ss[-6:]
    V = np.vander(sv, 3, increasing=True)
    coef = np.linalg.lstsq(V, D.reshape(len(sv), -1), rcond=None)[0]
    return coef[0].reshape(np.shape(D[0])), lF(0, ha, hb)


def partial_waves(channel, c, h1, h2, hbt, ha, hb, Nmax, a=1.0, y=1.0, x=0.0):
    """Physical partial waves of the disc <V_{h1}(z1) V_{h2}(z2) Psi_{hbt}(x)> at z_{1,2} = -+a + i y, boundary point x.

    channel 'bulk': ha = hb = the internal bulk weight h (pass the same array twice);
    channel 'bdy' : ha = h' (array), hb = h'' (array of the same length).
    """
    z1, z2, xx = _path_point(channel, 1.0, a, y, x)
    F, lM, v4c, v5c, Rs = _frame_logs(channel, z1, z2, xx)
    w = _weights(channel, h1, h2, hbt)
    d, t = F['d'].real, F['t']
    tot = comb_block(c, ha, hb, w['L'], w['R'], w['x'], w[v4c], w[v5c], d, t, 1/Rs, Nmax)
    logN, lF0 = _normalisation(channel, h1, h2, hbt, ha, hb, a, y, x)
    return np.exp(logN + lF0 - (hb - w['x'] - ha)*np.log(t))*tot
