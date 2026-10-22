"""
CHECK 02 - transported kernel identities on the real momentum line (paper Table II, row "real-line identities").
==============================================================================================================

The Virasoro-Wick rotation maps every c >= 25 kernel identity to an identity of RF, RS integrated over the REAL line
(not over the physical contour eps + iR).  These are the identities from which the data of Table I are derived:

 1. Boundary crossing = the rotated pentagon (Sec. S3, "Gluing and boundary crossing"):
        int_R dPs RF_{s2,Ps}[P2 P1; s3 s1] RF_{s3,P4}[P3 Ps; s4 s1] RF_{Ps,Pt}[P2 P3; P1 P4]
            = RF_{s2,P4}[Pt P1; s4 s1] RF_{s3,Pt}[P3 P2; s4 s2],
    the VWR ratios cancel; pointwise, bhat^2 = 1/2 (paper: 7e-12).
 2. The disc two-point function, Eq. (S16), with the data of Table I:
        int_R dP rhohat(P) Chat(P,P1,P2) Rhat(P|s) RF_{P,Pb}[P1 P1; P2 P2] = Rhat(P1,Pb|s) Rhat(P2,Pb|s)/Shat(Pb;s,s);
    holds iff kappa = 1/sqrt2.  Paper: P1,2 = 0.13, 0.21, s = 0.19, Pb = 0.27, 0.53: both sides 3.308295403120 and
    10.980925470946; with kappa four times larger the left side is 4.0000000000 times the right.
 3. The discs with three insertions, transported (Sec. S6): the c >= 25 identities (S28) and (S31) with F -> RF,
    S -> RS, literal-rotation data, momenta on the real line, and the braiding phases transported as the SOURCE phase
    at h = 1 - hhat ('thm'); with the naive replacement h -> hhat ('nat') the two sides differ by a constant phase:
       disc V Psi Psi   ('thm' holds to ~1e-6;  'nat': ratio -1)
       disc V V Psi     ('thm' holds to ~1e-10; 'nat': ratio -i)
    The second integration variable runs on a slightly tilted line t e^{+-i 0.15} (as in the paper's runs).

Usage:  python checks/check_02_real_line_identities.py          (items 1-2 and the V V Psi transport, ~1.5 min)
        python checks/check_02_real_line_identities.py --full   (also the V Psi Psi transport, ~2 min)
"""
import time

import numpy as np
from scipy import integrate

from _common import Report, options
from tlbcft.theory import Timelike


def pentagon(R, T):
    P1, P2, P3, P4 = 0.13, 0.21, 0.17, 0.29
    s1, s2, s3, s4 = 0.19, 0.21, 0.24, 0.28
    for Pt in (0.37, 0.61):
        yy = np.linspace(-5.0, 5.0, 2400)
        f = T.RFm(s2, yy, P2, P1, s3, s1)*T.RFm(s3, P4, P3, yy, s4, s1)*T.RF(yy, Pt, P1, P2, P3, P4)
        lhs = integrate.simpson(f, x=yy)
        rhs = T.RFm(s2, P4, Pt, P1, s4, s1)*T.RFm(s3, Pt, P3, P2, s4, s2)
        R.agree(f"rotated pentagon, Pt = {Pt} (|integrand| at +-5: {abs(f[0]):.0e})", lhs, rhs, 1e-10)


def disc(R, T):
    P1, P2, s = 0.13, 0.21, 0.19
    for Pb, printed in ((0.27, "3.308295403120"), (0.53, "10.980925470946")):
        yy = np.linspace(-5.0, 5.0, 2400)
        f = T.rhoh(yy)*T.Ch(yy, P1, P2)*T.Rh1(yy, s)*T.RFm(yy, Pb, P1, P1, P2, P2)
        lhs = integrate.simpson(f, x=yy)
        rhs = T.Rh2(P1, Pb, s)*T.Rh2(P2, Pb, s)/T.Sh(Pb, s, s)
        R.agree(f"disc two-point identity (S16), Pb = {Pb}", lhs, rhs, 1e-10, names=("bulk", "boundary"))
        # printed to 13 digits; the closed-form kernels are good to ~1e-12 (check_00), hence a relative 2e-12
        R.paper(f"   boundary side (no quadrature), Pb = {Pb}", rhs, printed, rtol=2e-12)
        R.ratio(f"   with kappa four times larger (Shat x 4): bulk/boundary", lhs, rhs/4, 4.0, 1e-9)


def vvpsi_transport(R, conv, m=1, n=2, TP=5.0, NP=200, Tc=6.0, Nc=240, th=0.15):
    """Paper Eq. (S31) transported:  int dP dc rhohat Chat Rhat(P,Pb|s) RF_{P,c}[P1 Pb; P2 P] Om(c)
       RF_{P,P''}[c P1; P2 P2] RF_{c,P'}[P1 P1; Pb P''] = Phi Rhat(P1,P'|s) Rhat(P2,P''|s) (Chat)^{sss}_{Pb P'' P'}/(Shat Shat),
       literal-rotation data; 'thm': Om = -e^{-i pi (hh_c - hh_1 - hh_b)} (= e^{i pi (h_c - h_1 - h_b)} at h = 1 - hh),
       Phi = e^{-i pi (1 - hh' + hh'' - hh_b)/2};  'nat': Om = e^{i pi (hh_c - hh_1 - hh_b)}, Phi = e^{-i pi (hh'-hh''+hh_b)/2}."""
    T = Timelike((m, n), normalization='literal')
    P1, P2, Pb, s = 0.13, 0.21, 0.17, 0.19
    th = abs(th) if conv == 'thm' else -abs(th)
    hP = 2*TP/NP
    P = -TP + (np.arange(NP) + 0.5)*hP
    hc = 2*Tc/Nc
    c = (-Tc + (np.arange(Nc) + 0.5)*hc)*np.exp(1j*th)
    hh = T.hh
    B = T.rhoh(P)*T.Ch(P1, P2, P)*T.Rh2(P, Pb, s)
    Fpc = T.RFm(P[:, None], c[None, :], P1, Pb, P2, P[:, None])
    if conv == 'nat':
        Om = np.exp(1j*np.pi*(hh(c) - hh(P1) - hh(Pb)))
    else:
        Om = -np.exp(-1j*np.pi*(hh(c) - hh(P1) - hh(Pb)))
    expected = 1.0 if conv == 'thm' else -1j
    for (Pp, Ppp) in ((0.27, 0.31), (0.40, 0.31), (0.31, 0.45)):
        Fppp = T.RFm(P[:, None], Ppp, c[None, :], P1, P2, P2)
        Fcp = T.RFm(c, Pp, P1, P1, Pb, Ppp)
        G = np.nan_to_num(B[:, None]*Fpc*Fppp*Om[None, :]*Fcp[None, :])
        lhs = hP*hc*np.exp(1j*th)*np.sum(G)
        rhs = T.Rh2(P1, Pp, s)*T.Rh2(P2, Ppp, s)*T.Ch3(Pb, Ppp, Pp, s, s, s)/(T.Sh(Pp, s, s)*T.Sh(Ppp, s, s))
        if conv == 'nat':
            Phi = np.exp(-1j*np.pi*(hh(Pp) - hh(Ppp) + hh(Pb))/2)
        else:
            Phi = np.exp(-1j*np.pi*(1 - hh(Pp) + hh(Ppp) - hh(Pb))/2)
        R.ratio(f"V V Psi transport [{conv}], (P', P'') = ({Pp}, {Ppp}): lhs/(Phi rhs)", lhs, rhs*Phi, expected,
                1e-8 if conv == 'thm' else 1e-6)


def vpsipsi_transport(R, conv, T_=5.0, N=200, th4=0.15):
    """Paper Eq. (S28) transported, literal-rotation data, bhat^2 = 1/2:
       Rhat(P2,P5|P') RF_{P1,P5}[P0 P0'; P' P'] = int dP3 dP4 Rhat(P2,P3|P1) RF_{P',P3}[P0' P0; P1 P1] Om
                                                   RF_{P3,P4}[P0' P2; P0 P2] RF_{P4,P5}[P0' P0; P2 P2],
       Om = e^{i pi (2h4 - 2h2 - h0 - h0' + h3/2 + h5/2)} with h = 1 - hh ('thm') or h = hh ('nat')."""
    T = Timelike((1, 2), normalization='literal')
    P1, Pp, P2, P0, P0p, P5 = 0.19, 0.23, 0.13, 0.21, 0.17, 0.27
    hh = T.hh
    th4 = abs(th4) if conv == 'thm' else -abs(th4)
    hs = 2*T_/N
    P3 = -T_ + (np.arange(N) + 0.5)*hs
    P4 = (-T_ + (np.arange(N) + 0.5)*hs)*np.exp(1j*th4)
    A3 = T.Rh2(P2, P3, P1)*T.RFm(Pp, P3, P0p, P0, P1, P1)
    A4 = T.RFm(P4, P5, P0p, P0, P2, P2)
    M = T.RFm(P3[:, None], P4[None, :], P0p, P2, P0, P2)
    h = (lambda x: 1 - hh(x)) if conv == 'thm' else hh
    ph = np.exp(1j*np.pi*(2*h(P4)[None, :] - 2*h(P2) - h(P0) - h(P0p) + h(P3)[:, None]/2 + h(P5)/2))
    rhs = hs*hs*np.exp(1j*th4)*np.sum(np.nan_to_num(A3[:, None]*M*A4[None, :]*ph))
    lhs = T.Rh2(P2, P5, Pp)*T.RFm(P1, P5, P0, P0p, Pp, Pp)
    R.ratio(f"V Psi Psi transport [{conv}]: rhs/lhs", rhs, lhs, 1.0 if conv == 'thm' else -1.0, 1e-5)


def main():
    A = options(__doc__)
    R = Report(__file__, "real-line (transported) kernel identities", "Table II; Secs. S3 and S6")
    T = Timelike((1, 2))
    R.section("1. rotated pentagon (boundary crossing on the real line), bhat^2 = 1/2")
    pentagon(R, T)
    R.section("2. rotated disc two-point identity (S16) with the data of Table I, bhat^2 = 1/2")
    disc(R, T)
    R.section("3a. disc V V Psi: transport of the c >= 25 identity (S31), bhat^2 = 1/2")
    for conv in ('thm', 'nat'):
        t0 = time.time()
        vvpsi_transport(R, conv)
        R.note(f"[{time.time() - t0:.0f} s]")
    if A.full:
        R.section("3b. disc V Psi Psi: transport of the c >= 25 identity (S28), bhat^2 = 1/2")
        for conv in ('thm', 'nat'):
            t0 = time.time()
            vpsipsi_transport(R, conv)
            R.note(f"[{time.time() - t0:.0f} s]")
    R.done()


if __name__ == "__main__":
    main()
