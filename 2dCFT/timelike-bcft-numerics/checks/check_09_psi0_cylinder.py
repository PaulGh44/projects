"""
CHECK 09 - the cylinder with a Psi_0 insertion: an exact statement for every bhat (Sec. S4 "The Psi_0 cylinder").
=================================================================================================================

At the weight-zero insertion P0 = 0hat the torus one-point blocks are characters chi_P = q^{-P^2}/eta(q), and the trace
kernel of the cylinder (the open-channel density (Chat)/Shat = 2 RF) is elementary, paper Eq. (S23):

  RF_{s_a,P}[0hat P; s_a s_b] = int_R dp sin(4 pi P p) sin(4 pi s_a p) cos(4 pi s_b p) / (sinh(2 pi bhat p) sinh(2 pi p/bhat))
                              = -(1/pi) sum_{r,s odd} sum_{e,e' = +-} e lambda_rs / (lambda_rs^2 + 4 (P + e s_a + e' s_b)^2),
  lambda_rs = r bhat + s/bhat   (the integral for |Im P| < (bhat + 1/bhat)/2 - |v_a| - |v_b|, the sum everywhere).

The double sum converges slowly; here the sum over s is done in closed form with the digamma function,
    sum_{s odd >= 1} [1/(A + s/bhat) - 1/(B + s/bhat)] = (bhat/2) [psi((B bhat + 1)/2) - psi((A bhat + 1)/2)],
and the sum over r is cut at R, 2R, 4R and Richardson-extrapolated (error ~1e-9).

The kernel has poles exactly on the label-shifted lattice P = i(e_a v_a + e_b v_b + lambda_rs/2) (and reflections) with
residues e_a m_rs/(4 pi i), and nothing else.  Since the density is odd, the open channel on eps + iR is the residue
sum 2 pi sum_{Im P > 0} Res, i.e.   eta(q) Z_open = -i sum_{r,s odd} sum_{e_a, e_b} e_a q^{(lambda_rs/2 + e_a v_a + e_b v_b)^2},
and modular covariance against the closed channel  int dP/i rhohat(P) Rhat(P,0hat|s_a) Rhat(P|s_b) chi_P(qt)  is
Eq. (S21) read backwards: a theorem for every bhat.

What is verified:
 1. the three forms of the trace kernel (closed-form F at bhat^2 = 1/2 with the external weight-zero momentum reached
    by symmetric displacement; the Fourier integral; the lattice sum) agree;
 2. residues of the closed-form kernel at the label-shifted points, e_a/(4 pi i), including the DOUBLED residue at
    bhat + 3/bhat = 5 bhat + 1/bhat (bhat^2 = 1/2), and absence of poles at the unshifted Kac momenta;
 3. the open channel as a line integral on eps + iR (lattice-sum kernel) = its residue sum = the closed channel
    (line integral of the elementary closed density), for rational and irrational bhat.

Usage:  python checks/check_09_psi0_cylinder.py      (~20 s)
"""
import numpy as np
from scipy import integrate
from scipy.special import digamma

from _common import Report, options
from tlbcft.theory import Timelike
from tlbcft.numerics import circle_residue


def _lattice(P, bh, sa, sb, R):
    """-(1/pi) sum_{r odd < R} sum_{s odd} sum_{e,e'} e lam/(lam^2 + 4 x^2), x = P + e sa + e' sb (s-sum via digamma)."""
    P = np.asarray(P, dtype=complex)
    r = np.arange(1, R, 2)
    out = np.zeros(P.shape, dtype=complex)
    for e in (1, -1):
        for e2 in (1, -1):
            x = P[..., None] + e*sa + e2*sb
            for pm in (1, -1):                               # lam/(lam^2 + 4x^2) = (1/2) sum_pm 1/(lam +- 2 i x)
                out = out + e*0.5*(-bh/2)*np.sum(digamma((bh/2)*(r*bh + 1/bh + pm*2j*x)), axis=-1)
    return -out/np.pi


def trace_lattice(P, bh, sa, sb, R=400):
    a, b, c = _lattice(P, bh, sa, sb, R), _lattice(P, bh, sa, sb, 2*R), _lattice(P, bh, sa, sb, 4*R)
    return (4*(2*c - b) - (2*b - a))/3


def trace_integral(P, bh, sa, sb):
    f = lambda p: np.sin(4*np.pi*P*p)*np.sin(4*np.pi*sa*p)*np.cos(4*np.pi*sb*p)/(np.sinh(2*np.pi*bh*p)*np.sinh(2*np.pi*p/bh))
    re = integrate.quad(lambda p: f(p).real, 1e-12, 14, limit=500, epsabs=1e-15)[0]
    im = integrate.quad(lambda p: f(p).imag, 1e-12, 14, limit=500, epsabs=1e-15)[0]
    return 2*(re + 1j*im)


def main():
    options(__doc__)
    R = Report(__file__, "the Psi_0 cylinder: elementary trace kernel, residues, modular covariance",
               "Sec. S4 'The Psi_0 cylinder', Eq. (S23); Table II row 'cylinder, Psi_0'")
    T = Timelike((1, 2))
    bh = T.bh
    sa, sb = 0.19j, 0.24j

    def closed_form(P):        # RF_{sa,P}[0hat P; sa sb] with 0hat in an external slot: symmetric displacement average
        return np.mean([T.RFm(sa, P, T.zero + d, P, sa, sb) for d in (1e-4, -1e-4, 1e-4j, -1e-4j)], axis=0)

    R.section("1. three forms of the trace kernel, bhat^2 = 1/2, s_a = 0.19i, s_b = 0.24i")
    for P in (0.15 + 0.1j, 0.15 + 0.3j, 0.3 - 0.2j, 0.05 + 0.45j):
        b = trace_integral(P, bh, sa, sb)
        R.agree(f"P = {P}: closed-form kernel vs Fourier integral", closed_form(P), b, 1e-7, names=("closed", "integral"))
        R.agree(f"P = {P}: lattice sum vs Fourier integral", trace_lattice(P, bh, sa, sb), b, 1e-8, names=("lattice", "integral"))

    R.section("2. residues of the closed-form kernel on the label-shifted lattice: m_rs e_a/(4 pi i)")
    va, vb = 0.19, 0.24
    kern = lambda p: np.mean([T.RFm(sa, p, T.zero + d, p, sa, sb) for d in (1e-5, -1e-5)], axis=0)
    for (r, s, mult) in [(1, 1, 1), (3, 1, 1), (1, 3, 2)]:       # bhat + 3/bhat = 5 bhat + 1/bhat at bhat^2 = 1/2: m = 2
        for ea in (1, -1):
            for eb in (1, -1):
                Pst = 1j*(ea*va + eb*vb + 0.5*(r*bh + s/bh))
                res = circle_residue(kern, Pst, rad=0.01, npts=32)
                R.agree(f"(r,s) = ({r},{s}), signs ({ea:+d},{eb:+d}): residue", res, mult*ea/(4j*np.pi), 1e-5,
                        names=("computed", f"{mult} e_a/(4 pi i)"))
    for (r, s) in [(1, 1), (2, 1), (1, 2)]:
        Pk = 0.5j*(r*bh + s/bh)
        res = circle_residue(lambda p: p*kern(p), Pk, rad=0.01, npts=32)
        R.small(f"no pole at the Kac momentum P_(r,-s), (r,s) = ({r},{s}): |residue of P*kernel|", res, 1e-7)

    R.section("3. Psi_0 cylinder: open channel (line integral) = residue sum = closed channel, tau = i T")
    for (bh_, lab) in [(1/np.sqrt(2), "bhat^2 = 1/2"), (1.0, "bhat = 1"), (0.7, "bhat^2 = 0.49"), (2**-0.25, "bhat^2 = 1/sqrt2")]:
        for T_ in (0.8, 1.3):
            q = np.exp(-2*np.pi*T_)
            y = np.linspace(-4, 4, 2001)
            P = 0.15 + 1j*y
            Zo_line = integrate.simpson(2*trace_lattice(P, bh_, sa, sb, R=200)*q**(-P**2), x=y)
            rr = np.arange(1, 400, 2)
            lam = (rr[:, None]*bh_ + rr[None, :]/bh_).ravel()
            Zo_res = -1j*sum(ea*np.sum(q**((lam/2 + ea*va + eb*vb)**2)) for ea in (1, -1) for eb in (1, -1))
            yc = np.linspace(-9, 9, 120001)
            Pc = 0.15 + 1j*yc
            rho0 = lambda x: -4*np.sqrt(2)*np.sin(2*np.pi*bh_*x)*np.sin(2*np.pi*x/bh_)
            # rhohat(P) Rhat(P, 0hat|s_a) Rhat(P|s_b) = P^2/(2 rho0(iP)) * 4 sin(4 pi s_a P)/P * 4 cos(4 pi s_b P)/P
            dens_c = Pc**2/(2*rho0(1j*Pc))*(4*np.sin(4*np.pi*sa*Pc)/Pc)*(4*np.cos(4*np.pi*sb*Pc)/Pc)
            Zc = integrate.simpson(dens_c*np.exp(2*np.pi*Pc**2/T_), x=yc)/np.sqrt(T_)
            R.agree(f"{lab}, T = {T_}: open line integral = residue sum", Zo_line, Zo_res, 1e-7, names=("line", "residues"))
            R.agree(f"{lab}, T = {T_}: residue sum (open) = closed channel", Zo_res, Zc, 1e-10, names=("open", "closed"))
    R.done()


if __name__ == "__main__":
    main()
