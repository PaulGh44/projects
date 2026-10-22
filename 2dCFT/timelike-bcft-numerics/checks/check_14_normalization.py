"""
CHECK 14 - the normalization of the boundary data: the free constants, how they scale, and the final choice (Sec. S3).
======================================================================================================================

Relative to the "literal rotation" (legs rho0(Pa) -> Pa, kappa = 1/2, (Chat) = Shat RF) multiply the one-point function,
bulk-boundary coefficient, two- and three-point coefficients by (k1, k2, k3, k4) and the closed and open measures by
c_cl and 1/c_op.  Then (paper "The constants left free, and how they are fixed"):
   boundary crossing and the disc V Psi Psi hold for all values;  the cylinder requires k4/(c_op k3) = c_cl k1 k2;
   the disc two-point function requires c_cl k1 = k2^2/(c_op k3);  the annulus is multiplied by c_cl k1^2;
   the disc V V Psi: bulk channel ~ c_cl k2, boundary channel ~ k2^2 k4/(c_op^2 k3^2);
   a rescaling Psi -> lambda Psi of the boundary fields multiplies (k2, k3, k4) by (lambda, lambda^2, lambda^3);
   the invariant (S17)  [cyl closed/cyl open] [disc bdy/disc bulk] = k2^3/k4  must equal one.
With c_cl = c_op = 1 the general solution is, modulo lambda, (k, 1, 1/k, 1); the literal rotation gives bulk = 2 x boundary
in the disc V V Psi (check_07), hence c_cl k1^2 = 2.  Table I takes (k1, k2, k3, k4) = (sqrt2, sqrt2, sqrt2, 2 sqrt2).

What is verified (pointwise, on random complex momenta, bhat^2 = 1/2):
 1. every sewing density of Table I divided by its literal-rotation counterpart equals the factor claimed in Sec. S3:
    four-point 4 sqrt2, cylinder (open and closed) 2, disc two-point sqrt2, disc V Psi Psi 2 sqrt2, disc V V Psi bulk sqrt2
    and boundary 2 sqrt2, annulus 2;
 2. for random (k1, k2, k3, k4, c_cl, c_op) every density scales as stated, so that the conditions above are exactly the
    conditions for the sewing constraints to survive the rescaling, and the invariant (S17) equals k2^3/k4;
 3. the equivalent "odd-pairing" form of the data, Eq. (S18) (even R', S', C' with explicit label factors), gives the
    same densities as Table I (paper: to 2e-15).

Usage:  python checks/check_14_normalization.py      (~1 s)
"""
import numpy as np

from _common import Report, options
from tlbcft.theory import Timelike

r2 = np.sqrt(2)
u = 1j*np.array([0.13, 0.21, 0.17, 0.29])
v = 1j*np.array([0.19, 0.21, 0.24, 0.28])
Pq, Pq2, P0 = 0.31 + 0.47j, 0.23 - 0.36j, 0.17j


def densities(D, c_cl=1.0, c_op=1.0):
    """The sewing densities of Eqs. (4), (S15), (S26), at fixed generic momenta; closed measure c_cl rhohat dP/i,
    open measure dP/(i c_op Shat)."""
    S = lambda *a: c_op*D.Sh(*a)
    rho = lambda P: c_cl*D.rhoh(P)
    out = {}
    out['4pt'] = D.Ch3(Pq, u[1], u[0], v[2], v[1], v[0])*D.Ch3(u[3], u[2], Pq, v[3], v[2], v[0])/S(Pq, v[0], v[2])
    out['cyl open'] = D.Ch3(Pq, P0, Pq, v[0], v[0], v[1])/S(Pq, v[1], v[0])
    out['cyl closed'] = rho(Pq)*D.Rh2(Pq, P0, v[0])*D.Rh1(Pq, v[1])
    out['disc bulk'] = rho(Pq)*D.Ch(Pq, u[0], u[1])*D.Rh1(Pq, v[0])
    out['disc bdy'] = D.Rh2(u[0], Pq, v[0])*D.Rh2(u[1], Pq, v[0])/S(Pq, v[0], v[0])
    out['VPsiPsi'] = D.Rh2(u[0], Pq, v[1])*D.Ch3(Pq, u[1], u[2], v[1], v[0], v[1])/S(Pq, v[1], v[1])
    out['VVPsi bulk'] = rho(Pq)*D.Ch(u[0], u[1], Pq)*D.Rh2(Pq, u[2], v[0])
    out['VVPsi bdy'] = (D.Rh2(u[0], Pq, v[0])*D.Rh2(u[1], Pq2, v[0])*D.Ch3(u[2], Pq2, Pq, v[0], v[0], v[0])
                        / (S(Pq, v[0], v[0])*S(Pq2, v[0], v[0])))
    out['annulus'] = rho(Pq)*D.Rh1(Pq, v[0])*D.Rh1(Pq, v[1])
    return out


def main():
    options(__doc__)
    R = Report(__file__, "normalization: the free constants and Table I", "Sec. S3 'The constants left free', Eqs. (S17)-(S18)")
    Tf = Timelike((1, 2))
    Tl = Timelike((1, 2), normalization='literal')
    df, dl = densities(Tf), densities(Tl)
    R.section("1. Table I / literal rotation, pointwise")
    claimed = {'4pt': 4*r2, 'cyl open': 2, 'cyl closed': 2, 'disc bulk': r2, 'disc bdy': r2, 'VPsiPsi': 2*r2,
               'VVPsi bulk': r2, 'VVPsi bdy': 2*r2, 'annulus': 2}
    for k in claimed:
        R.ratio(k, df[k], dl[k], claimed[k], 1e-13)

    R.section("2. general rescaling (k1, k2, k3, k4, c_cl, c_op): scaling laws and the invariant (S17)")
    rng = np.random.default_rng(11)
    for trial in range(3):
        k1, k2, k3, k4, ccl, cop = 0.5 + 2*rng.random(6)
        D = Timelike((1, 2), scales=(k1, k2, k3, k4))
        d = densities(D, c_cl=ccl, c_op=cop)
        law = {'4pt': k4**2/(cop*k3), 'cyl open': k4/(cop*k3), 'cyl closed': ccl*k1*k2, 'disc bulk': ccl*k1,
               'disc bdy': k2**2/(cop*k3), 'VPsiPsi': k2*k4/(cop*k3), 'VVPsi bulk': ccl*k2,
               'VVPsi bdy': k2**2*k4/(cop**2*k3**2), 'annulus': ccl*k1**2}
        for k in law:
            R.ratio(f"#{trial} {k}", d[k], dl[k], law[k], 1e-13)
        inv = (d['cyl closed']/d['cyl open'])*(d['disc bdy']/d['disc bulk'])
        inv_l = (dl['cyl closed']/dl['cyl open'])*(dl['disc bdy']/dl['disc bulk'])
        R.ratio(f"#{trial} invariant (S17) relative to the literal rotation = k2^3/k4", inv, inv_l, k2**3/k4, 1e-13)
    R.note("Consequences: boundary crossing (4pt: k4^2/(c_op k3) in both channels) and V Psi Psi (k2 k4/(c_op k3) in both")
    R.note("channels) are invariant constraints; the cylinder needs k4/(c_op k3) = c_cl k1 k2, the disc two-point function")
    R.note("c_cl k1 = k2^2/(c_op k3).  With c_cl = c_op = 1 and lambda removed: (k, 1, 1/k, 1); the literal ratio")
    R.note("bulk/boundary = 2 of the disc V V Psi then gives c_cl k1^2 = 2 (check_07).")
    for k in (np.sqrt(2), 1.7):
        lam = 1.3
        D = Timelike((1, 2), scales=(k*1.0, lam, lam**2/k, lam**3))
        d = densities(D)
        R.ratio(f"family (k, 1, 1/k, 1) x (1, lam, lam^2, lam^3), k = {k:.4f}: cylinder open/closed relative to literal",
                d['cyl open']/d['cyl closed'], dl['cyl open']/dl['cyl closed'], 1.0, 1e-13)
        R.ratio(f"   disc bulk/bdy relative to literal", d['disc bulk']/d['disc bdy'], dl['disc bulk']/dl['disc bdy'], 1.0, 1e-13)
        R.ratio(f"   V V Psi bulk/bdy relative to literal = 1/k^2", d['VVPsi bulk']/d['VVPsi bdy'], dl['VVPsi bulk']/dl['VVPsi bdy'],
                1/k**2, 1e-13)
        R.ratio(f"   annulus relative to literal = k^2", d['annulus'], dl['annulus'], k**2, 1e-13)

    R.section("3. the odd-pairing form of the data, Eq. (S18)")
    T = Tf
    Sop = lambda P, a, b: 2*r2*T.S_even(P, a, b)/(a*b*P)
    C3 = lambda P3, P2, P1, t3, t2, t1: T.C3_even(P3, P2, P1, t3, t2, t1)/(t1*t2*t3)
    odd = {'4pt': C3(Pq, u[1], u[0], v[2], v[1], v[0])*C3(u[3], u[2], Pq, v[3], v[2], v[0])/Sop(Pq, v[0], v[2]),
           'disc bulk': T.rhoh(Pq)*T.Ch(Pq, u[0], u[1])*T.R1_even(Pq, v[0])/Pq,
           'disc bdy': (T.R2_even(u[0], Pq, v[0])/v[0])*(T.R2_even(u[1], Pq, v[0])/v[0])/Sop(Pq, v[0], v[0]),
           'cyl closed': T.rhoh(Pq)*(T.R2_even(Pq, P0, v[0])/v[0])*T.R1_even(Pq, v[1])/Pq}
    for k, val in odd.items():
        R.ratio(f"{k}: odd-pairing form / Table I", val, df[k], 1.0, 1e-12)
    R.done()


if __name__ == "__main__":
    main()
