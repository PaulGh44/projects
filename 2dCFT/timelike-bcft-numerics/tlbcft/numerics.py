r"""
Quadrature on the physical contour, circle residues, and small reporting helpers.
================================================================================

All sewing tests are line integrals on the physical contour P = eps + i y, y in [-Y, Y], with
    int_{eps + iR} dP/i f(P) = int dy f(eps + i y)
(paper Sec. S8, "Methods").  The integrands decay like Gaussians, so a cut at |y| = Y ~ 4.5 is exact to double
precision (the scripts print the size of the integrand at the ends).  Two rules are used, as in the paper:
Simpson's rule on an odd number of points (four-point, cylinder, disc two-point) and the midpoint rule (discs with
three insertions).
"""
import numpy as np
from scipy import integrate


def line(eps, Y=4.5, npts=1201):
    """Grid y in [-Y, Y] (npts points, endpoints included) and P = eps + i y."""
    y = np.linspace(-Y, Y, npts)
    return y, eps + 1j*y


def midpoint(eps, Y=4.5, N=800):
    """Midpoint grid: y_j = -Y + (j + 1/2) 2Y/N, P = eps + i y, weight 2Y/N."""
    h = 2*Y/N
    y = -Y + (np.arange(N) + 0.5)*h
    return y, eps + 1j*y, h


def simpson(f, y):
    """Simpson's rule for int dy f(y)."""
    return integrate.simpson(np.asarray(f), x=y)


def circle_residue(f, z0, rad=0.02, npts=16):
    """Res_{z0} f by the trapezoidal rule on a circle (exponentially accurate for meromorphic f).
    f is called on an array of points if possible, else point by point."""
    th = 2*np.pi*(np.arange(npts) + 0.5)/npts
    pts = z0 + rad*np.exp(1j*th)
    try:
        vals = np.asarray(f(pts), dtype=complex)
        if vals.shape != pts.shape:
            raise ValueError
    except Exception:
        vals = np.array([f(p) for p in pts], dtype=complex)
    return rad*np.mean(vals*np.exp(1j*th))


def rel(a, b):
    """|a - b|/|a|."""
    return abs(a - b)/abs(a)
