import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eigvalsh, cholesky
from scipy.stats import linregress


# ============================================================
# Free scalar chain entanglement entropy benchmark
# ============================================================
#
# Hamiltonian:
#
#   H = 1/2 sum_i pi_i^2 + 1/2 sum_{i,j} phi_i K_ij phi_j
#
# with
#
#   K_ii     = m^2 + 2
#   K_i,i+1 = K_i+1,i = -1
#
# For periodic boundary conditions we also set
#
#   K_0,N-1 = K_N-1,0 = -1
#
# The massless c = 1 CFT benchmark is cleanest with periodic
# boundary conditions and very small m, used only to regulate
# the zero mode.
#
# Expected finite-size CFT form for periodic chain:
#
#   S(L) = (c/3) log[ (N/pi) sin(pi L/N) ] + s_a
#
# For a free scalar, c should be approximately 1.
# ============================================================


def Kbuilder(m: float, N: int, boundary: str = "periodic"):
    """
    Builds the lattice kernel K for a free scalar chain.

    Parameters
    ----------
    m:
        Dimensionless lattice mass.
        Use a very small nonzero value, e.g. m=1e-6, for the
        periodic massless benchmark to remove the zero mode.
    N:
        Number of lattice sites.
    boundary:
        "periodic" or "open".

    Returns
    -------
    K:
        NxN positive-definite matrix for m > 0.
    """
    if N <= 1:
        raise ValueError("N must be larger than 1.")

    if m < 0:
        raise ValueError("m must be non-negative.")

    if boundary not in {"periodic", "open"}:
        raise ValueError("boundary must be either 'periodic' or 'open'.")

    diag = (m**2 + 2.0) * np.ones(N)
    offdiag = -np.ones(N - 1)

    K = np.diag(diag)
    K += np.diag(offdiag, k=1)
    K += np.diag(offdiag, k=-1)

    if boundary == "periodic":
        K[0, -1] = -1.0
        K[-1, 0] = -1.0

    return K


def full_correlators(m: float, N: int, boundary: str = "periodic"):
    """
    Computes the ground-state correlators

        X = <phi phi> = 1/2 K^{-1/2}
        P = <pi pi>  = 1/2 K^{1/2}

    for a quadratic bosonic Hamiltonian.
    """
    K = Kbuilder(m=m, N=N, boundary=boundary)

    w, O = eigh(K)

    min_w = np.min(w)
    if min_w <= 0:
        raise ValueError(
            f"K is not positive definite. min eigenvalue = {min_w}. "
            "For periodic boundary conditions, use m > 0 to lift the zero mode."
        )

    sqrtK = (O * np.sqrt(w)) @ O.T
    invsqrtK = (O * (1.0 / np.sqrt(w))) @ O.T

    # Remove tiny numerical asymmetries.
    sqrtK = 0.5 * (sqrtK + sqrtK.T)
    invsqrtK = 0.5 * (invsqrtK + invsqrtK.T)

    X = 0.5 * invsqrtK
    P = 0.5 * sqrtK

    return X, P

def entropy_for_interval(left: int, right: int, X, P, clip_tol: float = 1e-8):
    """
    Entanglement entropy of the interval [left, right).

    The eigenvalues lambdas are nu_alpha^2 and should satisfy lambda >= 1/4.
    Tiny violations below 1/4 are numerical roundoff and are clipped.
    """
    N = X.shape[0]

    if left < 0 or right > N:
        raise ValueError("The interval is outside the chain.")

    if right <= left:
        raise ValueError("Need right > left.")

    XA = X[left:right, left:right]
    PA = P[left:right, left:right]

    R = cholesky(XA, lower=False)

    M = R @ PA @ R.T
    M = 0.5 * (M + M.T)

    lambdas = eigvalsh(M)

    min_lambda = float(np.min(lambdas))
    violation = 0.25 - min_lambda

    if violation > clip_tol:
        print(
            "Warning: sizeable lambda < 1/4 violation: "
            f"min(lambda) = {min_lambda}, violation = {violation}. "
            "Clipping anyway for this diagnostic benchmark."
        )

    lambdas = np.maximum(lambdas, 0.25)

    nu = np.sqrt(lambdas)

    x_plus = nu + 0.5
    x_minus = nu - 0.5

    entropy_terms = x_plus * np.log(x_plus)

    xminus_log_xminus = np.zeros_like(x_minus)
    positive = x_minus > 1e-14
    xminus_log_xminus[positive] = (
        x_minus[positive] * np.log(x_minus[positive])
    )

    entropy_terms -= xminus_log_xminus

    return float(np.sum(entropy_terms))

def entropy_for_centered_cut(L: int, X, P):
    """
    Entanglement entropy of a centered interval of length L.
    """
    N = X.shape[0]

    if L <= 0:
        raise ValueError("L must be positive.")

    if L >= N:
        raise ValueError("L must be smaller than N.")

    left = (N - L) // 2
    right = left + L

    return entropy_for_interval(left, right, X, P)


def entropy_as_function_of_cut(
    Lmin: int = 2,
    Lmax: int = 100,
    m: float = 1e-6,
    N: int = 500,
    boundary: str = "periodic",
):
    """
    Computes S(L) for centered intervals of length L.
    """
    if Lmin <= 0:
        raise ValueError("Lmin must be positive.")

    if Lmax >= N:
        raise ValueError("Lmax must be smaller than N.")

    if Lmax < Lmin:
        raise ValueError("Need Lmax >= Lmin.")

    X, P = full_correlators(m=m, N=N, boundary=boundary)

    L_values = np.arange(Lmin, Lmax + 1)
    S_values = np.array([entropy_for_centered_cut(L, X, P) for L in L_values])

    return L_values, S_values


def fit_log_entropy_infinite_line(
    L_values,
    S_values,
    fit_L_min: int | None = None,
    fit_L_max: int | None = None,
):
    """
    Fits

        S(L) = (c/3) log(L) + s_a.

    This is only appropriate when L << N and finite-size effects are negligible.
    """
    L_values = np.asarray(L_values)
    S_values = np.asarray(S_values)

    mask = np.isfinite(S_values)
    mask &= L_values > 0

    if fit_L_min is not None:
        mask &= L_values >= fit_L_min

    if fit_L_max is not None:
        mask &= L_values <= fit_L_max

    if np.sum(mask) < 3:
        raise ValueError("Not enough points in the fitting window.")

    x = np.log(L_values[mask])
    y = S_values[mask]

    result = linregress(x, y)

    return {
        "c_fit": 3.0 * result.slope,
        "s_a_fit": result.intercept,
        "c_error": 3.0 * result.stderr,
        "s_a_error": result.intercept_stderr,
        "slope": result.slope,
        "intercept": result.intercept,
        "r_value": result.rvalue,
        "p_value": result.pvalue,
        "r_squared": result.rvalue**2,
        "mask": mask,
        "x_fit": x,
        "S_fit_data": y,
    }


def fit_log_entropy_periodic_chain(
    L_values,
    S_values,
    N: int,
    fit_L_min: int | None = None,
    fit_L_max: int | None = None,
):
    """
    Fits the periodic finite-size CFT formula

        S(L) = (c/3) log[ (N/pi) sin(pi L/N) ] + s_a.

    This is the recommended benchmark for extracting c = 1.
    """
    L_values = np.asarray(L_values)
    S_values = np.asarray(S_values)

    mask = np.isfinite(S_values)
    mask &= L_values > 0
    mask &= L_values < N

    if fit_L_min is not None:
        mask &= L_values >= fit_L_min

    if fit_L_max is not None:
        mask &= L_values <= fit_L_max

    if np.sum(mask) < 3:
        raise ValueError("Not enough points in the fitting window.")

    x = np.log((N / np.pi) * np.sin(np.pi * L_values[mask] / N))
    y = S_values[mask]

    result = linregress(x, y)

    return {
        "c_fit": 3.0 * result.slope,
        "s_a_fit": result.intercept,
        "c_error": 3.0 * result.stderr,
        "s_a_error": result.intercept_stderr,
        "slope": result.slope,
        "intercept": result.intercept,
        "r_value": result.rvalue,
        "p_value": result.pvalue,
        "r_squared": result.rvalue**2,
        "mask": mask,
        "x_fit": x,
        "S_fit_data": y,
    }


def local_effective_c_periodic(L_values, S_values, N: int):
    """
    Computes a local effective central charge using the finite-size variable

        x(L) = log[ (N/pi) sin(pi L/N) ]

    and

        c_eff(L) = 3 dS/dx.
    """
    L_values = np.asarray(L_values)
    S_values = np.asarray(S_values)

    x = np.log((N / np.pi) * np.sin(np.pi * L_values / N))
    dS_dx = np.gradient(S_values, x)

    return L_values, 3.0 * dS_dx


def plot_entropy_with_fit(
    L_values,
    S_values,
    fit,
    N: int | None = None,
    fit_type: str = "periodic",
    title: str | None = None,
):
    """
    Plots S(L) and the fitted curve.

    fit_type:
        "periodic" uses x = log[(N/pi) sin(pi L/N)].
        "infinite" uses x = log(L).
    """
    L_values = np.asarray(L_values)
    S_values = np.asarray(S_values)

    if fit_type == "periodic":
        if N is None:
            raise ValueError("N must be supplied for periodic fit plot.")
        x_all = np.log((N / np.pi) * np.sin(np.pi * L_values / N))
    elif fit_type == "infinite":
        x_all = np.log(L_values)
    else:
        raise ValueError("fit_type must be 'periodic' or 'infinite'.")

    S_fit_all = fit["slope"] * x_all + fit["intercept"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(L_values, S_values, label="Numerical EE")

    mask = fit["mask"]
    ax.plot(
        L_values[mask],
        S_fit_all[mask],
        "--",
        label=rf"Fit: $c={fit['c_fit']:.6f}$",
    )

    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$S_{\mathrm{VN}}(L)$")

    if title is None:
        title = "Free scalar EE benchmark"

    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    return fig, ax


def plot_local_effective_c(L_values, S_values, N: int):
    """
    Plots c_eff(L) for the periodic finite-size variable.
    """
    L, c_eff = local_effective_c_periodic(L_values, S_values, N)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(L, c_eff)
    ax.axhline(1.0, linestyle="--", label=r"$c=1$")

    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$c_{\mathrm{eff}}(L)$")
    ax.set_title("Local effective central charge")
    ax.legend()
    fig.tight_layout()

    return fig, ax


if __name__ == "__main__":
    # Recommended clean benchmark:
    # periodic chain, tiny nonzero mass to regulate the zero mode.
    N = 500
    m = 1e-6
    boundary = "periodic"

    Lmin = 2
    Lmax = 100

    fit_L_min = 10
    fit_L_max = 80

    L_values, S_values = entropy_as_function_of_cut(
        Lmin=Lmin,
        Lmax=Lmax,
        m=m,
        N=N,
        boundary=boundary,
    )

    fit = fit_log_entropy_periodic_chain(
        L_values,
        S_values,
        N=N,
        fit_L_min=fit_L_min,
        fit_L_max=fit_L_max,
    )

    print("Periodic finite-size CFT fit")
    print("--------------------------------")
    print("c =", fit["c_fit"], "+/-", fit["c_error"])
    print("s_a =", fit["s_a_fit"], "+/-", fit["s_a_error"])
    print("r^2 =", fit["r_squared"])

    fig, ax = plot_entropy_with_fit(
        L_values,
        S_values,
        fit,
        N=N,
        fit_type="infinite",
        title=rf"Free scalar EE, periodic chain, $m={m}$, $N={N}$",
    )

    fig2, ax2 = plot_local_effective_c(L_values, S_values, N=N)

    plt.show()
