import numpy as np
from scipy.linalg import eigh, eigvalsh, cholesky
import matplotlib.pyplot as plt
from scipy.stats import linregress


def Kbuilder(m: float, N: int):
    diag = (m**2 + 2.0) * np.ones(N)
    offdiag = -np.ones(N - 1)

    K = np.diag(diag)
    K += np.diag(offdiag, k=1)
    K += np.diag(offdiag, k=-1)

    return K


def full_correlators(m: float, N: int):
    K = Kbuilder(m, N)

    # K = O diag(w) O.T
    w, O = eigh(K)

    if np.min(w) <= 0:
        raise ValueError("K is not positive definite.")

    sqrtK = (O * np.sqrt(w)) @ O.T
    invsqrtK = (O * (1.0 / np.sqrt(w))) @ O.T

    # Symmetrize to remove tiny round-off asymmetries
    sqrtK = 0.5 * (sqrtK + sqrtK.T)
    invsqrtK = 0.5 * (invsqrtK + invsqrtK.T)

    X = 0.5 * invsqrtK
    P = 0.5 * sqrtK

    return X, P


def entropy_for_cut(L: int, X, P):
    N = X.shape[0]

    left = (N - L) // 2
    right = left + L

    if left < 0 or right > N:
        raise ValueError("The cut is too large.")

    XA = X[left:right, left:right]
    PA = P[left:right, left:right]

    # Cholesky: XA = R.T @ R
    R = cholesky(XA, lower=False)

    # Symmetric matrix similar to XA @ PA
    M = R @ PA @ R.T
    #M should be symmetric but to remove tiny round-off asymmetries
    M = 0.5 * (M + M.T)

    lambdas = eigvalsh(M)

    # These are nu_alpha^2
    tol = 1e-12
    if np.min(lambdas) < 0.25 - tol:
        raise ValueError(
            f"Eigenvalue too small: min(lambda) = {np.min(lambdas)}"
        )

    # Clip only tiny numerical violations of the bound
    lambdas = np.maximum(lambdas, 0.25)

    nu = np.sqrt(lambdas)

    # Entropy formula
    x_plus = nu + 0.5
    x_minus = nu - 0.5

    # Define x log x -> 0 as x -> 0
    entropy_terms = x_plus * np.log(x_plus)
    entropy_terms -= np.where(
        x_minus > 1e-14,
        x_minus * np.log(x_minus),
        0.0
    )

    return np.sum(entropy_terms)


def entropy_as_function_of_cut(Lmin=1, Lmax=100, m=0, N=1000):
    X, P = full_correlators(m, N)

    L_values = np.arange(Lmin, Lmax + 1)
    S_values = np.array([entropy_for_cut(L, X, P) for L in L_values])

    return L_values, S_values



def plot_entropy(Lmin=1, Lmax=50, m=0.5, N=1000):
    L_values, S_values = entropy_as_function_of_cut(Lmin, Lmax, m, N)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(L_values, S_values)
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$S_{\mathrm{VN}}(L)$")
    ax.set_title(f"EE as a function of the cut size for a free scalar field in 1+1d with m = {m}")
    fig.tight_layout()

    return fig, ax

def fit_cft_entropy_finite_chain(L_values, S_values, N, fit_L_min=None, fit_L_max=None):
    """
    Fits

        S(L) = (c/3) log[(N/pi) sin(pi L/N)] + s_a

    for a block of length L in the middle of a finite chain.
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

    x = np.log((N / np.pi) * np.sin(np.pi * L_values[mask] / N))
    y = S_values[mask]

    result = linregress(x, y)

    slope = result.slope
    intercept = result.intercept

    return {
        "c_fit": 3.0 * slope,
        "s_a_fit": intercept,
        "c_error": 3.0 * result.stderr,
        "s_a_error": result.intercept_stderr,
        "slope": slope,
        "intercept": intercept,
        "r_value": result.rvalue,
        "p_value": result.pvalue,
        "mask": mask,
        "x_fit": x,
        "S_fit_data": y,
    }


if __name__ == "__main__":
    N = 2000
    m = 1e-8

    L_values, S_values = entropy_as_function_of_cut(
        Lmin=2,
        Lmax=300,
        m=m,
        N=N,
    )

    fit = fit_cft_entropy_finite_chain(
        L_values,
        S_values,
        N=N,
        fit_L_min=10,
        fit_L_max=200,
    )

    print("c =", fit["c_fit"], "+/-", fit["c_error"])
    print("s_a =", fit["s_a_fit"], "+/-", fit["s_a_error"])
    print("r^2 =", fit["r_value"]**2)