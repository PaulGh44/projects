import numpy as np
from scipy.linalg import eigh, eigvalsh, cholesky
import matplotlib.pyplot as plt


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


def entropy_as_function_of_cut(Lmin=1, Lmax=100, m=0.5, N=1000):
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


if __name__ == "__main__":
    fig, ax = plot_entropy(Lmin=1, Lmax=50, m=0, N=1000)
    plt.show()