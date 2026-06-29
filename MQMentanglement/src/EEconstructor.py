import numpy as np
from scipy.linalg import eigh, eigvalsh, cholesky



# ============================================================
# Liouville minisuperspace EE with explicit lattice spacing
# ============================================================
#
# Continuum mass profile:
#
#     m(phi)^2 = 4*pi*mu*exp(2*b*phi)
#
# Lattice coordinate:
#
#     phi_n = phi_min + n*a
#
# After dropping the global 1/a^2 prefactor in the Hamiltonian,
# the lattice kernel is
#
#     K_nn     = 2 + a^2*m(phi_n)^2
#              = 2 + 4*pi*mu*a^2*exp(2*b*phi_n)
#
#     K_n,n+1 = K_n+1,n = -1
#
# The global prefactor does not affect the entanglement entropy.
# ============================================================

class eEconstructor:
    
    @staticmethod
    def phi_grid(phi_min: float, phi_max: float, a: float):
        """
        Builds the Liouville lattice:

            phi_n = phi_min + n*a

        The endpoint phi_max is included up to rounding.
        """
        if a <= 0:
            raise ValueError("The lattice spacing a must be positive.")

        if phi_max <= phi_min:
            raise ValueError("Need phi_max > phi_min.")

        number_of_sites = int(np.floor((phi_max - phi_min) / a)) + 1
        phi_sites = phi_min + a * np.arange(number_of_sites)

        return phi_sites

    @staticmethod
    def mass_squared_lattice(phi_sites, mu: float, a: float, b: float = 1.0):
        """
        Returns the dimensionless lattice mass squared:

            a^2 m(phi)^2 = 4*pi*mu*a^2*exp(2*b*phi)

        This is the quantity that enters K.
        """
        if mu < 0:
            raise ValueError("mu must be non-negative.")

        return 4.0 * np.pi * mu * a**2 * np.exp(2.0 * b * phi_sites)

    @staticmethod
    def Kbuilder(
        mu: float,
        phi_min: float = -20.0,
        phi_max: float = 5.0,
        a: float = 0.05,
        b: float = 1.0,
    ):
        """
        Builds the Liouville lattice kernel:

            K_ij = delta_ij * (2 + a^2 m_i^2)
                - delta_{i,j+1}
                - delta_{i,j-1}

        with

            m_i^2 = 4*pi*mu*exp(2*b*phi_i).
        """
        phi_sites = eEconstructor.phi_grid(phi_min, phi_max, a)
        M = len(phi_sites)

        lattice_mass2 = eEconstructor.mass_squared_lattice(phi_sites, mu, a, b)

        diag = 2.0 * np.ones(M) + lattice_mass2
        offdiag = -np.ones(M - 1)

        K = np.diag(diag)
        K += np.diag(offdiag, k=1)
        K += np.diag(offdiag, k=-1)

        return K, phi_sites

    @staticmethod
    def full_correlators(
        mu: float,
        phi_min: float = -20.0,
        phi_max: float = 5.0,
        a: float = 0.05,
        b: float = 1.0,
    ):
        """
        Computes the ground-state correlators:

            X = <Psi Psi> = 1/2 K^{-1/2}
            P = <Pi Pi>  = 1/2 K^{1/2}
        """
        K, phi_sites = eEconstructor.Kbuilder(mu, phi_min, phi_max, a, b)

        w, O = eigh(K)

        if np.min(w) <= 0:
            raise ValueError(f"K is not positive definite. min eigenvalue = {np.min(w)}")

        sqrtK = (O * np.sqrt(w)) @ O.T
        invsqrtK = (O * (1.0 / np.sqrt(w))) @ O.T

        sqrtK = 0.5 * (sqrtK + sqrtK.T)
        invsqrtK = 0.5 * (invsqrtK + invsqrtK.T)

        X = 0.5 * invsqrtK
        P = 0.5 * sqrtK

        return X, P, phi_sites

    @staticmethod
    def entropy_for_interval(phi_left: float, phi_right: float, X, P, phi_sites):
        """
        Entanglement entropy of the interval:

            A = [phi_left, phi_right]

        The actual cut is snapped to the nearest available lattice sites.
        """
        if phi_right <= phi_left:
            raise ValueError("Need phi_right > phi_left.")

        if phi_left < phi_sites[0] or phi_right > phi_sites[-1]:
            raise ValueError("The interval is outside the lattice window.")

        left = np.searchsorted(phi_sites, phi_left, side="left")
        right = np.searchsorted(phi_sites, phi_right, side="right")

        if right <= left:
            raise ValueError("The interval contains no lattice sites.")

        XA = X[left:right, left:right]
        PA = P[left:right, left:right]

        # Cholesky: XA = R.T @ R
        R = cholesky(XA, lower=False)

        # Symmetric matrix similar to XA @ PA
        M = R @ PA @ R.T
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

        x_plus = nu + 0.5
        x_minus = nu - 0.5

        entropy_terms = x_plus * np.log(x_plus)
        entropy_terms -= np.where(
            x_minus > 1e-14,
            x_minus * np.log(x_minus),
            0.0,
        )

        return np.sum(entropy_terms)

    @staticmethod
    def entropy_as_function_of_left_endpoint(
        phi_right: float = 5,
        phi_left_min: float = -15,
        phi_left_max: float = 4.95,
        mu: float = 0.1,
        phi_min: float = -70,
        phi_max: float = 15,
        a: float = 0.1,
        b: float = 1.0,
    ):
        """
        Computes S([phi_left, phi_right]) as a function of phi_left.

        Since Liouville is not translation invariant in phi, the absolute
        position of the interval matters.
        """
        X, P, phi_sites = eEconstructor.full_correlators(
            mu=mu,
            phi_min=phi_min,
            phi_max=phi_max,
            a=a,
            b=b,
        )

        phi_left_values = np.arange(phi_left_min - 0.5*a, phi_left_max, a)
        

        S_values = np.array([
            (eEconstructor.entropy_for_interval(phi_left, phi_right, X, P, phi_sites))
            for phi_left in phi_left_values
        ])

        return phi_left_values, S_values

    @staticmethod
    def entropy_as_function_of_right_endpoint(
        phi_left: float = -15,
        phi_right_min: float = -14.999999,
        phi_right_max: float = 5,
        mu: float = 0.1,
        phi_min: float = -70,
        phi_max: float = 15,
        a: float = 0.1,
        b: float = 1.0,
    ):
        """
        Computes S([phi_left, phi_right]) as a function of phi_right.

        Since Liouville is not translation invariant in phi, the absolute
        position of the interval matters.
        """
        X, P, phi_sites = eEconstructor.full_correlators(
            mu=mu,
            phi_min=phi_min,
            phi_max=phi_max,
            a=a,
            b=b,
        )

        phi_right_values = np.arange(phi_right_min, phi_right_max + 0.5 * a, a)
        

        S_values = np.array([
            (eEconstructor.entropy_for_interval(phi_left, phi_right, X, P, phi_sites))
            for phi_right in phi_right_values
        ])

        return phi_right_values, S_values

    @staticmethod
    def entropy_as_function_of_center_point(
        phi_center_min: float = -15,
        phi_center_max: float = 5,
        Length: float = 0.5,
        mu: float = 0.1,
        phi_min: float = -70,
        phi_max: float = 15,
        a: float = 0.1,
        b: float = 1.0,
    ):
        """
        Computes S([phi_center]) as a function of phi_center for fixed Length L

        Since Liouville is not translation invariant in phi, the absolute
        position of the interval matters.
        """
        X, P, phi_sites = eEconstructor.full_correlators(
            mu=mu,
            phi_min=phi_min,
            phi_max=phi_max,
            a=a,
            b=b,
        )

        phi_center_values = np.arange(phi_center_min, phi_center_max + 0.5 * a, a)
        


        S_values = np.array([
            (eEconstructor.entropy_for_interval(phi_center - Length/2, phi_center + Length/2, X, P, phi_sites))
            for phi_center in phi_center_values
        ])

        return phi_center_values, S_values

    @staticmethod
    def entropy_as_function_of_length_with_fixed_center(
        phi_center: float = -25.0,
        L_min: float = 0.5,
        L_max: float = 10.0,
        mu: float = 0.1,
        phi_min: float = -100.0,
        phi_max: float = 10.0,
        a: float = 0.1,
        b: float = 1.0,
    ):
        """
        Computes

            S(L) = S([phi_center - L/2, phi_center + L/2])

        as a function of the interval length L, keeping the center phi_center fixed.

        This is the correct setup for extracting the c/3 logarithmic behavior
        in a non-translation-invariant Liouville background.
        """
        X, P, phi_sites = eEconstructor.full_correlators(
            mu=mu,
            phi_min=phi_min,
            phi_max=phi_max,
            a=a,
            b=b,
        )

        L_values = np.arange(L_min, L_max + 0.5 * a, a)

        S_values = np.array([
            eEconstructor.entropy_for_interval(
                phi_center - 0.5 * L,
                phi_center + 0.5 * L,
                X,
                P,
                phi_sites,
            )
            for L in L_values
        ])

        return L_values, S_values

    @staticmethod
    def entropy_grid(
        phi_left_min: float = -15.0,
        phi_left_max: float | None = None,
        phi_right_min: float | None = None,
        phi_right_max: float = 5.0,
        phi_min: float = -70.0,
        phi_max: float = 15.0,
        mu: float = 0.1,
        a: float = 0.1,
        b: float = 1.0,
    ):
        """
        Returns a 2D grid

            S_grid[i, j] = S([phi_left_values[i], phi_right_values[j]])

        Invalid intervals with phi_right <= phi_left are filled with np.nan.

        The endpoint arrays are chosen directly from phi_sites to avoid
        floating-point snapping artifacts in entropy_for_interval.
        """
        X, P, phi_sites = eEconstructor.full_correlators(
            mu=mu,
            phi_min=phi_min,
            phi_max=phi_max,
            a=a,
            b=b,
        )

        if phi_left_max is None:
            phi_left_max = phi_right_max - a

        if phi_right_min is None:
            phi_right_min = phi_left_min + a

        phi_left_values = phi_sites[
            (phi_sites >= phi_left_min) & (phi_sites <= phi_left_max)
        ]

        phi_right_values = phi_sites[
            (phi_sites >= phi_right_min) & (phi_sites <= phi_right_max)
        ]

        S_grid = np.full(
            (len(phi_left_values), len(phi_right_values)),
            np.nan,
            dtype=float,
        )

        for i, phi_left in enumerate(phi_left_values):
            for j, phi_right in enumerate(phi_right_values):
                if phi_right > phi_left:
                    S_grid[i, j] = eEconstructor.entropy_for_interval(
                        phi_left,
                        phi_right,
                        X,
                        P,
                        phi_sites,
                    )

        return phi_left_values, phi_right_values, S_grid
    
    @staticmethod
    def ssa_from_entropy_grid(S_grid):
        ssa = (
            S_grid[:-1, 1:]
            - S_grid[1:, 1:]
            - S_grid[:-1, :-1]
            + S_grid[1:, :-1]
        )
        return ssa

if __name__ == "__main__":

    phi_left_values, phi_right_values, S_grid = eEconstructor.entropy_grid(
        phi_left_min=-15,
        phi_right_max=5,
        phi_min=-70,
        phi_max=15,
        mu=0.1,
        a=0.1,
        b=1.0,
    )

    ssa = eEconstructor.ssa_from_entropy_grid(S_grid)

    tol = 1e-10
    valid = np.isfinite(ssa)

    violations = valid & (ssa > tol)

    print("max SSA defect =", np.nanmax(ssa[valid]))
    print("number of SSA violations =", np.sum(violations))