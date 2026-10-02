from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
import numpy as np
import scipy.integrate as integrate


@dataclass
class ContourSegment:
    name: str
    x_start: float
    x_end: float

    # Complex Lorentzian time t(x)
    t: Callable[[float], complex]
    dt_dx: Callable[[float], complex]

    # Scale factor as a function of the numerical coordinate x
    a: Callable[[float], complex]


@dataclass
class Model(ABC):

    @abstractmethod
    def contour(
        self,
        delta: float,
        t_final: float
    ) -> list[ContourSegment]:
        """Return the ordered contour segments."""
        pass

    @abstractmethod
    def initial_W(
        self,
        n: int,
        delta: float
    ) -> complex:
        """Initial condition for the Riccati variable W."""
        pass

    @abstractmethod
    def scale_factor_euclidean(
        self,
        tau: float
    ) -> float:
        pass

    @abstractmethod
    def scale_factor_lorentzian(
        self,
        t: float
    ) -> float:
        pass

    @abstractmethod
    def Analytic_power_spectrum(
        self,
        n: int,
        t_final: float
    ) -> float:
        pass

    @abstractmethod
    def Analytic_mode_function_Euclidean(
            self,
            n: int,
            tau: float
    ) -> complex:
        pass

    @abstractmethod
    def Analytic_mode_function_Lorentzian(
                self,
                n: int,
                t: float
    ) -> complex:
        pass

    @staticmethod
    def k2(n: int) -> float:
        return n**2 - 1



@dataclass
class NoBoundaryModel(Model):
    H: float
    model_name = "No-boundary"

    @property
    def tau_SP(self) -> float:
        return -np.pi / (2.0 * self.H)

    def scale_factor_euclidean(
        self,
        tau: float
    ) -> float:
        return np.cos(self.H * tau) / self.H

    def scale_factor_lorentzian(
        self,
        t: float
    ) -> float:
        return np.cosh(self.H * t) / self.H

    def contour(
        self,
        delta: float,
        t_final: float
    ) -> list[ContourSegment]:

        tau_initial = self.tau_SP + delta / self.H

        euclidean = ContourSegment(
            name="Euclidean",
            x_start=tau_initial,
            x_end=0.0,
            t=lambda tau: -1j * tau,
            dt_dx=lambda tau: -1j,
            a=self.scale_factor_euclidean,
        )

        lorentzian = ContourSegment(
            name="Lorentzian",
            x_start=0.0,
            x_end=t_final,
            t=lambda t: complex(t),
            dt_dx=lambda t: 1.0,
            a=self.scale_factor_lorentzian,
        )

        return [euclidean, lorentzian]

    def initial_W(
        self,
        n: int,
        delta: float
    ) -> complex:

        tau_initial = self.tau_SP + delta / self.H

        a_initial = self.scale_factor_euclidean(
            tau_initial
        )

        # delta = H (tau_initial - tau_SP)
        #
        # Phi_n ~ delta^(n - 1)
        #
        # therefore
        #
        # Phi'/Phi ~ H (n - 1) / delta

        R_initial = (
            self.H
            * (n - 1)
            / delta
        )

        return a_initial**3 * R_initial

    def Analytic_power_spectrum(
            self,
            n: int,
            t_final: float
        ) -> float:
        return (self.H**2/(2*n*(n**2-1))) * (1+(n**2-1)/((self.scale_factor_lorentzian(t_final)*self.H)**2))

    def Analytic_mode_function_Euclidean(
        self,
        n: int,
        tau: float
    ) -> complex:
        # Analytic continuation t = -i tau of eq. (3.38).
        # The explicit complex cast is important for even n, for which
        # half-integer powers of the negative denominator occur.
        s = np.asarray(np.sin(self.H * tau), dtype=complex)
        chi = (
            (1.0 + s)**((n - 1) / 2)
            * (s - n)
            / (s - 1.0)**((n + 1) / 2)
        )
        return complex(chi) if np.ndim(chi) == 0 else chi

    def Analytic_mode_function_Lorentzian(
        self,
        n: int,
        t: float
    ) -> complex:
        # Eq. (3.38), normalized so that chi_n(t -> +infinity) = 1.
        s = np.asarray(1j * np.sinh(self.H * t), dtype=complex)
        chi = (
            (1.0 + s)**((n - 1) / 2)
            * (s - n)
            / (s - 1.0)**((n + 1) / 2)
        )
        return complex(chi) if np.ndim(chi) == 0 else chi

    

@dataclass
class WineglassModel(Model):
    H_II: float
    rho_rad: float
    model_name = "Wineglass" 

    def __post_init__(self):

        if self.H_II <= 0:
            raise ValueError("H_II must be positive.")

        if self.rho_rad <= 0:
            raise ValueError(
                "rho_rad must be positive for H^2 = 3/(16 rho_rad)."
            )

        rho_min = 1.0 / (8.0 * self.H_II**2)

        if self.rho_rad <= rho_min:
            raise ValueError(
                f"Wineglass solution requires "
                f"rho_rad > 1/(8 H_II^2) = {rho_min}."
            )

    # =========================================================
    # Model parameters derived from H_II and rho_rad
    # =========================================================

    @property
    def A_I(self) -> float:
        return (1.0 + np.sqrt(5.0)) / 2.0

    @property
    def c_II(self) -> float:
        return (
            16.0
            * self.H_II**2
            * self.rho_rad
            - 1.0
        )

    @property
    def H(self) -> float:
        return np.sqrt(
            3.0 / (16.0 * self.rho_rad)
        )

    @property
    def H_I(self) -> float:

        numerator = (
            self.H_II**2
            * (np.sqrt(5.0) - 2.0)
        )

        denominator = (
            16.0
            * self.rho_rad
            * self.H_II**2
            - 2.0
        )

        return np.sqrt(
            numerator / denominator
        )

    @property
    def tau_min(self) -> float:
        return -np.pi / (2.0 * self.H_II)

    # =========================================================
    # Region I
    # =========================================================

    def _bar_tau_I(self, tau: float) -> float:
        return (
            2.0
            * self.H_I
            * (tau - self.tau_min)
        )

    def _scale_factor_region_I(
        self,
        tau: float
    ) -> float:

        bar_tau = self._bar_tau_I(tau)

        cosh = np.cosh(bar_tau)

        a2 = (
            (
                np.sqrt(self.A_I) * cosh
                - 1.0 / np.sqrt(self.A_I)
            )**2
            /
            (
                4.0
                * self.H_I**2
                * cosh
            )
        )

        return np.sqrt(a2)

    # =========================================================
    # Euclidean region II
    # =========================================================

    def _scale_factor_region_II(
        self,
        tau: float
    ) -> float:

        bar_tau = 2.0 * self.H_II * tau

        a2 = (
            np.cos(bar_tau) + self.c_II
        ) / (4.0 * self.H_II**2)

        return np.sqrt(a2)

    # =========================================================
    # Lorentzian region
    # =========================================================

    def scale_factor_lorentzian(
        self,
        t: float
    ) -> float:

        a2 = (
            2.0 + np.cosh(2.0 * self.H * t)
        ) / (4.0 * self.H**2)

        return np.sqrt(a2)

    # =========================================================
    # Generic Euclidean interface
    # =========================================================

    def scale_factor_euclidean(
        self,
        tau: float
    ) -> float:

        if tau <= self.tau_min:
            return self._scale_factor_region_I(tau)

        if tau <= 0.0:
            return self._scale_factor_region_II(tau)

        raise ValueError(
            "Euclidean scale factor is defined only for tau <= 0."
        )
    

    def _scale_factor_euclidean_values(self, tau: float | np.ndarray) -> np.ndarray:
        """Vectorized Euclidean scale factor on tau <= 0."""

        tau_array = np.asarray(tau, dtype=float)
        if np.any(tau_array > 0.0):
            raise ValueError("Euclidean scale factor is defined only for tau <= 0.")

        result = np.empty(tau_array.shape, dtype=float)
        mask_I = tau_array <= self.tau_min
        if np.any(mask_I):
            result[mask_I] = self._scale_factor_region_I(tau_array[mask_I])
        if np.any(~mask_I):
            result[~mask_I] = self._scale_factor_region_II(tau_array[~mask_I])
        return result

    def _tau_from_u(self, u: float) -> float:
        return (
            self.tau_min
            + np.log(u) / (2.0 * self.H_I)
        )
    
    
    def contour(
        self,
        delta: float,
        t_final: float
    ) -> list[ContourSegment]:

        if not (0.0 < delta < 1.0):
            raise ValueError(
                "For the wineglass model, delta must satisfy "
                "0 < delta < 1 and represents the initial u."
            )

        # -------------------------------------------------
        # Region I: compactified EAdS
        # -------------------------------------------------

        region_I = ContourSegment(
            name="Euclidean I",
            x_start=delta,
            x_end=1.0,

            t=lambda u:
                -1j * self._tau_from_u(u),

            dt_dx=lambda u:
                -1j / (2.0 * self.H_I * u),

            a=lambda u:
                self._scale_factor_region_I(
                    self._tau_from_u(u)
                ),
        )

        # -------------------------------------------------
        # Euclidean region II
        # -------------------------------------------------

        region_II_E = ContourSegment(
            name="Euclidean II",
            x_start=self.tau_min,
            x_end=0.0,

            t=lambda tau:
                -1j * tau,

            dt_dx=lambda tau:
                -1j,

            a=self._scale_factor_region_II,
        )

        # -------------------------------------------------
        # Lorentzian region II
        # -------------------------------------------------

        region_II_L = ContourSegment(
            name="Lorentzian II",
            x_start=0.0,
            x_end=t_final,

            t=lambda t:
                complex(t),

            dt_dx=lambda t:
                1.0,

            a=self.scale_factor_lorentzian,
        )

        return [
            region_I,
            region_II_E,
            region_II_L,
        ]
    
    def initial_W(
        self,
        n: int,
        delta: float
    ) -> complex:

        u = delta

        tau_initial = self._tau_from_u(u)

        a_initial = self._scale_factor_region_I(
            tau_initial
        )
        
        R_initial = (
            3
            * self.H_I
        )

        return a_initial**3 * R_initial

    def _conformal_times(self) -> tuple[float, float]:
        """Return the Euclidean and Lorentzian pieces of eta_infinity, eq. (B.27)."""

        # Euclidean region I:
        # y = 2 H_I (tau - tau_min), followed by exp(y) = s^2.
        A = self.A_I
        eta_E_I = integrate.quad(
            lambda s: (
                2.0
                * np.sqrt(2.0 * A)
                * np.sqrt(1.0 + s**4)
                / (A * (1.0 + s**4) - 2.0 * s**2)
            ),
            0.0,
            1.0,
        )[0]

        # Euclidean region II: x = H_II tau.
        eta_E_II = integrate.quad(
            lambda x: 2.0 / np.sqrt(self.c_II + np.cos(2.0 * x)),
            -np.pi / 2.0,
            0.0,
        )[0]

        # Lorentzian region: x = H t, followed by u = exp(-x).
        eta_L = integrate.quad(
            lambda u: (
                2.0
                * np.sqrt(2.0)
                / np.sqrt(u**4 + 4.0 * u**2 + 1.0)
            ),
            0.0,
            1.0,
        )[0]

        return eta_E_I + eta_E_II, eta_L

    def _wkb_matching_constants(
        self,
        n: int
    ) -> tuple[float, float, float, complex, complex, complex]:
        """Constants entering eqs. (B.14), (B.17), (B.20), and (B.26)."""

        k_n = np.sqrt(Model.k2(n))
        eta_E, eta_L = self._conformal_times()

        theta_E = k_n * eta_E
        theta_L = k_n * eta_L
        eta_infty = eta_E + 1j * eta_L

        N_n = (
            1j
            * k_n
            / (2.0 * self.H)
            / np.sinh(k_n * eta_infty)
        )

        A_plus = N_n * np.exp(-theta_E - 1j * theta_L)
        A_minus = N_n * np.exp(theta_E + 1j * theta_L)

        return k_n, theta_E, theta_L, N_n, A_plus, A_minus

    def Analytic_power_spectrum(
        self,
        n: int,
        t_final: float
    ) -> float:

        eta_E, eta_L = self._conformal_times()
        eta_infty = eta_E + 1j * eta_L

        # k_n^2 = n^2 - 1. Eq. (B.29) contains k_n^3 and
        # Re[coth(k_n eta_infty)].
        k_n = np.sqrt(Model.k2(n))
        coth = 1.0 / np.tanh(k_n * eta_infty)

        return (
            self.H**2
            / (2.0 * k_n**3)
            / np.real(coth)
            * (
                1.0
                + k_n**2
                / (
                    self.scale_factor_lorentzian(t_final)
                    * self.H
                )**2
            )
        )

    def _eta_E_to_tau_scalar(self, tau: float) -> float:
        """Euclidean conformal time int_{-infinity}^tau d tau'/a(tau')."""

        if tau > 0.0:
            raise ValueError("tau must satisfy tau <= 0.")

        A = self.A_I

        def integrand_s(s: float) -> float:
            return (
                2.0
                * np.sqrt(2.0 * A)
                * np.sqrt(1.0 + s**4)
                / (A * (1.0 + s**4) - 2.0 * s**2)
            )

        if tau <= self.tau_min:
            # u = exp[2 H_I (tau-tau_min)] = s^2.
            u = np.exp(2.0 * self.H_I * (tau - self.tau_min))
            return integrate.quad(
                integrand_s,
                0.0,
                np.sqrt(u),
            )[0]

        eta_E_I = integrate.quad(
            integrand_s,
            0.0,
            1.0,
        )[0]

        return eta_E_I + integrate.quad(
            lambda x: 1.0 / self._scale_factor_region_II(x),
            self.tau_min,
            tau,
        )[0]

    def _eta_L_from_t_scalar(self, t: float) -> float:
        """Lorentzian conformal time int_t^infinity dt'/a(t')."""

        if t < 0.0:
            raise ValueError("t must satisfy t >= 0.")

        # x = H t and then u = exp(-x).  This form stays accurate
        # even at late times, when the conformal-time tail is small.
        u_max = np.exp(-self.H * t)
        return integrate.quad(
            lambda u: (
                2.0
                * np.sqrt(2.0)
                / np.sqrt(u**4 + 4.0 * u**2 + 1.0)
            ),
            0.0,
            u_max,
        )[0]

    def Analytic_mode_function_EAdS(
        self,
        n: int,
        tau: float
    ) -> complex:
        """Asymptotic EAdS estimate: first line of eq. (B.31)."""

        tau_array = np.asarray(tau, dtype=float)
        if np.any(tau_array > self.tau_min):
            # The formula can be evaluated beyond the asymptotic region, but
            # its controlled domain is the EAdS side.  Do not forbid such an
            # evaluation here because it is useful for displaying the overlap.
            pass

        k_n, _, _, N_n, _, _ = self._wkb_matching_constants(n)
        a = self._scale_factor_euclidean_values(tau_array)
        z_E = k_n / (self.H_I * a)

        # z[e^z(1-1/z)+e^-z(1+1/z)] = 2(z cosh z-sinh z).
        # The prefactor is H_I/k_n from matching to (B.10).
        chi = (
            N_n
            * self.H_I
            / k_n
            * 2.0
            * (z_E * np.cosh(z_E) - np.sinh(z_E))
        )
        return complex(chi) if np.ndim(chi) == 0 else chi

    def Analytic_mode_function_WKB_Euclidean(
        self,
        n: int,
        tau: float,
        eta_E: float | np.ndarray | None = None,
    ) -> complex:
        """Central Euclidean WKB estimate from (B.10)/(B.31).

        eta_E denotes int_{-infinity}^tau d tau'/a(tau').
        Passing it explicitly is useful when evaluating the mode on a grid.
        """

        tau_array = np.asarray(tau, dtype=float)
        if np.any(tau_array > 0.0):
            raise ValueError("The Euclidean mode is defined only for tau <= 0.")

        k_n, _, _, N_n, _, _ = self._wkb_matching_constants(n)

        if eta_E is None:
            eta_values = np.array([
                self._eta_E_to_tau_scalar(float(value))
                for value in np.atleast_1d(tau_array)
            ]).reshape(tau_array.shape)
        else:
            eta_values = np.asarray(eta_E, dtype=float)

        a = self._scale_factor_euclidean_values(tau_array)
        theta = k_n * eta_values
        chi = N_n / a * (np.exp(theta) + np.exp(-theta))
        return complex(chi) if np.ndim(chi) == 0 else chi

    def Analytic_mode_function_WKB_Lorentzian(
        self,
        n: int,
        t: float,
        eta_L_to_infinity: float | np.ndarray | None = None,
    ) -> complex:
        """Central Lorentzian WKB estimate from (B.16)-(B.20).

        The phase convention used here is
            chi_WKB,+/- = a^{-1} exp[+/- i k_n int_t^infinity dt'/a].
        With this convention (B.18) and the coefficients (B.20) are mutually
        consistent.  Equivalently, the integration limits in the printed
        version of (B.16) should be t -> infinity.
        """

        t_array = np.asarray(t, dtype=float)
        if np.any(t_array < 0.0):
            raise ValueError("The Lorentzian mode is defined only for t >= 0.")

        k_n, _, _, _, A_plus, A_minus = self._wkb_matching_constants(n)

        if eta_L_to_infinity is None:
            eta_values = np.array([
                self._eta_L_from_t_scalar(float(value))
                for value in np.atleast_1d(t_array)
            ]).reshape(t_array.shape)
        else:
            eta_values = np.asarray(eta_L_to_infinity, dtype=float)

        a = self.scale_factor_lorentzian(t_array)
        phase = k_n * eta_values
        chi = (
            A_plus / a * np.exp(1j * phase)
            + A_minus / a * np.exp(-1j * phase)
        )
        return complex(chi) if np.ndim(chi) == 0 else chi

    def Analytic_mode_function_dS(
        self,
        n: int,
        t: float
    ) -> complex:
        """Asymptotic-dS estimate: last line of eq. (B.31)."""

        t_array = np.asarray(t, dtype=float)
        if np.any(t_array < 0.0):
            raise ValueError("The Lorentzian mode is defined only for t >= 0.")

        k_n, _, _, _, A_plus, A_minus = self._wkb_matching_constants(n)
        a = self.scale_factor_lorentzian(t_array)
        z_L = k_n / (a * self.H)

        f_plus = np.exp(1j * z_L) * (z_L + 1j)
        f_minus = np.exp(-1j * z_L) * (z_L - 1j)
        chi = self.H / k_n * (A_plus * f_plus + A_minus * f_minus)
        return complex(chi) if np.ndim(chi) == 0 else chi

    @property
    def tau_WKB_start(self) -> float:
        """Heuristic start of the central Euclidean WKB region.

        The EAdS approximation requires H_I tau << -1.  For the diagnostic
        mode-function plots we therefore switch to the central WKB estimate
        once tau reaches -1/H_I.  The extra min prevents the asymptotic EAdS
        formula from being used inside Euclidean region II if parameters are
        such that -1/H_I lies to the right of tau_min.
        """

        return min(-2 / self.H_I, self.tau_min)

    @property
    def t_WKB_end(self) -> float:
        """Heuristic end of the central Lorentzian WKB region."""

        return 2 / self.H

    def Analytic_mode_function_Euclidean(
        self,
        n: int,
        tau: float
    ) -> complex:
        """Piecewise analytic estimate on the Euclidean branch.

        We use the asymptotic EAdS solution for tau < tau_WKB_start and the
        central WKB expression of (B.31) from tau_WKB_start up to the
        Euclidean/Lorentzian junction.  Since tau is negative, the criterion
        corresponding to |H_I tau| < 1 is tau > -1/H_I.

        This time-domain split is intended for displaying the four analytic
        regions of Appendix B.  It is a heuristic visualization choice; the
        formal overlap conditions remain 1 << z_E << k_n.
        """

        tau_array = np.asarray(tau, dtype=float)
        if np.any(tau_array > 0.0):
            raise ValueError("The Euclidean mode is defined only for tau <= 0.")

        eta_values = np.array([
            self._eta_E_to_tau_scalar(float(value))
            for value in np.atleast_1d(tau_array)
        ]).reshape(tau_array.shape)

        chi_eads = self.Analytic_mode_function_EAdS(n, tau_array)
        chi_wkb = self.Analytic_mode_function_WKB_Euclidean(
            n, tau_array, eta_E=eta_values
        )
        chi = np.where(
            tau_array < self.tau_WKB_start,
            chi_eads,
            chi_wkb,
        )
        return complex(chi) if np.ndim(chi) == 0 else chi

    def Analytic_mode_function_Lorentzian(
        self,
        n: int,
        t: float
    ) -> complex:
        """Piecewise analytic estimate on the Lorentzian branch.

        The central Lorentzian WKB expression of (B.31) is used from t=0
        through t=1/H.  No extra N_n multiplies this expression: the
        coefficients A_{n,+/-} already contain N_n through (B.20).  For
        t>1/H we use the asymptotic-dS expression in the last line of (B.31).

        As on the Euclidean side, the switch at H t = 1 is a transparent
        plotting prescription rather than a replacement for the formal
        overlap condition 1 << z_L << k_n.
        """

        t_array = np.asarray(t, dtype=float)
        if np.any(t_array < 0.0):
            raise ValueError("The Lorentzian mode is defined only for t >= 0.")

        eta_values = np.array([
            self._eta_L_from_t_scalar(float(value))
            for value in np.atleast_1d(t_array)
        ]).reshape(t_array.shape)

        chi_wkb = self.Analytic_mode_function_WKB_Lorentzian(
            n, t_array, eta_L_to_infinity=eta_values
        )
        chi_ds = self.Analytic_mode_function_dS(n, t_array)
        chi = np.where(
            t_array <= self.t_WKB_end,
            chi_wkb,
            chi_ds,
        )
        return complex(chi) if np.ndim(chi) == 0 else chi

