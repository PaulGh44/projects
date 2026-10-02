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

        tau_initial = self.tau_SP + delta

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
        return (1+np.sin(self.H*tau))**((n-1)/2) * (np.sin(self.H*tau)-n) / ((np.sin(self.H*tau)-1)**((n+1)/2))

    def Analytic_mode_function_Lorentzian(
                    self,
                    n: int,
                    t: float
            ) -> complex:
        return (1+1j*np.sinh(self.H*t))**((n-1)/2) * (1j*np.sinh(self.H*t)-n) / ((1j*np.sinh(self.H*t)-1)**((n+1)/2))

    

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

    def Analytic_power_spectrum(
        self,
        n: int,
        t_final: float
    ) -> float:

        # Eq. (B.27): total complex conformal time along the contour.
        # The changes of variables below map the infinite Euclidean and
        # Lorentzian ranges to finite intervals and avoid numerical overflow.

        # Euclidean region I:
        # y = 2 H_I (tau - tau_min), followed by u = exp(y) = s^2.
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

        eta_infty = eta_E_I + eta_E_II + 1j * eta_L

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


