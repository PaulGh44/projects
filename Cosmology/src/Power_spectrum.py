# Power_spectrum.py

from dataclasses import dataclass
import numpy as np

from Solveur import ModeSolution


@dataclass
class SpectrumPoint:
    n: int
    W_final: complex
    power: float
    dimensionless_power: float


class PowerSpectrum:

    @staticmethod
    def extract(solution: ModeSolution) -> SpectrumPoint:

        n = solution.n

        if n == 1:
            raise ValueError(
                "The n = 1 homogeneous mode is singular "
                "for a massless scalar field."
            )

        W_final = solution.W_final
        Re_W = W_final.real

        if Re_W <= 0:
            raise ValueError(
                f"Re(W) = {Re_W} <= 0 for n = {n}. "
                "The wave function is not Gaussian suppressed."
            )

        power = 1.0 / (2.0 * Re_W)

        dimensionless_power = (
            n * (n**2 - 1)
            / (2.0 * np.pi**2)
            * power
        )

        return SpectrumPoint(
            n=n,
            W_final=W_final,
            power=power,
            dimensionless_power=dimensionless_power,
        )