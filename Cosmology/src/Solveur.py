from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from Model import WineglassModel

from Model import Model, ContourSegment


@dataclass
class SegmentSolution:
    name: str
    x: np.ndarray
    t: np.ndarray
    W: np.ndarray


@dataclass
class ModeSolution:
    n: int
    segments: list[SegmentSolution]

    @property
    def W_final(self) -> complex:
        return self.segments[-1].W[-1]


class Solveur:

    def __init__(
        self,
        model: Model,
        N_eval: int = 1000,
        rtol: float = 1e-10,
        atol: float = 1e-12,
        method: str = "DOP853",
    ):
        self.model = model
        self.N_eval = N_eval
        self.rtol = rtol
        self.atol = atol
        self.method = method

    def _rhs_W(
        self,
        x: float,
        y: np.ndarray,
        n: int,
        segment: ContourSegment,
    ) -> np.ndarray:

        W = y[0]

        a = segment.a(x)
        dt_dx = segment.dt_dx(x)

        k2 = self.model.k2(n)

        dW_dx = 1j * dt_dx * (
            a * k2 - W**2 / a**3
        )

        return np.array([dW_dx], dtype=complex)

    def solve_W(
        self,
        n: int,
        delta: float,
        t_final: float,
    ) -> ModeSolution:

        segments = self.model.contour(
            delta=delta,
            t_final=t_final,
        )

        W_initial = complex(
            self.model.initial_W(n, delta)
        )

        solutions = []

        for segment in segments:

            x_eval = np.linspace(
                segment.x_start,
                segment.x_end,
                self.N_eval,
            )

            sol = solve_ivp(
                fun=lambda x, y: self._rhs_W(
                    x,
                    y,
                    n,
                    segment,
                ),
                t_span=(
                    segment.x_start,
                    segment.x_end,
                ),
                y0=np.array(
                    [W_initial],
                    dtype=complex,
                ),
                t_eval=x_eval,
                method=self.method,
                rtol=self.rtol,
                atol=self.atol,
            )

            if not sol.success:
                raise RuntimeError(
                    f"Integration failed in "
                    f"{segment.name}: {sol.message}"
                )

            t_values = np.array(
                [segment.t(x) for x in sol.t],
                dtype=complex,
            )

            segment_solution = SegmentSolution(
                name=segment.name,
                x=sol.t,
                t=t_values,
                W=sol.y[0],
            )

            solutions.append(segment_solution)

            # Initial condition for the next
            # contour segment
            W_initial = sol.y[0, -1]

        return ModeSolution(
            n=n,
            segments=solutions,
        )