from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import cumulative_simpson, solve_ivp

from Model import Model, ContourSegment


DEFAULT_MODE_FUNCTION_N_VALUES = [2, 3, 5, 15, 20]


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
    mode_function_n_values: list[int] = field(
        default_factory=lambda: DEFAULT_MODE_FUNCTION_N_VALUES.copy()
    )
    # One chi_n array per contour segment, aligned with ``segments``.
    # It is populated only for n in mode_function_n_values.
    chi_numerical: list[np.ndarray] | None = None

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
        mode_function_n_values: list[int] | None = None,
    ):
        self.model = model
        self.N_eval = N_eval
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.mode_function_n_values = (
            DEFAULT_MODE_FUNCTION_N_VALUES.copy()
            if mode_function_n_values is None
            else list(mode_function_n_values)
        )

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

    def _reconstruct_mode_along_contour(
        self,
        segment_solutions: list[SegmentSolution],
        contour_segments: list[ContourSegment],
    ) -> list[np.ndarray]:
        """Reconstruct chi_n on the complete complex contour from W_n.

        The definition W_n = -i a^3 (partial_t chi_n)/chi_n gives, on a
        parametrized contour t=t(x),

            chi_n(x) = chi_n(x_f)
                       exp[-i int_x^{x_f} W_n/a^3 (dt/dx') dx'].

        This is the contour version of eq. (B.30).  The numerical evolution
        ends at a large but finite t_final, where we set chi_n(t_final)=1.
        The omitted tail to +infinity is exponentially small for the values
        used in main.py.
        """

        if len(segment_solutions) != len(contour_segments):
            raise ValueError(
                "segment_solutions and contour_segments must have the same length."
            )

        chi_segments: list[np.ndarray] = [
            np.empty(0, dtype=complex)
            for _ in segment_solutions
        ]

        # Normalization lim_{t -> +infinity} chi_n(t)=1, approximated at
        # the final numerical point.  Reconstruct backwards so that the
        # normalization and continuity are propagated through every junction.
        chi_at_segment_end = 1.0 + 0.0j

        for index in range(len(segment_solutions) - 1, -1, -1):
            solution = segment_solutions[index]
            segment = contour_segments[index]

            x_values = np.asarray(solution.x, dtype=float)
            W_values = np.asarray(solution.W, dtype=complex)
            a_values = np.asarray(
                [segment.a(float(x)) for x in x_values],
                dtype=complex,
            )
            dt_dx_values = np.asarray(
                [segment.dt_dx(float(x)) for x in x_values],
                dtype=complex,
            )

            integrand = W_values * dt_dx_values / a_values**3
            integral_from_start = cumulative_simpson(
                integrand,
                x=x_values,
                initial=0.0,
            )
            integral_to_end = integral_from_start[-1] - integral_from_start

            chi_values = (
                chi_at_segment_end
                * np.exp(-1j * integral_to_end)
            )
            chi_segments[index] = chi_values

            # The start of this segment is the end of the preceding one.
            chi_at_segment_end = chi_values[0]

        return chi_segments

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

            # Initial condition for the next contour segment.
            W_initial = sol.y[0, -1]

        chi_numerical = None
        if n in self.mode_function_n_values:
            chi_numerical = self._reconstruct_mode_along_contour(
                segment_solutions=solutions,
                contour_segments=segments,
            )

        return ModeSolution(
            n=n,
            segments=solutions,
            mode_function_n_values=self.mode_function_n_values.copy(),
            chi_numerical=chi_numerical,
        )
