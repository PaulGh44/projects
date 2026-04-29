"""
complex_structure_viewer.py

A lightweight numerical viewer for the complex structure of a function f(z).

It displays:
  - phase portrait arg(f),
  - log modulus log10(|f|),
  - inverse log modulus -log10(|f|),
  - approximate zeros,
  - approximate poles,
  - manually chosen branch cuts,
  - manually specified branch points,
  - optional branch point candidates tested by monodromy.

This is a diagnostic plotting tool, not a rigorous complex-analysis solver.
For a general black-box function, poles, zeros and branch points cannot be
classified with complete reliability from sampled values alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal

import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp


PlotMode = Literal["phase", "logabs", "minus_logabs", "combined"]


@dataclass
class ComplexStructureViewer:
    """
    Numerical viewer for the complex structure of a function f.

    Parameters
    ----------
    f:
        Complex-valued function. It should accept an mpmath complex number.

    xlim, ylim:
        Plot domain in the complex plane.

    resolution:
        Number of grid points in each direction.

    zero_threshold:
        Points with |f(z)| < zero_threshold are displayed as approximate zeros.

    pole_threshold:
        Points with |f(z)| > pole_threshold are displayed as approximate poles.

    branch_points:
        List of manually specified branch points.

    branch_cuts:
        List of manually specified cuts. Each cut is stored as a NumPy array
        of complex numbers.
    """

    f: Callable[[complex], complex]

    xlim: tuple[float, float] = (-2.0, 2.0)
    ylim: tuple[float, float] = (-2.0, 2.0)
    resolution: int = 300

    zero_threshold: float = 1e-6
    pole_threshold: float = 1e6

    branch_points: list[complex] = field(default_factory=list)
    branch_cuts: list[np.ndarray] = field(default_factory=list)

    def evaluate_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate f on the rectangular grid.

        Returns
        -------
        X, Y, Z:
            X and Y are real coordinate arrays.
            Z[i, j] = f(X[i, j] + i Y[i, j]).
        """
        x = np.linspace(self.xlim[0], self.xlim[1], self.resolution)
        y = np.linspace(self.ylim[0], self.ylim[1], self.resolution)

        X, Y = np.meshgrid(x, y)
        Z = np.empty_like(X, dtype=complex)

        for i in range(self.resolution):
            for j in range(self.resolution):
                z = mp.mpc(X[i, j], Y[i, j])
                try:
                    Z[i, j] = complex(self.f(z))
                except Exception:
                    Z[i, j] = np.nan + 1j * np.nan

        return X, Y, Z

    @staticmethod
    def phase_data(Z: np.ndarray) -> np.ndarray:
        """Return arg(f)."""
        return np.angle(Z)

    @staticmethod
    def log_abs_data(Z: np.ndarray, eps: float = 1e-30) -> np.ndarray:
        """Return log10(|f| + eps)."""
        return np.log10(np.abs(Z) + eps)

    @staticmethod
    def inverse_log_abs_data(Z: np.ndarray, eps: float = 1e-30) -> np.ndarray:
        """
        Return -log10(|f| + eps).

        Large positive values indicate zeros.
        """
        return -np.log10(np.abs(Z) + eps)

    def detect_zeros_from_grid(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
    ) -> np.ndarray:
        """Approximate zeros by grid thresholding."""
        mask = np.abs(Z) < self.zero_threshold
        return X[mask] + 1j * Y[mask]

    def detect_poles_from_grid(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
    ) -> np.ndarray:
        """Approximate poles by grid thresholding."""
        mask = np.abs(Z) > self.pole_threshold
        return X[mask] + 1j * Y[mask]

    def add_branch_point(self, z: complex) -> None:
        """Manually add a branch point."""
        self.branch_points.append(complex(z))

    def add_branch_cut(self, z_start: complex, z_end: complex, npts: int = 200) -> None:
        """
        Add a straight numerical branch cut from z_start to z_end.

        Branch cuts are choices, not intrinsic objects.
        """
        z_start = complex(z_start)
        z_end = complex(z_end)

        t = np.linspace(0.0, 1.0, npts)
        cut = z_start + t * (z_end - z_start)

        self.branch_cuts.append(cut)

    def add_ray_branch_cut(
        self,
        z_start: complex,
        angle: float = 0.0,
        length: float = 10.0,
        npts: int = 300,
    ) -> None:
        """
        Add a ray branch cut starting at z_start.

        Parameters
        ----------
        z_start:
            Starting point of the cut.

        angle:
            Direction angle in radians.
            angle = 0 gives a ray to the right.
            angle = pi gives a ray to the left.

        length:
            Length of the numerical cut.
        """
        z_start = complex(z_start)
        t = np.linspace(0.0, length, npts)
        cut = z_start + t * np.exp(1j * angle)

        self.branch_cuts.append(cut)

    def monodromy_difference(
        self,
        z0: complex,
        radius: float = 1e-3,
        npts: int = 200,
    ) -> tuple[complex, complex, complex]:
        """
        Numerically go once around z0 and compare f at the beginning and end.

        Returns
        -------
        diff, f_start, f_end:
            diff = f_end - f_start.

        Warning
        -------
        For functions implemented with a principal branch, this test detects
        the numerical branch convention, not an abstract Riemann surface.
        """
        z0 = complex(z0)
        vals: list[complex] = []

        for k in range(npts + 1):
            theta = 2 * np.pi * k / npts
            z = z0 + radius * np.exp(1j * theta)

            try:
                vals.append(complex(self.f(mp.mpc(z))))
            except Exception:
                vals.append(np.nan + 1j * np.nan)

        f_start = vals[0]
        f_end = vals[-1]
        diff = f_end - f_start

        return diff, f_start, f_end

    def detect_branch_points_by_monodromy(
        self,
        candidate_points: Iterable[complex],
        radius: float = 1e-3,
        npts: int = 200,
        threshold: float = 1e-4,
    ) -> np.ndarray:
        """
        Test candidate points for branch-type monodromy.

        A point is flagged if |f_end - f_start| > threshold.
        """
        detected: list[complex] = []

        for z0 in candidate_points:
            diff, _f_start, _f_end = self.monodromy_difference(
                z0,
                radius=radius,
                npts=npts,
            )

            if abs(diff) > threshold:
                detected.append(complex(z0))

        return np.array(detected, dtype=complex)

    def plot(
        self,
        mode: PlotMode = "phase",
        show_zeros: bool = True,
        show_poles: bool = True,
        show_branch_points: bool = True,
        show_branch_cuts: bool = True,
        zero_color: str = "white",
        pole_color: str = "black",
        branch_point_color: str = "red",
        branch_cut_color: str = "red",
        figsize: tuple[float, float] = (8, 7),
        title: str = "Complex structure viewer",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Plot the complex structure.

        Modes
        -----
        phase:
            Plot arg(f).

        logabs:
            Plot log10(|f|). Poles appear as positive peaks; zeros as negative wells.

        minus_logabs:
            Plot -log10(|f|). Zeros appear as positive peaks.

        combined:
            Plot phase as color and log10(|f|) as contour lines.
        """
        X, Y, Z = self.evaluate_grid()

        fig, ax = plt.subplots(figsize=figsize)

        if mode == "phase":
            data = self.phase_data(Z)
            im = ax.contourf(X, Y, data, levels=100, cmap="hsv")
            plt.colorbar(im, ax=ax, label=r"$\arg f(z)$")

        elif mode == "logabs":
            data = self.log_abs_data(Z)
            im = ax.contourf(X, Y, data, levels=100)
            plt.colorbar(im, ax=ax, label=r"$\log_{10}|f(z)|$")

        elif mode == "minus_logabs":
            data = self.inverse_log_abs_data(Z)
            im = ax.contourf(X, Y, data, levels=100)
            plt.colorbar(im, ax=ax, label=r"$-\log_{10}|f(z)|$")

        elif mode == "combined":
            phase = self.phase_data(Z)
            logabs = self.log_abs_data(Z)

            im = ax.contourf(X, Y, phase, levels=100, cmap="hsv")
            plt.colorbar(im, ax=ax, label=r"$\arg f(z)$")

            finite_logabs = np.where(np.isfinite(logabs), logabs, np.nan)
            ax.contour(
                X,
                Y,
                finite_logabs,
                levels=15,
                colors="black",
                linewidths=0.4,
                alpha=0.45,
            )

        else:
            raise ValueError(
                "mode must be one of: 'phase', 'logabs', 'minus_logabs', 'combined'"
            )

        if show_zeros:
            zeros = self.detect_zeros_from_grid(X, Y, Z)
            if len(zeros) > 0:
                ax.scatter(
                    zeros.real,
                    zeros.imag,
                    s=10,
                    c=zero_color,
                    marker="o",
                    label="approx. zeros",
                )

        if show_poles:
            poles = self.detect_poles_from_grid(X, Y, Z)
            if len(poles) > 0:
                ax.scatter(
                    poles.real,
                    poles.imag,
                    s=20,
                    c=pole_color,
                    marker="x",
                    label="approx. poles",
                )

        if show_branch_cuts:
            for idx, cut in enumerate(self.branch_cuts):
                ax.plot(
                    cut.real,
                    cut.imag,
                    color=branch_cut_color,
                    linewidth=2.0,
                    label="branch cut" if idx == 0 else None,
                )

        if show_branch_points and len(self.branch_points) > 0:
            bp = np.array(self.branch_points, dtype=complex)
            ax.scatter(
                bp.real,
                bp.imag,
                s=100,
                c=branch_point_color,
                marker="*",
                label="branch points",
            )

        ax.axhline(0, linewidth=0.8, color="black", alpha=0.5)
        ax.axvline(0, linewidth=0.8, color="black", alpha=0.5)

        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel(r"$\operatorname{Re} z$")
        ax.set_ylabel(r"$\operatorname{Im} z$")
        ax.set_title(title)

        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys())

        plt.show()

        return X, Y, Z


# =========================================
# Test: square-root function
# =========================================


def sqrt_test_function(z):
    """
    Test function f(z) = sqrt(z - 1).

    It has a branch point at z = 1.
    The branch cut shown below is manually chosen to be the ray [1, +infinity).
    """
    return mp.sqrt(z - 1)


if __name__ == "__main__":
    mp.dps = 30

    viewer = ComplexStructureViewer(
        sqrt_test_function,
        xlim=(-3.0, 4.0),
        ylim=(-3.0, 3.0),
        resolution=300,
        zero_threshold=1e-3,
        pole_threshold=1e6,
    )

    # f(z) = sqrt(z - 1) has a branch point at z = 1.
    viewer.add_branch_point(1.0 + 0.0j)

    # Choose the branch cut by hand. Here: ray from 1 to +infinity.
    viewer.add_ray_branch_cut(
        z_start=1.0 + 0.0j,
        angle=0.0,
        length=3.0,
        npts=300,
    )


    # Main plot: phase portrait with branch point and chosen branch cut.
    viewer.plot(
        mode="phase",
        show_zeros=True,
        show_poles=True,
        show_branch_points=True,
        show_branch_cuts=True,
        title=r"Phase portrait of $f(z)=\sqrt{z-1}$",
    )

