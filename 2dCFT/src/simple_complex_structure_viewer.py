"""
simple_complex_structure_viewer.py

A simplified complex-structure viewer.

Given a numerical complex function f(z), this plots only:
    - the argument arg f(z),
    - approximate zeros,
    - approximate poles,
    - numerical branch cuts detected as phase jumps.

It does NOT display branch-point candidates.

This is meant as a diagnostic tool for functions implemented with principal branches,
for example mpmath sqrt/log. The displayed branch cuts are therefore the numerical
cuts chosen by the implementation, not necessarily the cuts one would choose
analytically by hand.

Run:
    python simple_complex_structure_viewer.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp


@dataclass
class SimpleComplexStructureViewer:
    f: Callable[[complex], complex]

    xlim: tuple[float, float] = (-3.0, 3.0)
    ylim: tuple[float, float] = (-3.0, 3.0)
    resolution: int = 300

    # Detection thresholds.
    # Increase zero_threshold if square-root zeros are missed.
    zero_threshold: float = 1e-2

    # Decrease pole_threshold if poles are missed.
    pole_threshold: float = 1e4

    # Principal sqrt/log cuts often give phase jumps close to pi.
    phase_jump_threshold: float = 2.2

    # Clustering only affects the displayed zero/pole markers.
    cluster_radius: float = 0.08

    zeros: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    poles: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    branch_cut_points: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))

    def safe_eval(self, z: complex) -> complex:
        """
        Evaluate f(z), returning nan+inan if evaluation fails.
        """
        try:
            return complex(self.f(mp.mpc(z)))
        except Exception:
            return np.nan + 1j * np.nan

    def evaluate_grid(self):
        """
        Evaluate f on a rectangular grid.

        Returns:
            X, Y, Z with Z[i,j] = f(X[i,j] + iY[i,j]).
        """
        x = np.linspace(self.xlim[0], self.xlim[1], self.resolution)
        y = np.linspace(self.ylim[0], self.ylim[1], self.resolution)

        X, Y = np.meshgrid(x, y)
        Z = np.empty_like(X, dtype=complex)

        for i in range(self.resolution):
            for j in range(self.resolution):
                Z[i, j] = self.safe_eval(X[i, j] + 1j * Y[i, j])

        return X, Y, Z

    def cluster_points(self, points, radius: float | None = None):
        """
        Simple greedy clustering of nearby points.
        This avoids plotting hundreds of markers around one zero or pole.
        """
        points = np.array(list(points), dtype=complex)

        if len(points) == 0:
            return np.array([], dtype=complex)

        if radius is None:
            radius = self.cluster_radius

        unused = list(points)
        clusters = []

        while unused:
            seed = unused.pop(0)
            cluster = [seed]

            changed = True
            while changed:
                changed = False
                center = np.mean(cluster)
                keep = []

                for p in unused:
                    if abs(p - center) < radius:
                        cluster.append(p)
                        changed = True
                    else:
                        keep.append(p)

                unused = keep

            clusters.append(np.mean(cluster))

        return np.array(clusters, dtype=complex)

    def detect_zeros(self, X, Y, Z):
        """
        Detect approximate zeros.

        Uses:
            1. direct threshold |f| < zero_threshold,
            2. local minima of |f|.

        The local-minimum part helps with square-root zeros, where |f|
        decays slowly near the zero.
        """
        A = np.abs(Z)
        finite = np.isfinite(A)

        candidates = []

        # Direct threshold.
        mask = finite & (A < self.zero_threshold)
        candidates.extend(list(X[mask] + 1j * Y[mask]))

        # Local minima.
        n, m = A.shape

        for i in range(1, n - 1):
            for j in range(1, m - 1):
                if not finite[i, j]:
                    continue

                center = A[i, j]
                neigh = A[i - 1:i + 2, j - 1:j + 2]

                if not np.all(np.isfinite(neigh)):
                    continue

                if center == np.nanmin(neigh):
                    local_median = np.nanmedian(neigh)

                    if local_median > 0 and center < 0.5 * local_median:
                        candidates.append(X[i, j] + 1j * Y[i, j])

        self.zeros = self.cluster_points(candidates)
        return self.zeros

    def detect_poles(self, X, Y, Z):
        """
        Detect approximate poles by thresholding |f|.
        """
        A = np.abs(Z)
        finite = np.isfinite(A)

        mask = finite & (A > self.pole_threshold)
        candidates = X[mask] + 1j * Y[mask]

        self.poles = self.cluster_points(candidates)
        return self.poles

    def detect_branch_cuts(self, X, Y, Z):
        """
        Detect numerical branch cuts as phase-jump curves.

        This detects where neighboring grid points have a large jump in arg(f).
        It shows the cuts imposed by the numerical implementation of f.
        """
        phase = np.angle(Z)

        dx_jump = np.abs(np.angle(np.exp(1j * (phase[:, 1:] - phase[:, :-1]))))
        dy_jump = np.abs(np.angle(np.exp(1j * (phase[1:, :] - phase[:-1, :]))))

        points = []

        rows, cols = np.where(dx_jump > self.phase_jump_threshold)
        for i, j in zip(rows, cols):
            points.append(
                0.5 * (X[i, j] + X[i, j + 1])
                + 1j * 0.5 * (Y[i, j] + Y[i, j + 1])
            )

        rows, cols = np.where(dy_jump > self.phase_jump_threshold)
        for i, j in zip(rows, cols):
            points.append(
                0.5 * (X[i, j] + X[i + 1, j])
                + 1j * 0.5 * (Y[i, j] + Y[i + 1, j])
            )

        self.branch_cut_points = np.array(points, dtype=complex)
        return self.branch_cut_points

    def analyze(self):
        """
        Run all detectors.
        """
        X, Y, Z = self.evaluate_grid()

        self.detect_zeros(X, Y, Z)
        self.detect_poles(X, Y, Z)
        self.detect_branch_cuts(X, Y, Z)

        return X, Y, Z

    def plot_argument(
        self,
        zero_color: str = "white",
        pole_color: str = "black",
        branch_cut_color: str = "red",
        figsize=(8, 7),
    ):
        """
        Plot arg f(z), with numerical branch cuts, approximate zeros and poles.
        """
        X, Y, Z = self.analyze()

        phase = np.angle(Z)

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.contourf(
            X,
            Y,
            phase,
            levels=100,
            cmap="hsv",
        )

        plt.colorbar(im, ax=ax, label=r"$\arg f(z)$")

        if len(self.branch_cut_points) > 0:
            ax.scatter(
                self.branch_cut_points.real,
                self.branch_cut_points.imag,
                s=5,
                c=branch_cut_color,
                marker=".",
                alpha=0.8,
                label="numerical branch cut",
            )

        if len(self.zeros) > 0:
            ax.scatter(
                self.zeros.real,
                self.zeros.imag,
                s=45,
                c=zero_color,
                marker="o",
                edgecolors="black",
                linewidths=0.6,
                label="approx. zeros",
            )

        if len(self.poles) > 0:
            ax.scatter(
                self.poles.real,
                self.poles.imag,
                s=70,
                c=pole_color,
                marker="x",
                linewidths=1.8,
                label="approx. poles",
            )

        ax.axhline(0, linewidth=0.8, color="black", alpha=0.5)
        ax.axvline(0, linewidth=0.8, color="black", alpha=0.5)

        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel(r"$\operatorname{Re} z$")
        ax.set_ylabel(r"$\operatorname{Im} z$")
        ax.set_title("Argument plot with zeros, poles and numerical branch cuts")

        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys())

        plt.show()

        return X, Y, Z


# ============================================================
# Tests
# ============================================================

def test_target_function():
    """
    Test:

        f(z) = sqrt(z^2 - 1) / ((z + 1)(z - 2))

    Expected:
        - approximate zero near z = 1,
        - approximate pole near z = 2,
        - numerical branch cuts inherited from mp.sqrt(z^2 - 1).

    Note:
        z = -1 is not an ordinary pole. It is a branch-point singularity.
        This simplified viewer does not classify branch points; it only displays
        cuts, zeros and poles.
    """

    def f(z):
        return mp.sqrt(z**2 - 1) / ((z + 1) * (z - 2)) * mp.log(z)

    viewer = SimpleComplexStructureViewer(
        f,
        xlim=(-3, 3),
        ylim=(-3, 3),
        resolution=350,
        zero_threshold=0.15,
        pole_threshold=50,
        phase_jump_threshold=2.2,
        cluster_radius=0.08,
    )

    viewer.plot_argument()


def test_sqrt_z_minus_1():
    """
    Test:

        f(z) = sqrt(z - 1)

    Expected:
        - numerical branch cut inherited from principal sqrt: (-infty, 1],
        - approximate zero near z = 1.
    """

    def f(z):
        return mp.sqrt(z - 1)

    viewer = SimpleComplexStructureViewer(
        f,
        xlim=(-3, 3),
        ylim=(-3, 3),
        resolution=300,
        zero_threshold=0.12,
        pole_threshold=1e6,
        phase_jump_threshold=2.2,
        cluster_radius=0.08,
    )

    viewer.plot_argument()


def test_rational_function():
    """
    Test:

        f(z) = (z - 1) / ((z + 1)(z - 2))

    Expected:
        - zero near z = 1,
        - poles near z = -1 and z = 2,
        - no branch cuts.
    """

    def f(z):
        return (z - 1) / ((z + 1) * (z - 2))

    viewer = SimpleComplexStructureViewer(
        f,
        xlim=(-3, 3),
        ylim=(-3, 3),
        resolution=300,
        zero_threshold=0.05,
        pole_threshold=50,
        phase_jump_threshold=2.8,
        cluster_radius=0.08,
    )

    viewer.plot_argument()


if __name__ == "__main__":
    mp.dps = 30

    # Main test from the discussion.
    test_target_function()

    # Other tests:
    # test_sqrt_z_minus_1()
    # test_rational_function()
