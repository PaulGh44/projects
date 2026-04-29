"""
complex_structure_finder.py

Numerical complex-structure viewer and heuristic finder.

This module tries to display the complex structure of a black-box complex function f(z):
    - phase portrait
    - log modulus
    - approximate zeros
    - approximate poles
    - candidate branch points by numerical monodromy
    - automatically chosen numerical branch cuts

Important mathematical warning
------------------------------
For a completely general black-box function, there is no perfect numerical method that can
rigorously distinguish poles, essential singularities, branch points, and discontinuities.
This class is therefore a diagnostic tool.

The key idea is:
    - poles are detected by large |f|;
    - zeros are detected by small |f|;
    - branch points are detected by monodromy:
          compare f(z0 + r) before and after one loop around z0.
      If f_end != f_start, the point is treated as a branch-point candidate.

This works well for functions like:
    sqrt(z-a),
    log(z-a),
    sqrt((z-a)(z-b)),
    rational functions times square roots,
    determinants under square roots.

Usage:
    python complex_structure_finder.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal

import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp


Mode = Literal["phase", "logabs", "minus_logabs", "combined"]


@dataclass
class ComplexStructureFinder:
    f: Callable[[complex], complex]

    xlim: tuple[float, float] = (-2.0, 2.0)
    ylim: tuple[float, float] = (-2.0, 2.0)
    resolution: int = 250

    zero_threshold: float = 1e-6
    pole_threshold: float = 1e6

    # Monodromy settings
    monodromy_radius: float = 5e-3
    monodromy_points: int = 160
    monodromy_abs_threshold: float = 1e-4
    monodromy_rel_threshold: float = 1e-3

    # Clustering settings
    cluster_radius: float = 0.05

    # Internal data
    zeros: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    poles: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    branch_points: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    branch_cuts: list[np.ndarray] = field(default_factory=list)

    def safe_eval(self, z: complex) -> complex:
        """
        Safely evaluate f(z), returning nan+inan if evaluation fails.
        """
        try:
            return complex(self.f(mp.mpc(z)))
        except Exception:
            return np.nan + 1j * np.nan

    def evaluate_grid(self):
        """
        Evaluate f on the rectangular grid.

        Returns:
            X, Y, Z where Z[i,j] = f(X[i,j] + i Y[i,j]).
        """
        x = np.linspace(self.xlim[0], self.xlim[1], self.resolution)
        y = np.linspace(self.ylim[0], self.ylim[1], self.resolution)

        X, Y = np.meshgrid(x, y)
        Z = np.empty_like(X, dtype=complex)

        for i in range(self.resolution):
            for j in range(self.resolution):
                Z[i, j] = self.safe_eval(X[i, j] + 1j * Y[i, j])

        return X, Y, Z

    @staticmethod
    def phase_data(Z):
        return np.angle(Z)

    @staticmethod
    def log_abs_data(Z, eps=1e-300):
        return np.log10(np.abs(Z) + eps)

    @staticmethod
    def minus_log_abs_data(Z, eps=1e-300):
        return -np.log10(np.abs(Z) + eps)

    def detect_zeros_from_grid(self, X, Y, Z):
        """
        Detect approximate zeros by thresholding |f|.

        This returns many grid points near the same zero.
        Use cluster_points afterwards to compress them.
        """
        mask = np.isfinite(np.abs(Z)) & (np.abs(Z) < self.zero_threshold)
        pts = X[mask] + 1j * Y[mask]
        self.zeros = self.cluster_points(pts)
        return self.zeros

    def detect_poles_from_grid(self, X, Y, Z):
        """
        Detect approximate poles by thresholding |f|.

        This returns many grid points near the same pole.
        Use cluster_points afterwards to compress them.
        """
        mask = np.isfinite(np.abs(Z)) & (np.abs(Z) > self.pole_threshold)
        pts = X[mask] + 1j * Y[mask]
        self.poles = self.cluster_points(pts)
        return self.poles

    def cluster_points(self, points: Iterable[complex], radius: float | None = None):
        """
        Very simple greedy clustering of nearby points.

        It is intentionally dependency-free. For high precision work, replace this
        by scipy.cluster or DBSCAN.
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
                keep = []
                center = np.mean(cluster)

                for p in unused:
                    if abs(p - center) < radius:
                        cluster.append(p)
                        changed = True
                    else:
                        keep.append(p)

                unused = keep

            clusters.append(np.mean(cluster))

        return np.array(clusters, dtype=complex)

    def monodromy_loop_values(self, z0: complex, radius: float | None = None, npts: int | None = None):
        """
        Evaluate f along a loop around z0.

        Returns:
            values along the loop.
        """
        if radius is None:
            radius = self.monodromy_radius

        if npts is None:
            npts = self.monodromy_points

        z0 = complex(z0)
        values = []

        for k in range(npts + 1):
            theta = 2 * np.pi * k / npts
            z = z0 + radius * np.exp(1j * theta)
            values.append(self.safe_eval(z))

        return np.array(values, dtype=complex)

    def monodromy_score(self, z0: complex, radius: float | None = None, npts: int | None = None):
        """
        Compare final and initial values of f along a loop.

        Returns:
            score, f_start, f_end

        The score is relative when possible:
            |f_end - f_start| / max(1, |f_start|)
        """
        vals = self.monodromy_loop_values(z0, radius=radius, npts=npts)

        if not np.all(np.isfinite(vals)):
            return np.inf, vals[0], vals[-1]

        f_start = vals[0]
        f_end = vals[-1]

        abs_diff = abs(f_end - f_start)
        rel_diff = abs_diff / max(1.0, abs(f_start))

        return rel_diff, f_start, f_end

    def is_branch_point_by_monodromy(self, z0: complex, radius: float | None = None):
        """
        Decide whether z0 behaves like a branch point according to monodromy.

        This is a heuristic. It is most meaningful when z0 is already a candidate
        singular point or zero of something under a square root/log.
        """
        score, f_start, f_end = self.monodromy_score(z0, radius=radius)

        abs_diff = abs(f_end - f_start)
        rel_diff = score

        return (
            abs_diff > self.monodromy_abs_threshold
            and rel_diff > self.monodromy_rel_threshold
        )

    def candidate_points_from_phase_defects(self, X, Y, Z, phase_jump_threshold=np.pi):
        """
        Build candidate points by locating strong phase discontinuities.

        This is useful for functions with principal branches, because branch cuts
        show up as phase jumps. Branch points are often endpoints of those jumps.

        This routine returns grid locations where neighboring phase values jump
        strongly. It is not yet the final branch-point list.
        """
        phase = np.angle(Z)

        finite = np.isfinite(phase)
        phase = np.where(finite, phase, np.nan)

        dx_jump = np.abs(np.angle(np.exp(1j * (phase[:, 1:] - phase[:, :-1]))))
        dy_jump = np.abs(np.angle(np.exp(1j * (phase[1:, :] - phase[:-1, :]))))

        pts = []

        rows, cols = np.where(dx_jump > phase_jump_threshold)
        for i, j in zip(rows, cols):
            pts.append(0.5 * (X[i, j] + X[i, j + 1]) + 1j * 0.5 * (Y[i, j] + Y[i, j + 1]))

        rows, cols = np.where(dy_jump > phase_jump_threshold)
        for i, j in zip(rows, cols):
            pts.append(0.5 * (X[i, j] + X[i + 1, j]) + 1j * 0.5 * (Y[i, j] + Y[i + 1, j]))

        return np.array(pts, dtype=complex)

    def endpoints_from_cut_cloud(self, points):
        """
        Given many points lying near phase-jump curves, return crude endpoint candidates.

        For one simple branch cut, this often returns the two endpoints of the visible cut.
        For a ray going to the boundary, one endpoint is the branch point, the other is on
        the plotting boundary.

        This is heuristic and should be filtered by monodromy afterwards.
        """
        points = np.array(points, dtype=complex)

        if len(points) < 2:
            return np.array([], dtype=complex)

        # Compress the cloud a bit.
        clustered = self.cluster_points(points, radius=2 * self.cluster_radius)

        if len(clustered) <= 2:
            return clustered

        # Pick points with extreme projection along the main principal axis.
        xy = np.column_stack([clustered.real, clustered.imag])
        center = xy.mean(axis=0)
        centered = xy - center

        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]

        proj = centered @ direction
        p_min = clustered[np.argmin(proj)]
        p_max = clustered[np.argmax(proj)]

        return np.array([p_min, p_max], dtype=complex)

    def detect_branch_points(
        self,
        use_zeros: bool = True,
        use_poles: bool = True,
        use_phase_defects: bool = True,
        extra_candidates: Iterable[complex] | None = None,
        radius: float | None = None,
    ):
        """
        Heuristically detect branch points.

        Candidate sources:
            - approximate zeros,
            - approximate poles,
            - endpoints of numerical phase-jump curves,
            - user-provided extra candidates.

        Each candidate is then tested by monodromy.
        """
        X, Y, Z = self.evaluate_grid()

        candidates = []

        if use_zeros:
            zeros = self.detect_zeros_from_grid(X, Y, Z)
            candidates.extend(list(zeros))

        if use_poles:
            poles = self.detect_poles_from_grid(X, Y, Z)
            candidates.extend(list(poles))

        if use_phase_defects:
            cut_cloud = self.candidate_points_from_phase_defects(X, Y, Z)
            endpoints = self.endpoints_from_cut_cloud(cut_cloud)
            candidates.extend(list(endpoints))

        if extra_candidates is not None:
            candidates.extend(list(extra_candidates))

        candidates = self.cluster_points(candidates)

        detected = []
        for z0 in candidates:
            if self.is_branch_point_by_monodromy(z0, radius=radius):
                detected.append(z0)

        self.branch_points = self.cluster_points(detected)
        return self.branch_points

    def choose_branch_cuts(
        self,
        strategy: Literal["ray_right", "ray_left", "pairwise"] = "ray_right",
        length: float | None = None,
        npts: int = 300,
        angle: float = 0.0,
    ):
        """
        Choose numerical branch cuts from detected branch points.

        strategy:
            "ray_right":
                ray from each branch point to the right.
            "ray_left":
                ray from each branch point to the left.
            "pairwise":
                connect branch points pairwise by straight cuts.
                If there is an odd number of branch points, the last one is sent to the right.

        Branch cuts are not intrinsic. This routine just chooses convenient cuts.
        """
        if length is None:
            length = max(self.xlim[1] - self.xlim[0], self.ylim[1] - self.ylim[0])

        bps = np.array(self.branch_points, dtype=complex)
        self.branch_cuts = []

        if len(bps) == 0:
            return self.branch_cuts

        if strategy == "ray_right":
            for z0 in bps:
                t = np.linspace(0.0, length, npts)
                cut = z0 + t * np.exp(1j * angle)
                self.branch_cuts.append(cut)

        elif strategy == "ray_left":
            for z0 in bps:
                t = np.linspace(0.0, length, npts)
                cut = z0 - t
                self.branch_cuts.append(cut)

        elif strategy == "pairwise":
            # Greedy nearest-neighbor pairing
            unused = list(bps)

            while len(unused) >= 2:
                z0 = unused.pop(0)
                distances = [abs(z - z0) for z in unused]
                idx = int(np.argmin(distances))
                z1 = unused.pop(idx)

                t = np.linspace(0.0, 1.0, npts)
                cut = z0 + t * (z1 - z0)
                self.branch_cuts.append(cut)

            if len(unused) == 1:
                z0 = unused[0]
                t = np.linspace(0.0, length, npts)
                cut = z0 + t
                self.branch_cuts.append(cut)

        else:
            raise ValueError("Unknown cut strategy.")

        return self.branch_cuts

    def auto_analyze(
        self,
        cut_strategy: Literal["ray_right", "ray_left", "pairwise"] = "ray_right",
        extra_branch_candidates: Iterable[complex] | None = None,
    ):
        """
        Run the full heuristic pipeline:
            1. sample grid,
            2. detect zeros and poles,
            3. detect branch points by monodromy,
            4. choose branch cuts.
        """
        self.detect_branch_points(extra_candidates=extra_branch_candidates)
        self.choose_branch_cuts(strategy=cut_strategy)
        return {
            "zeros": self.zeros,
            "poles": self.poles,
            "branch_points": self.branch_points,
            "branch_cuts": self.branch_cuts,
        }

    def plot(
        self,
        mode: Mode = "phase",
        show_zeros: bool = True,
        show_poles: bool = True,
        show_branch_points: bool = True,
        show_branch_cuts: bool = True,
        zero_color: str = "white",
        pole_color: str = "black",
        branch_point_color: str = "red",
        branch_cut_color: str = "red",
        figsize=(8, 7),
    ):
        """
        Plot the complex structure.

        mode:
            "phase":
                plot arg f.
            "logabs":
                plot log10 |f|.
            "minus_logabs":
                plot -log10 |f|. Large positive values indicate zeros.
            "combined":
                phase portrait with log|f| contour lines.
        """
        X, Y, Z = self.evaluate_grid()

        # Refresh zeros/poles for plotting if requested.
        if show_zeros:
            self.detect_zeros_from_grid(X, Y, Z)

        if show_poles:
            self.detect_poles_from_grid(X, Y, Z)

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
            data = self.minus_log_abs_data(Z)
            im = ax.contourf(X, Y, data, levels=100)
            plt.colorbar(im, ax=ax, label=r"$-\log_{10}|f(z)|$")

        elif mode == "combined":
            phase = self.phase_data(Z)
            logabs = self.log_abs_data(Z)

            im = ax.contourf(X, Y, phase, levels=100, cmap="hsv")
            plt.colorbar(im, ax=ax, label=r"$\arg f(z)$")

            ax.contour(
                X,
                Y,
                logabs,
                levels=15,
                colors="black",
                linewidths=0.4,
                alpha=0.45,
            )

        else:
            raise ValueError("mode must be phase, logabs, minus_logabs, or combined.")

        if show_zeros and len(self.zeros) > 0:
            ax.scatter(
                self.zeros.real,
                self.zeros.imag,
                s=35,
                c=zero_color,
                marker="o",
                edgecolors="black",
                linewidths=0.4,
                label="approx. zeros",
            )

        if show_poles and len(self.poles) > 0:
            ax.scatter(
                self.poles.real,
                self.poles.imag,
                s=50,
                c=pole_color,
                marker="x",
                label="approx. poles",
            )

        if show_branch_cuts and len(self.branch_cuts) > 0:
            first = True
            for cut in self.branch_cuts:
                ax.plot(
                    cut.real,
                    cut.imag,
                    color=branch_cut_color,
                    linewidth=2.0,
                    label="chosen branch cut" if first else None,
                )
                first = False

        if show_branch_points and len(self.branch_points) > 0:
            ax.scatter(
                self.branch_points.real,
                self.branch_points.imag,
                s=120,
                c=branch_point_color,
                marker="*",
                edgecolors="black",
                linewidths=0.5,
                label="detected branch points",
            )

        ax.axhline(0, linewidth=0.8, color="black", alpha=0.5)
        ax.axvline(0, linewidth=0.8, color="black", alpha=0.5)

        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel(r"$\operatorname{Re} z$")
        ax.set_ylabel(r"$\operatorname{Im} z$")
        ax.set_title("Complex structure finder")

        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys())

        plt.show()

        return X, Y, Z


# ============================================================
# Test cases
# ============================================================

def test_sqrt_single_branch_point():
    """
    Test with f(z) = sqrt(z - 1).

    The principal branch has a branch point at z = 1 and a default
    numerical branch cut inherited from mpmath's sqrt.
    """

    def f(z):
        return mp.sqrt(z - 1)

    viewer = ComplexStructureFinder(
        f,
        xlim=(-3, 3),
        ylim=(-3, 3),
        resolution=220,
        zero_threshold=1e-3,
        pole_threshold=1e6,
        monodromy_radius=1e-2,
        monodromy_abs_threshold=1e-3,
        monodromy_rel_threshold=1e-3,
        cluster_radius=0.08,
    )

    # For sqrt(z-1), the zero at z=1 is also the branch point.
    # The algorithm should detect it from zero candidates + monodromy.
    result = viewer.auto_analyze(cut_strategy="ray_right")

    print("=== sqrt(z - 1) ===")
    print("Detected zeros:", result["zeros"])
    print("Detected poles:", result["poles"])
    print("Detected branch points:", result["branch_points"])

    viewer.plot(mode="phase")
    viewer.plot(mode="combined")


def test_sqrt_two_branch_points():
    """
    Test with f(z) = sqrt((z + 1)(z - 1)).

    Expected branch points: z=-1 and z=1.
    """

    def f(z):
        return mp.sqrt((z + 1) * (z - 1))

    viewer = ComplexStructureFinder(
        f,
        xlim=(-3, 3),
        ylim=(-3, 3),
        resolution=240,
        zero_threshold=1e-3,
        pole_threshold=1e6,
        monodromy_radius=1e-2,
        monodromy_abs_threshold=1e-3,
        monodromy_rel_threshold=1e-3,
        cluster_radius=0.08,
    )

    result = viewer.auto_analyze(cut_strategy="pairwise")

    print("=== sqrt((z + 1)(z - 1)) ===")
    print("Detected zeros:", result["zeros"])
    print("Detected poles:", result["poles"])
    print("Detected branch points:", result["branch_points"])

    viewer.plot(mode="phase")
    viewer.plot(mode="combined")


def test_rational_function():
    """
    Test with a meromorphic function.

    f(z) = (z - 1) / ((z + 1)(z - 2))

    Expected:
        zero at z=1,
        poles at z=-1 and z=2,
        no branch points.
    """

    def f(z):
        return (z - 1) / ((z + 1) * (z - 2))

    viewer = ComplexStructureFinder(
        f,
        xlim=(-3, 3),
        ylim=(-3, 3),
        resolution=240,
        zero_threshold=1e-3,
        pole_threshold=1e3,
        monodromy_radius=1e-2,
        cluster_radius=0.08,
    )

    result = viewer.auto_analyze(cut_strategy="ray_right")

    print("=== rational function ===")
    print("Detected zeros:", result["zeros"])
    print("Detected poles:", result["poles"])
    print("Detected branch points:", result["branch_points"])

    viewer.plot(mode="phase")
    viewer.plot(mode="combined")


if __name__ == "__main__":
    mp.dps = 30

    # Start with the square-root test requested.
    test_sqrt_single_branch_point()

    # Uncomment for more tests:
    # test_sqrt_two_branch_points()
    # test_rational_function()
