"""
complex_structure_finder_v3.py

Numerical complex-structure finder/viewer for black-box complex functions.

It tries to display:
    - phase portrait,
    - log modulus,
    - approximate zeros,
    - approximate poles,
    - numerical branch cuts chosen by the implementation,
    - branch-point candidates,
    - detected branch points by monodromy.

Important warning
-----------------
For a completely general black-box function, there is no perfect numerical method that
rigorously distinguishes zeros, poles, essential singularities, branch points, and branch cuts.
This is a diagnostic tool.

The design philosophy is:
    1. Detect numerical branch cuts as phase-jump curves.
    2. Extract possible branch-point candidates from endpoints/junctions of those cuts.
    3. Add ordinary zero/pole candidates.
    4. Optionally add analytic candidates, e.g. roots of the expression under a sqrt.
    5. Test candidates by monodromy.

Run:
    python complex_structure_finder_v3.py
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

    xlim: tuple[float, float] = (-3.0, 3.0)
    ylim: tuple[float, float] = (-3.0, 3.0)
    resolution: int = 300

    # Ordinary zero/pole detection thresholds.
    # For sqrt-type zeros, use a relatively loose zero_threshold.
    zero_threshold: float = 1e-2
    pole_threshold: float = 1e4

    # Phase-jump detection.
    # Principal sqrt/log cuts typically produce jumps close to pi.
    phase_jump_threshold: float = 2.2

    # Monodromy settings.
    monodromy_radius: float = 4e-2
    monodromy_points: int = 240
    monodromy_abs_threshold: float = 1e-4
    monodromy_rel_threshold: float = 1e-4

    # Clustering and boundary filtering.
    cluster_radius: float = 0.08
    boundary_tol: float = 0.15

    # Internal data.
    zeros: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    poles: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    phase_cut_points: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    branch_point_candidates: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
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

    @staticmethod
    def phase_data(Z):
        return np.angle(Z)

    @staticmethod
    def log_abs_data(Z, eps=1e-300):
        return np.log10(np.abs(Z) + eps)

    @staticmethod
    def minus_log_abs_data(Z, eps=1e-300):
        return -np.log10(np.abs(Z) + eps)

    def is_near_boundary(self, z: complex) -> bool:
        """
        True if z is close to the plotting boundary.
        Boundary endpoints of numerical cuts are usually not branch points.
        """
        z = complex(z)
        return (
            abs(z.real - self.xlim[0]) < self.boundary_tol
            or abs(z.real - self.xlim[1]) < self.boundary_tol
            or abs(z.imag - self.ylim[0]) < self.boundary_tol
            or abs(z.imag - self.ylim[1]) < self.boundary_tol
        )

    def cluster_points(self, points: Iterable[complex], radius: float | None = None):
        """
        Simple greedy clustering of nearby points.

        This is deliberately dependency-free. For more serious analysis,
        replace by scipy.spatial / DBSCAN.
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

    # ------------------------------------------------------------------
    # Zero and pole detectors
    # ------------------------------------------------------------------

    def detect_zeros_from_grid(self, X, Y, Z):
        """
        Detect zeros using both:
            1. an absolute threshold |f| < zero_threshold,
            2. local minima of |f|.

        This is better for square-root zeros, where |f| ~ sqrt(|z-z0|)
        and therefore may not become extremely small on a finite grid.
        """
        A = np.abs(Z)
        finite = np.isfinite(A)

        candidates = []

        # Absolute threshold.
        mask = finite & (A < self.zero_threshold)
        candidates.extend(list(X[mask] + 1j * Y[mask]))

        # Local-minimum detector.
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

                    # Significantly smaller than its local neighborhood.
                    if local_median > 0 and center < 0.5 * local_median:
                        candidates.append(X[i, j] + 1j * Y[i, j])

        self.zeros = self.cluster_points(candidates)
        return self.zeros

    def detect_poles_from_grid(self, X, Y, Z):
        """
        Detect ordinary pole candidates by thresholding |f|.
        """
        A = np.abs(Z)
        finite = np.isfinite(A)

        mask = finite & (A > self.pole_threshold)
        pts = X[mask] + 1j * Y[mask]

        self.poles = self.cluster_points(pts)
        return self.poles

    # ------------------------------------------------------------------
    # Numerical branch-cut detector
    # ------------------------------------------------------------------

    def detect_phase_cut_points(self, X, Y, Z):
        """
        Detect numerical branch cuts as phase-jump curves.

        If f uses principal sqrt/log internally, its chosen branch cuts appear as
        jumps in arg(f). This detects points sitting between neighboring grid
        cells where arg(f) jumps by more than phase_jump_threshold.
        """
        phase = np.angle(Z)

        # Difference of phases modulo 2pi, in [0, pi].
        dx_jump = np.abs(np.angle(np.exp(1j * (phase[:, 1:] - phase[:, :-1]))))
        dy_jump = np.abs(np.angle(np.exp(1j * (phase[1:, :] - phase[:-1, :]))))

        pts = []

        rows, cols = np.where(dx_jump > self.phase_jump_threshold)
        for i, j in zip(rows, cols):
            pts.append(
                0.5 * (X[i, j] + X[i, j + 1])
                + 1j * 0.5 * (Y[i, j] + Y[i, j + 1])
            )

        rows, cols = np.where(dy_jump > self.phase_jump_threshold)
        for i, j in zip(rows, cols):
            pts.append(
                0.5 * (X[i, j] + X[i + 1, j])
                + 1j * 0.5 * (Y[i, j] + Y[i + 1, j])
            )

        self.phase_cut_points = np.array(pts, dtype=complex)
        return self.phase_cut_points

    # ------------------------------------------------------------------
    # Branch-point candidates
    # ------------------------------------------------------------------

    def branch_point_candidates_from_phase_cuts(self, cut_points: np.ndarray):
        """
        Extract branch-point candidates from the numerical branch-cut cloud.

        This version handles nontrivial cut geometry better than just taking the two
        endpoints of one line. It is intended for examples like mp.sqrt(z**2 - 1),
        whose inherited principal branch cut has a cross-like geometry.

        Strategy:
            - cluster the phase-jump cloud into coarse points;
            - identify endpoints of the numerical cut cloud;
            - identify junction/corner-like points;
            - add extreme points of connected components as backup;
            - discard boundary artifacts.
        """
        cut_points = np.array(cut_points, dtype=complex)

        if len(cut_points) == 0:
            self.branch_point_candidates = np.array([], dtype=complex)
            return self.branch_point_candidates

        coarse = self.cluster_points(cut_points, radius=self.cluster_radius)

        if len(coarse) == 0:
            self.branch_point_candidates = np.array([], dtype=complex)
            return self.branch_point_candidates

        candidates = []
        neighbor_radius = 3.0 * self.cluster_radius

        # Degree-based endpoint/junction detection.
        for z in coarse:
            if self.is_near_boundary(z):
                continue

            neighbors = [w for w in coarse if 0 < abs(w - z) < neighbor_radius]
            degree = len(neighbors)

            # Endpoints of numerical cuts.
            if degree <= 1:
                candidates.append(z)

            # Junctions/corners are also candidates.
            if degree >= 3:
                candidates.append(z)

        # Connected components of coarse cut cloud.
        components = []
        unused = list(coarse)
        component_radius = 3.0 * self.cluster_radius

        while unused:
            seed = unused.pop(0)
            comp = [seed]

            changed = True
            while changed:
                changed = False
                keep = []

                for p in unused:
                    if min(abs(p - q) for q in comp) < component_radius:
                        comp.append(p)
                        changed = True
                    else:
                        keep.append(p)

                unused = keep

            components.append(np.array(comp, dtype=complex))

        # Add principal-axis extremes from each component as backup candidates.
        for comp in components:
            if len(comp) < 2:
                continue

            xy = np.column_stack([comp.real, comp.imag])
            center = xy.mean(axis=0)
            centered = xy - center

            try:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                direction = vh[0]
                proj = centered @ direction

                endpoints = np.array([
                    comp[np.argmin(proj)],
                    comp[np.argmax(proj)],
                ])

                for z in endpoints:
                    if not self.is_near_boundary(z):
                        candidates.append(z)

            except Exception:
                pass

        self.branch_point_candidates = self.cluster_points(candidates)
        return self.branch_point_candidates

    # ------------------------------------------------------------------
    # Monodromy test
    # ------------------------------------------------------------------

    def monodromy_loop_values(
        self,
        z0: complex,
        radius: float | None = None,
        npts: int | None = None,
    ):
        """
        Evaluate f along one circular loop around z0.

        Note:
            If f is implemented with principal branches, following this circle by direct
            calls to f does not analytically continue f sheet-by-sheet. Still, the jump
            in the principal value is often a good diagnostic of branch-type behavior.
        """
        if radius is None:
            radius = self.monodromy_radius

        if npts is None:
            npts = self.monodromy_points

        values = []
        z0 = complex(z0)

        for k in range(npts + 1):
            theta = 2 * np.pi * k / npts
            z = z0 + radius * np.exp(1j * theta)
            values.append(self.safe_eval(z))

        return np.array(values, dtype=complex)

    def monodromy_score(
        self,
        z0: complex,
        radius: float | None = None,
        npts: int | None = None,
    ):
        """
        Compare initial and final values after a numerical loop.

        Returns:
            rel_score, f_start, f_end
        """
        vals = self.monodromy_loop_values(z0, radius=radius, npts=npts)

        if not np.all(np.isfinite(vals)):
            return np.inf, vals[0], vals[-1]

        f_start = vals[0]
        f_end = vals[-1]

        abs_diff = abs(f_end - f_start)
        rel_diff = abs_diff / max(1.0, abs(f_start), abs(f_end))

        return rel_diff, f_start, f_end

    def is_branch_point_by_monodromy(self, z0: complex, radius: float | None = None):
        """
        Decide whether z0 behaves like a branch point according to monodromy.
        """
        score, f_start, f_end = self.monodromy_score(z0, radius=radius)
        abs_diff = abs(f_end - f_start)

        return (
            abs_diff > self.monodromy_abs_threshold
            and score > self.monodromy_rel_threshold
        )

    def classify_branch_point_type(self, z0: complex, small_radius: float | None = None):
        """
        Very rough classification of a branch point.

        It samples |f| near z0:
            - if |f| is small near z0, call it "branch zero";
            - if |f| is large near z0, call it "branch singularity";
            - otherwise call it "branch point".

        This is heuristic but useful for examples such as:
            sqrt(z-1)                     -> branch zero
            sqrt(z+1)/(z+1)               -> branch singularity
        """
        if small_radius is None:
            small_radius = self.monodromy_radius

        values = []
        z0 = complex(z0)

        for theta in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            z = z0 + small_radius * np.exp(1j * theta)
            val = self.safe_eval(z)
            if np.isfinite(val):
                values.append(abs(val))

        if len(values) == 0:
            return "branch point"

        med = float(np.median(values))

        if med < self.zero_threshold:
            return "branch zero"

        if med > self.pole_threshold:
            return "branch singularity"

        return "branch point"

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def detect_branch_points(
        self,
        X=None,
        Y=None,
        Z=None,
        extra_candidates: Iterable[complex] | None = None,
        radius: float | None = None,
    ):
        """
        Main automatic branch-point detector.

        Sources of candidates:
            - grid zeros,
            - grid poles,
            - numerical branch-cut endpoints/junctions,
            - optional extra analytic candidates.
        """
        if X is None or Y is None or Z is None:
            X, Y, Z = self.evaluate_grid()

        zeros = self.detect_zeros_from_grid(X, Y, Z)
        poles = self.detect_poles_from_grid(X, Y, Z)

        cut_points = self.detect_phase_cut_points(X, Y, Z)
        endpoint_candidates = self.branch_point_candidates_from_phase_cuts(cut_points)

        candidates = []
        candidates.extend(list(zeros))
        candidates.extend(list(poles))
        candidates.extend(list(endpoint_candidates))

        if extra_candidates is not None:
            candidates.extend(list(extra_candidates))

        candidates = self.cluster_points(candidates)

        detected = []

        for z0 in candidates:
            if self.is_near_boundary(z0):
                continue

            if self.is_branch_point_by_monodromy(z0, radius=radius):
                detected.append(z0)

        self.branch_points = self.cluster_points(detected)
        return self.branch_points

    def choose_branch_cuts(
        self,
        strategy: Literal["from_detected_cuts", "ray_right", "ray_left", "pairwise"] = "from_detected_cuts",
        length: float | None = None,
        npts: int = 400,
        angle: float = 0.0,
    ):
        """
        Choose numerical branch cuts.

        strategy:
            "from_detected_cuts":
                display the phase-jump points. This best shows the cuts already
                chosen by your numerical implementation.

            "ray_right":
                draw a ray to the right from each detected branch point.

            "ray_left":
                draw a ray to the left from each detected branch point.

            "pairwise":
                connect branch points pairwise.
        """
        self.branch_cuts = []

        if length is None:
            length = max(self.xlim[1] - self.xlim[0], self.ylim[1] - self.ylim[0])

        if strategy == "from_detected_cuts":
            if len(self.phase_cut_points) > 0:
                # Store as a point cloud. plot() will scatter these points.
                self.branch_cuts = [self.phase_cut_points]
            return self.branch_cuts

        bps = np.array(self.branch_points, dtype=complex)

        if len(bps) == 0:
            return self.branch_cuts

        if strategy == "ray_right":
            for z0 in bps:
                t = np.linspace(0.0, length, npts)
                self.branch_cuts.append(z0 + t * np.exp(1j * angle))

        elif strategy == "ray_left":
            for z0 in bps:
                t = np.linspace(0.0, length, npts)
                self.branch_cuts.append(z0 - t)

        elif strategy == "pairwise":
            unused = list(bps)

            while len(unused) >= 2:
                z0 = unused.pop(0)
                idx = int(np.argmin([abs(z - z0) for z in unused]))
                z1 = unused.pop(idx)

                t = np.linspace(0.0, 1.0, npts)
                self.branch_cuts.append(z0 + t * (z1 - z0))

            if len(unused) == 1:
                z0 = unused[0]
                t = np.linspace(0.0, length, npts)
                self.branch_cuts.append(z0 + t)

        else:
            raise ValueError("Unknown branch-cut strategy.")

        return self.branch_cuts

    def auto_analyze(
        self,
        cut_strategy: Literal["from_detected_cuts", "ray_right", "ray_left", "pairwise"] = "from_detected_cuts",
        extra_branch_candidates: Iterable[complex] | None = None,
    ):
        """
        Run the full diagnostic pipeline.
        """
        X, Y, Z = self.evaluate_grid()

        self.detect_branch_points(
            X,
            Y,
            Z,
            extra_candidates=extra_branch_candidates,
        )

        self.choose_branch_cuts(strategy=cut_strategy)

        branch_types = {
            complex(z): self.classify_branch_point_type(z)
            for z in self.branch_points
        }

        return {
            "zeros": self.zeros,
            "poles": self.poles,
            "phase_cut_points": self.phase_cut_points,
            "branch_point_candidates": self.branch_point_candidates,
            "branch_points": self.branch_points,
            "branch_point_types": branch_types,
            "branch_cuts": self.branch_cuts,
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        mode: Mode = "phase",
        show_zeros: bool = True,
        show_poles: bool = True,
        show_branch_points: bool = False,
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
                s=45,
                c=zero_color,
                marker="o",
                edgecolors="black",
                linewidths=0.5,
                label="approx. zeros",
            )

        if show_poles and len(self.poles) > 0:
            ax.scatter(
                self.poles.real,
                self.poles.imag,
                s=70,
                c=pole_color,
                marker="x",
                linewidths=1.8,
                label="approx. poles",
            )

        if show_branch_cuts and len(self.branch_cuts) > 0:
            first = True
            for cut in self.branch_cuts:
                if len(cut) > 0:
                    ax.scatter(
                        cut.real,
                        cut.imag,
                        s=6,
                        c=branch_cut_color,
                        marker=".",
                        alpha=0.75,
                        label="detected numerical branch cut" if first else None,
                    )
                    first = False

        if len(self.branch_point_candidates) > 0:
            ax.scatter(
                self.branch_point_candidates.real,
                self.branch_point_candidates.imag,
                s=80,
                facecolors="none",
                edgecolors="yellow",
                linewidths=1.2,
                marker="o",
                label="branch-point candidates",
            )

        if show_branch_points and len(self.branch_points) > 0:
            ax.scatter(
                self.branch_points.real,
                self.branch_points.imag,
                s=170,
                c=branch_point_color,
                marker="*",
                edgecolors="black",
                linewidths=0.7,
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
# Tests
# ============================================================

def test_target_function():
    """
    Main test requested:

        f(z) = sqrt(z^2 - 1) / ((z + 1)(z - 2))

    Expected analytic structure:
        - z = 1:
            zero and branch point.
        - z = -1:
            branch-point singularity, because
                sqrt((z-1)(z+1))/(z+1) ~ const / sqrt(z+1).
        - z = 2:
            ordinary pole.
        - numerical branch cuts:
            inherited from mpmath's principal sqrt applied to z^2 - 1.
    """

    def f(z):
        return mp.sqrt(z**2 - 1) / ((z + 1) * (z - 2))

    viewer = ComplexStructureFinder(
        f,
        xlim=(-3, 3),
        ylim=(-3, 3),
        resolution=350,
        zero_threshold=0.15,
        pole_threshold=50,
        phase_jump_threshold=2.2,
        monodromy_radius=0.04,
        monodromy_abs_threshold=1e-3,
        monodromy_rel_threshold=1e-3,
        cluster_radius=0.08,
        boundary_tol=0.15,
    )

    # The extra candidates are analytically natural here:
    # they are the zeros of the expression under sqrt, z^2 - 1.
    result = viewer.auto_analyze(
        cut_strategy="from_detected_cuts",
        extra_branch_candidates=[-1, 1],
    )

    print("=== target function ===")
    print("f(z) = sqrt(z^2 - 1) / ((z + 1)(z - 2))")
    print()
    print("Detected zeros:")
    print(result["zeros"])
    print()
    print("Detected poles:")
    print(result["poles"])
    print()
    print("Branch-point candidates:")
    print(result["branch_point_candidates"])
    print()
    print("Detected branch points:")
    print(result["branch_points"])
    print()
    print("Branch-point types:")
    for z, typ in result["branch_point_types"].items():
        print(f"  {z}: {typ}")
    print()
    print("Number of detected numerical branch-cut points:")
    print(len(result["phase_cut_points"]))

    viewer.plot(mode="phase")
    viewer.plot(mode="combined")
    viewer.plot(mode="minus_logabs")


def test_sqrt_single_branch_point():
    """
    f(z) = sqrt(z - 1)

    Expected:
        branch point at z=1;
        numerical branch cut (-infty, 1] inherited from principal sqrt.
    """

    def f(z):
        return mp.sqrt(z - 1)

    viewer = ComplexStructureFinder(
        f,
        xlim=(-3, 3),
        ylim=(-3, 3),
        resolution=300,
        zero_threshold=0.12,
        pole_threshold=1e6,
        phase_jump_threshold=2.2,
        monodromy_radius=0.04,
        monodromy_abs_threshold=1e-3,
        monodromy_rel_threshold=1e-3,
        cluster_radius=0.08,
        boundary_tol=0.15,
    )

    result = viewer.auto_analyze(
        cut_strategy="from_detected_cuts",
        extra_branch_candidates=[1],
    )

    print("=== sqrt(z - 1) ===")
    print("Detected zeros:", result["zeros"])
    print("Detected poles:", result["poles"])
    print("Branch-point candidates:", result["branch_point_candidates"])
    print("Detected branch points:", result["branch_points"])
    print("Branch-point types:", result["branch_point_types"])
    print("Number of detected numerical branch-cut points:", len(result["phase_cut_points"]))

    viewer.plot(mode="phase")
    viewer.plot(mode="combined")


def test_sqrt_z2_minus_1():
    """
    f(z) = sqrt(z^2 - 1)

    Expected:
        branch points at z=-1 and z=1.

    Numerical branch cuts are inherited from mpmath's principal sqrt:
        preimage of (-infty,0] under z^2 - 1.
    """

    def f(z):
        return mp.sqrt(z**2 - 1)

    viewer = ComplexStructureFinder(
        f,
        xlim=(-3, 3),
        ylim=(-3, 3),
        resolution=350,
        zero_threshold=0.15,
        pole_threshold=1e6,
        phase_jump_threshold=2.2,
        monodromy_radius=0.04,
        monodromy_abs_threshold=1e-3,
        monodromy_rel_threshold=1e-3,
        cluster_radius=0.08,
        boundary_tol=0.15,
    )

    result = viewer.auto_analyze(
        cut_strategy="from_detected_cuts",
        extra_branch_candidates=[-1, 1],
    )

    print("=== sqrt(z^2 - 1) ===")
    print("Detected zeros:", result["zeros"])
    print("Detected poles:", result["poles"])
    print("Branch-point candidates:", result["branch_point_candidates"])
    print("Detected branch points:", result["branch_points"])
    print("Branch-point types:", result["branch_point_types"])
    print("Number of detected numerical branch-cut points:", len(result["phase_cut_points"]))

    viewer.plot(mode="phase")
    viewer.plot(mode="combined")


def test_rational_function():
    """
    Meromorphic control test.

    f(z) = (z - 1) / ((z + 1)(z - 2))

    Expected:
        zero near z=1;
        poles near z=-1 and z=2;
        no branch points.
    """

    def f(z):
        return (z - 1) / ((z + 1) * (z - 2))

    viewer = ComplexStructureFinder(
        f,
        xlim=(-3, 3),
        ylim=(-3, 3),
        resolution=300,
        zero_threshold=0.05,
        pole_threshold=50,
        phase_jump_threshold=2.8,
        monodromy_radius=0.04,
        cluster_radius=0.08,
        boundary_tol=0.15,
    )

    result = viewer.auto_analyze(cut_strategy="from_detected_cuts")

    print("=== rational function ===")
    print("Detected zeros:", result["zeros"])
    print("Detected poles:", result["poles"])
    print("Branch-point candidates:", result["branch_point_candidates"])
    print("Detected branch points:", result["branch_points"])
    print("Branch-point types:", result["branch_point_types"])
    print("Number of detected numerical branch-cut points:", len(result["phase_cut_points"]))

    viewer.plot(mode="phase")
    viewer.plot(mode="combined")


if __name__ == "__main__":
    mp.dps = 30

    # Main test from the discussion.
    test_target_function()

    # Other useful tests:
    # test_sqrt_single_branch_point()
    # test_sqrt_z2_minus_1()
    # test_rational_function()
