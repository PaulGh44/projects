# main.py

import numpy as np
import matplotlib.pyplot as plt

from Model import NoBoundaryModel
from Solveur import Solveur
from Power_spectrum import PowerSpectrum


def main():

    # --------------------------------------------------
    # 1. Define the background
    # --------------------------------------------------

    H = 1.0

    model = NoBoundaryModel(
        H=H
    )

    # --------------------------------------------------
    # 2. Numerical parameters
    # --------------------------------------------------

    delta = 1e-4
    t_final = 10.0

    n_min = 2
    n_max = 80

    # --------------------------------------------------
    # 3. Create solver
    # --------------------------------------------------

    solver = Solveur(
        model=model,
        N_eval=1000,
        rtol=1e-10,
        atol=1e-12,
        method="DOP853",
    )

    # --------------------------------------------------
    # 4. Solve modes and extract power spectrum
    # --------------------------------------------------

    results = []

    for n in range(n_min, n_max + 1):

        solution = solver.solve_W(
            n=n,
            delta=delta,
            t_final=t_final,
        )

        spectrum_point = PowerSpectrum.extract(solution)

        results.append(spectrum_point)

    # --------------------------------------------------
    # 5. Convert results to arrays
    # --------------------------------------------------

    n_values = np.array([
        result.n
        for result in results
    ])

    power = np.array([
        result.power
        for result in results
    ])

    dimensionless_power = np.array([
        result.dimensionless_power
        for result in results
    ])

    # --------------------------------------------------
    # 6. Compare with exact no-boundary result
    # --------------------------------------------------

    exact_power = (
        H**2
        / (2.0 * n_values * (n_values**2 - 1))
    )

    exact_dimensionless_power = (
        H**2 / (4.0 * np.pi**2)
    )

    print(" n       numerical P(n)       exact P(n)")

    for n, P_num, P_exact in zip(
        n_values,
        power,
        exact_power,
    ):
        print(
            f"{n:2d}    "
            f"{P_num:.12e}    "
            f"{P_exact:.12e}"
        )

    # --------------------------------------------------
    # 7. Plot dimensionless power spectrum
    # --------------------------------------------------



    # Exact no-boundary result
    Delta_exact = H**2 / (4.0 * np.pi**2)

    # Relative error
    relative_error = (
        dimensionless_power - Delta_exact
    ) / Delta_exact


    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(6.5, 5.5),
        sharex=True,
        gridspec_kw={
            "height_ratios": [3, 1],
            "hspace": 0.08
        }
    )

    # ============================================================
    # Upper panel: power spectrum
    # ============================================================

    ax1.plot(
        n_values,
        dimensionless_power,
        "o",
        markersize=5,
        label="Numerical"
    )

    ax1.axhline(
        Delta_exact,
        linestyle="--",
        linewidth=1.8,
        label="Analytic"
    )

    ax1.set_ylabel(r"$\Delta_n^\Phi$")

    # Important: prevent misleading offset notation
    ax1.ticklabel_format(
        axis="y",
        style="plain",
        useOffset=False
    )

    # Show a physically sensible window around the exact value
    ax1.set_ylim(
        0.99 * Delta_exact,
        1.01 * Delta_exact
    )

    ax1.legend(frameon=False)
    ax1.grid(alpha=0.2)


    # ============================================================
    # Lower panel: relative numerical error
    # ============================================================

    ax2.plot(
        n_values,
        1e6 * relative_error,
        "o",
        markersize=4
    )

    ax2.axhline(
        0.0,
        linestyle="--",
        linewidth=1.2
    )

    ax2.set_xlabel(r"$n$")
    ax2.set_ylabel(
        r"$10^6\,(\Delta_n^\Phi/\Delta_{\rm exact}^\Phi-1)$"
    )

    ax2.grid(alpha=0.2)


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()