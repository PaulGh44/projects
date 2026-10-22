# main.py

import numpy as np
import matplotlib.pyplot as plt

from Model import NoBoundaryModel
from Model import WineglassModel
from Solveur import Solveur
from Power_spectrum import PowerSpectrum


def main():

    # # No boundary proposal part

    # # --------------------------------------------------
    # # 1. Define the background
    # # --------------------------------------------------

    # H = 1.0

    # model = NoBoundaryModel(
    #     H=H
    # )

    # # --------------------------------------------------
    # # 2. Numerical parameters
    # # --------------------------------------------------

    # delta = 1e-4
    # t_final = 10.0 / H

    # n_min = 2
    # n_max = 30

    # # --------------------------------------------------
    # # 3. Create solver
    # # --------------------------------------------------

    # solver = Solveur(
    #     model=model,
    #     N_eval=1000,
    #     rtol=1e-10,
    #     atol=1e-12,
    #     method="DOP853",
    # )

    # # --------------------------------------------------
    # # 4. Solve modes and extract power spectrum
    # # --------------------------------------------------

    # results = []
    # mode_solutions_NB = {}

    # for n in range(n_min, n_max + 1):

    #     solution = solver.solve_W(
    #         n=n,
    #         delta=delta,
    #         t_final=t_final,
    #     )

    #     spectrum_point = PowerSpectrum.extract(solution)

    #     results.append(spectrum_point)

    #     if solution.chi_numerical is not None:
    #         mode_solutions_NB[n] = solution

    # # --------------------------------------------------
    # # 5. Convert results to arrays
    # # --------------------------------------------------

    # n_values = np.array([
    #     result.n
    #     for result in results
    # ])

    # power = np.array([
    #     result.power
    #     for result in results
    # ])

    # dimensionless_power = np.array([
    #     result.dimensionless_power
    #     for result in results
    # ])

    # # --------------------------------------------------
    # # 6. Compare with exact no-boundary result
    # # --------------------------------------------------

    # exact_power = (
    #     H**2
    #     / (2.0 * n_values * (n_values**2 - 1))
    # )

    # exact_dimensionless_power = (
    #     H**2 / (4.0 * np.pi**2)
    # )

    # print(" n       numerical P(n)       exact P(n)")

    # for n, P_num, P_exact in zip(
    #     n_values,
    #     power,
    #     exact_power,
    # ):
    #     print(
    #         f"{n:2d}    "
    #         f"{P_num:.12e}    "
    #         f"{P_exact:.12e}"
    #     )

    # # --------------------------------------------------
    # # 7. Plot dimensionless power spectrum
    # # --------------------------------------------------



    # # Exact no-boundary result
    # Delta_exact = H**2 / (4.0 * np.pi**2)

    # # Relative error
    # relative_error = (
    #     dimensionless_power - Delta_exact
    # ) / Delta_exact


    # fig, (ax1, ax2) = plt.subplots(
    #     2, 1,
    #     figsize=(6.5, 5.5),
    #     sharex=True,
    #     gridspec_kw={
    #         "height_ratios": [3, 1],
    #         "hspace": 0.08
    #     }
    # )

    # # ============================================================
    # # Upper panel: power spectrum
    # # ============================================================

    # ax1.plot(
    #     n_values,
    #     dimensionless_power,
    #     "o",
    #     markersize=5,
    #     label="Numerical"
    # )

    # ax1.axhline(
    #     Delta_exact,
    #     linestyle="--",
    #     linewidth=1.8,
    #     label="Analytic"
    # )

    # ax1.set_ylabel(r"$\Delta_n^\Phi$")

    # # Important: prevent misleading offset notation
    # ax1.ticklabel_format(
    #     axis="y",
    #     style="plain",
    #     useOffset=False
    # )

    # # Show a physically sensible window around the exact value
    # ax1.set_ylim(
    #     0.99 * Delta_exact,
    #     1.01 * Delta_exact
    # )

    # ax1.legend(frameon=False)
    # ax1.grid(alpha=0.2)


    # # ============================================================
    # # Lower panel: relative numerical error
    # # ============================================================

    # ax2.plot(
    #     n_values,
    #     1e6 * relative_error,
    #     "o",
    #     markersize=4
    # )

    # ax2.axhline(
    #     0.0,
    #     linestyle="--",
    #     linewidth=1.2
    # )

    # ax2.set_xlabel(r"$n$")
    # ax2.set_ylabel(
    #     r"$10^6\,(\Delta_n^\Phi/\Delta_{\rm exact}^\Phi-1)$"
    # )

    # ax2.grid(alpha=0.2)


    # plt.tight_layout()

    # # --------------------------------------------------
    # # 8. Compare numerical and exact no-boundary modes
    # #    along s=tau<0 and s=t>0
    # # --------------------------------------------------

    # selected_n_values_NB = [
    #     n for n in solver.mode_function_n_values
    #     if n in mode_solutions_NB
    # ]

    # if selected_n_values_NB:
    #     n_panels = len(selected_n_values_NB)
    #     n_cols = 2
    #     n_rows = int(np.ceil(n_panels / n_cols))

    #     fig_modes_NB, axes_modes_NB = plt.subplots(
    #         n_rows,
    #         n_cols,
    #         figsize=(10.5, 3.4 * n_rows),
    #         sharex=True,
    #         squeeze=False,
    #     )
    #     axes_modes_NB = axes_modes_NB.ravel()

    #     for ax, n in zip(axes_modes_NB, selected_n_values_NB):
    #         solution = mode_solutions_NB[n]
    #         chi_segments = solution.chi_numerical

    #         tau = -np.asarray(solution.segments[0].t.imag, dtype=float)
    #         t = np.asarray(solution.segments[1].t.real, dtype=float)

    #         s_mode = np.concatenate([tau, t[1:]])
    #         chi_num = np.concatenate([
    #             np.asarray(chi_segments[0], dtype=complex),
    #             np.asarray(chi_segments[1][1:], dtype=complex),
    #         ])

    #         chi_exact = np.concatenate([
    #             np.asarray(
    #                 model.Analytic_mode_function_Euclidean(n, tau),
    #                 dtype=complex,
    #             ),
    #             np.asarray(
    #                 model.Analytic_mode_function_Lorentzian(n, t[1:]),
    #                 dtype=complex,
    #             ),
    #         ])

    #         line_re, = ax.plot(
    #             s_mode, chi_num.real, linewidth=1.4,
    #             label=r"$\Re\,\chi_n$ numerical",
    #         )
    #         ax.plot(
    #             s_mode, chi_exact.real, "--", linewidth=1.4,
    #             color=line_re.get_color(),
    #             label=r"$\Re\,\chi_n$ exact",
    #         )

    #         line_im, = ax.plot(
    #             s_mode, chi_num.imag, linewidth=1.4,
    #             label=r"$\Im\,\chi_n$ numerical",
    #         )
    #         ax.plot(
    #             s_mode, chi_exact.imag, "--", linewidth=1.4,
    #             color=line_im.get_color(),
    #             label=r"$\Im\,\chi_n$ exact",
    #         )

    #         ax.axvline(0.0, linestyle=":", linewidth=1.0, alpha=0.7)
    #         ax.set_title(rf"$n={n}$")
    #         ax.set_xlabel(r"$s$ ($s=\tau<0$, $s=t>0$)")
    #         ax.set_ylabel(r"$\chi_n$")
    #         ax.grid(alpha=0.2)

    #     for ax in axes_modes_NB[n_panels:]:
    #         ax.set_visible(False)

    #     axes_modes_NB[0].legend(frameon=False, fontsize=8)
    #     fig_modes_NB.suptitle("No-boundary mode functions")
    #     fig_modes_NB.tight_layout()

    # plt.show()

    #End of no-boundary part

    # ============================================================
    # 1. Wineglass model parameters
    # ============================================================

    H_II = 10
    rho_rad = 3 /16

    model = WineglassModel(
        H_II=H_II,
        rho_rad=rho_rad,
    )

    print("Wineglass parameters")
    print("--------------------")
    print(f"H_II    = {model.H_II:.8f}")
    print(f"H_I     = {model.H_I:.8f}")
    print(f"H       = {model.H:.8f}")
    print(f"A_I     = {model.A_I:.8f}")
    print(f"c_II    = {model.c_II:.8f}")
    print(f"tau_min = {model.tau_min:.8f}")
    print()

    # ============================================================
    # 2. Numerical parameters
    # ============================================================

    delta = 1e-4
    t_final = 10.0

    n_min = 2
    n_max = 15

    # ============================================================
    # 3. Solver
    # ============================================================

    solver = Solveur(
        model=model,
        N_eval=1000,
        rtol=1e-10,
        atol=1e-12,
        method="DOP853",
    )

    # ============================================================
    # 4. Compute wineglass power spectrum
    # ============================================================

    results = []
    W0_values = []
    mode_solutions = {}

    for n in range(n_min, n_max + 1):

        solution = solver.solve_W(
            n=n,
            delta=delta,
            t_final=t_final,
        )

        spectrum_point = PowerSpectrum.extract(solution)

        results.append(spectrum_point)

        if solution.chi_numerical is not None:
            mode_solutions[n] = solution

        # Riccati variable at the start of inflation t = 0.
        # For the wineglass contour this is the end of Euclidean region II.
        W0_values.append(solution.segments[1].W[-1])

        print(
            f"n = {n:2d}   "
            f"Delta = {spectrum_point.dimensionless_power:.10e}"
        )

    # ============================================================
    # 5. Convert results to arrays
    # ============================================================

    n_values = np.array([
        result.n
        for result in results
    ])

    Delta_WG = np.array([
        result.dimensionless_power
        for result in results
    ])

    # Analytic large-n/WKB prediction.
    # Analytic_power_spectrum returns P(n; t_final), so convert it
    # to the dimensionless spectrum.
    P_WG_analytic = np.array([
        model.Analytic_power_spectrum(
            n=int(n),
            t_final=t_final,
        )
        for n in n_values
    ])

    Delta_WG_analytic = (
        n_values
        * (n_values**2 - 1.0)
        / (2.0 * np.pi**2)
        * P_WG_analytic
    )

    # ============================================================
    # 6. Euclidean vacuum spectrum
    #
    # Same late-time Hubble constant H as the wineglass universe.
    #
    # Delta_BD = H^2 / (4 pi^2)
    # ============================================================

    Delta_BD = model.H**2 / (4.0 * np.pi**2)

    # ============================================================
    # 7. Plot
    # ============================================================

    fig, ax1 = plt.subplots(
        1,
        1,
        figsize=(7.0, 6.0),
        sharex=True,
        # gridspec_kw={
        #     "height_ratios": [3, 1],
        #     "hspace": 0.08,
        # },
    )

    # ------------------------------------------------------------
    # Upper panel: dimensionless power spectrum
    # ------------------------------------------------------------

    ax1.plot(
        n_values,
        Delta_WG,
        "o",
        markersize=4,
        label="Numerical",
    )

    # ax1.plot(
    #     n_values,
    #     Delta_WG_analytic,
    #     linewidth=1.5,
    #     label="Analytic (B.29)",
    # )

    ax1.axhline(
        Delta_BD,
        linestyle="--",
        linewidth=1.8,
        label=r"$H^2/(4\pi^2)$",
    )

    ax1.set_ylabel(
        r"$\Delta^\Phi_n$"
    )

    ax1.ticklabel_format(
        axis="y",
        style="plain",
        useOffset=False,
    )

    ax1.legend(
        frameon=False
    )

    # Wineglass background parameters used for this spectrum.
    parameter_text = (
        rf"$H_{{\rm I}}={model.H_I:.4g}$" "\n"
        rf"$H_{{\rm II}}={model.H_II:.4g}$" "\n"
        rf"$H={model.H:.4g}$" "\n"
        rf"$\tilde{{\rho}}_{{\rm rad}}^L={model.rho_rad:.4g}$"
    )

    ax1.text(
        0.7,
        0.5,
        parameter_text,
        transform=ax1.transAxes,
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "0.8",
            "alpha": 0.85,
        },
    )

    ax1.grid(
        alpha=0.2
    )

    plt.tight_layout()

    # ============================================================
    # 8. Numerical mode functions vs the analytic estimates of
    #    Appendix B along the complete complex contour
    #
    # We use one real contour coordinate
    #
    #       s = tau   on the Euclidean branch (tau < 0),
    #       s = t     on the Lorentzian branch (t > 0).
    #
    # The numerical chi_n is reconstructed from W_n on all contour
    # segments using the contour version of eq. (B.30).
    #
    # For the analytic estimate we display the four regions of (B.31):
    #
    #   tau < -1/H_I      : asymptotic EAdS solution,
    #   -1/H_I <= tau < 0 : Euclidean WKB,
    #   0 <= t <= 1/H     : Lorentzian WKB,
    #   t > 1/H            : asymptotic dS solution.
    #
    # If -1/H_I lies to the right of tau_min, WineglassModel moves the
    # Euclidean switch back to tau_min so that the EAdS approximation is
    # never used in Euclidean region II.
    # ============================================================

    selected_n_values = [
        n
        for n in solver.mode_function_n_values
        if n in mode_solutions
    ]

    if selected_n_values:
        reference_solution = mode_solutions[selected_n_values[0]]
        segment_E_I = reference_solution.segments[0]
        segment_E_II = reference_solution.segments[1]
        segment_L = reference_solution.segments[2]

        tau_I = -np.asarray(segment_E_I.t.imag, dtype=float)
        tau_II = -np.asarray(segment_E_II.t.imag, dtype=float)
        t_L = np.asarray(segment_L.t.real, dtype=float)

        # Common contour coordinate used on the horizontal axis.  Junction
        # points are included only once.
        s_mode = np.concatenate([
            tau_I,
            tau_II[1:],
            t_L[1:],
        ])

        tau_switch = model.tau_WKB_start
        t_switch = model.t_WKB_end

        n_panels = len(selected_n_values)
        n_cols = 2
        n_rows = int(np.ceil(n_panels / n_cols))

        fig_modes, axes_modes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(10.5, 3.4 * n_rows),
            sharex=True,
            squeeze=False,
        )
        axes_modes = axes_modes.ravel()

        # mode_data_blocks = []

        for ax, n in zip(axes_modes, selected_n_values):

            ax.set_xlim(-5.0, 5.0)

            solution = mode_solutions[n]
            chi_segments = solution.chi_numerical

            if chi_segments is None:
                raise RuntimeError(
                    f"chi_n was not retained for n={n}."
                )

            chi_numerical = np.concatenate([
                np.asarray(chi_segments[0], dtype=complex),
                np.asarray(chi_segments[1][1:], dtype=complex),
                np.asarray(chi_segments[2][1:], dtype=complex),
            ])

            # Piecewise analytic estimates.  On the Euclidean side the
            # method switches from EAdS to WKB at tau=-1/H_I (or tau_min
            # when required by the geometry).  On the Lorentzian side it
            # uses the WKB expression of (B.31), without an additional N_n,
            # until t=1/H and the asymptotic dS expression afterwards.
            chi_analytic_I = np.asarray(
                model.Analytic_mode_function_Euclidean(
                    n=n,
                    tau=tau_I,
                ),
                dtype=complex,
            )
            chi_analytic_II = np.asarray(
                model.Analytic_mode_function_Euclidean(
                    n=n,
                    tau=tau_II,
                ),
                dtype=complex,
            )
            chi_analytic_L = np.asarray(
                model.Analytic_mode_function_Lorentzian(
                    n=n,
                    t=t_L,
                ),
                dtype=complex,
            )

            chi_analytic = np.concatenate([
                chi_analytic_I,
                chi_analytic_II[1:],
                chi_analytic_L[1:],
            ])

            data = np.column_stack((
                s_mode,
                chi_numerical.real,
                chi_numerical.imag,
                chi_analytic.real,
                chi_analytic.imag,
            ))

            # mode_data_blocks.append((n, data))

            line_re, = ax.plot(
                s_mode,
                chi_numerical.real,
                linewidth=1.4,
                label=r"$\Re\,\chi_n$ numerical",
            )
            ax.plot(
                s_mode,
                chi_analytic.real,
                linestyle="--",
                linewidth=1.4,
                color=line_re.get_color(),
                label=r"$\Re\,\chi_n$ analytic",
            )

            line_im, = ax.plot(
                s_mode,
                chi_numerical.imag,
                linewidth=1.4,
                label=r"$\Im\,\chi_n$ numerical",
            )
            ax.plot(
                s_mode,
                chi_analytic.imag,
                linestyle="--",
                linewidth=1.4,
                color=line_im.get_color(),
                label=r"$\Im\,\chi_n$ analytic",
            )

            # Boundaries between the four analytic approximations.
            ax.axvline(
                tau_switch,
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
            )
            ax.axvline(
                0.0,
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
            )
            ax.axvline(
                t_switch,
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
            )

            ax.set_title(rf"$n={n}$")
            ax.set_ylabel(r"$\chi_n$")
            ax.grid(alpha=0.2)

        # with open("mode_functions.dat", "w") as f:

        #     # General model parameters
        #     f.write("# Wineglass model\n")
        #     f.write(f"# H_I = {model.H_I:.16e}\n")
        #     f.write(f"# H_II = {model.H_II:.16e}\n")
        #     f.write(f"# H = {model.H:.16e}\n")
        #     f.write(f"# rho_rad = {model.rho_rad:.16e}\n")
        #     f.write(f"# tau_WKB_start = {model.tau_WKB_start:.16e}\n")
        #     f.write(f"# t_WKB_end = {model.t_WKB_end:.16e}\n")
        #     f.write("\n")

        #     for n, data in mode_data_blocks:

        #         f.write(f"# n = {n}\n")

        #         np.savetxt(
        #             f,
        #             data,
        #             delimiter=",",
        #             header="s,Re_chi_num,Im_chi_num,Re_chi_analytic,Im_chi_analytic",
        #             comments="",
        #             fmt="%.16e",
        #         )

        #         f.write("\n")

        for ax in axes_modes[n_panels:]:
            ax.set_visible(False)

        for ax in axes_modes[:n_panels]:
            ax.set_xlabel(r"$s$ ($s=\tau<0$, $s=t>0$)")

        axes_modes[0].legend(
            frameon=False,
            fontsize=8,
        )

        fig_modes.suptitle(
            r"Wineglass modes: EAdS $\to$ WKB$_E$ $\to$ WKB$_L$ $\to$ dS"
        )
        fig_modes.tight_layout()

    # ============================================================
    # 9. Discrepancy with Euclidean at the start of inflation
    #
    # To display on a single plot the dependence on the radiation density rho_rad, add more values of rho_rad to the list below.
    #
    # H_II^2 = 2 H_I^2 /
    #          [16 rho_rad H_I^2 - (sqrt(5) - 2)].
    # ============================================================

    H_I_fixed = model.H_I

    rho_rad_values = [
        3.0 / 16.0,
        0.2,
        1,
    ]

    fig_discrepancy, ax_discrepancy = plt.subplots(
        figsize=(7.0, 4.5)
    )

    # with open("discrepancy.csv", "w") as f:

    for rho in rho_rad_values:

            if np.isclose(rho, rho_rad):
                # # Parameters of the current model
                # f.write(f"# rho_rad = {rho:.16e}\n")
                # f.write(f"# H_I = {H_I_fixed:.16e}\n")
                # f.write(f"# H_II = {model.H_II:.16e}\n")
                # f.write(f"# H = {model.H:.16e}\n")
                
                # Reuse the values already computed for the baseline model.
                W0 = np.array(W0_values, dtype=complex)

            else:
                denominator = (
                    16.0 * rho * H_I_fixed**2
                    - (np.sqrt(5.0) - 2.0)
                )

                if denominator <= 0.0:
                    raise ValueError(
                        "Chosen rho_rad is incompatible with the fixed H_I."
                    )

                H_II_rho = np.sqrt(
                    2.0 * H_I_fixed**2 / denominator
                )

                model_rho = WineglassModel(
                    H_II=H_II_rho,
                    rho_rad=rho,
                )

                # # Parameters of the current model
                # f.write(f"# rho_rad = {rho:.16e}\n")
                # f.write(f"# H_I = {H_I_fixed:.16e}\n")
                # f.write(f"# H_II = {model_rho.H_II:.16e}\n")
                # f.write(f"# H = {model_rho.H:.16e}\n")


                solver_rho = Solveur(
                    model=model_rho,
                    N_eval=1000,
                    rtol=1e-10,
                    atol=1e-12,
                    method="DOP853",
                    mode_function_n_values=[],
                )

                W0 = []

                for n in range(n_min, n_max + 1):
                    solution_rho = solver_rho.solve_W(
                        n=n,
                        delta=delta,
                        # Only W_n(0) is needed here.
                        t_final=1e-8,
                    )

                    W0.append(
                        solution_rho.segments[1].W[-1]
                    )

                W0 = np.array(W0, dtype=complex)

            # W_n^BD(0) = (n^2 - 1)/(n H^2), with
            # H^2 = 3/(16 rho_rad).
            W_BD_0 = (
                4
                * rho
                * (n_values**2 - 1.0)
                / (n_values)
            )

            discrepancy = (W0 - W_BD_0) / W0

            # data = np.column_stack((
            #                 n_values,
            #                 discrepancy.real,
            #                 discrepancy.imag
            # ))

            # # Numerical data
            # np.savetxt(
            #     f,
            #     data,
            #     delimiter=",",
            #     header="n,discrepancy_real,discrepancy_imag",
            #     comments="",
            #     fmt=["%d", "%.16e", "%.16e"]
            # )

            # # Separate successive blocks
            # f.write("\n")
            
            ax_discrepancy.plot(
                n_values,
                discrepancy.real,
                "o-",
                markersize=3,
                linewidth=1.1,
                label=rf"$\tilde{{\rho}}_{{\rm rad}}^L={rho:.4g}$",
            )

    parameter_text = (
            rf"$H_{{\rm II}}={model.H_II:.4g}$"
        )
    
    ax_discrepancy.text(
        0.7,
        0.5,
        parameter_text,
        transform=ax_discrepancy.transAxes,
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "0.8",
            "alpha": 0.85,
        },
    )

    ax_discrepancy.set_xlabel(r"$n$")
    ax_discrepancy.set_ylabel(
        r"$\left.\Delta\mathcal{W}_n/\mathcal{W}_n\right|_{t=0}$"
    )
    ax_discrepancy.legend(frameon=False)
    ax_discrepancy.grid(alpha=0.2)

    fig_discrepancy.tight_layout()

    plt.show()


    # ============================================================
        # 10. Optional: save the power spectrum data points in csv files for later analysis 
    # ============================================================

    # data = np.column_stack((
    #     n_values,
    #     Delta_WG,
    #     np.full_like(n_values, Delta_BD, dtype=float),
    # ))
    
    # header = (
    #     f"Model: Wineglass\n"
    #     f"H_I = {model.H_I:.16e}\n"
    #     f"H_II = {model.H_II:.16e}\n"
    #     f"H = {model.H:.16e}\n"
    #     f"rho_rad = {model.rho_rad:.16e}\n"
    #     f"delta = {delta:.16e}\n"
    #     f"t_final = {t_final:.16e}\n"
    #     f"rtol = {solver.rtol:.1e}\n"
    #     f"atol = {solver.atol:.1e}\n"
    #     f"n,Delta_WG,Delta_BD"
    # )

    # np.savetxt(
    #     "wineglass_spectrum.csv",
    #     data,
    #     delimiter=",",
    #     header=header,
    #     comments="# ",
    #     fmt=["%d", "%.16e", "%.16e"]
    # )
    


if __name__ == "__main__":
    main()