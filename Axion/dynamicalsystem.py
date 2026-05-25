import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------- PARAMETERS ----------
alpha = 0.2
d = 3.0
v0 = 1.5
ufinal = 10.0
N = 1000
nI = 20
nJ = 20
# Window for initial conditions in the projected (I, J) plane
IC_I_WINDOW = (0, 20)
IC_J_WINDOW = (-10, 10)
# Window used to display the final plot
PLOT_I_WINDOW = (-5, 25)
PLOT_J_WINDOW = (-25, 25)


# ---------- ODE SYSTEM ----------
def Idot(I, J, H):
    return J / alpha


def Jdot(I, J, H):
    return -d * J * H + np.sin(I) / alpha


def Hdot(I, J, H):
    return -H**2 + J**2 / d - 2.0 * (v0 + np.cos(I)) / (d * (d - 1.0))


def system(u, X):
    I, J, H = X
    return np.array([
        Idot(I, J, H),
        Jdot(I, J, H),
        Hdot(I, J, H),
    ])


# ---------- NUMERICAL SOLVER ----------
def solve_system(X0, u_span=(0.0, ufinal), n_points=N):
    u_eval = np.linspace(u_span[0], u_span[1], n_points)

    sol = solve_ivp(
        system,
        u_span,
        X0,
        t_eval=u_eval,
        method="RK45",
        rtol=1e-9,
        atol=1e-11,
    )

    return sol.t, sol.y, sol


# ---------- INITIAL CONDITIONS ----------
def make_initial_conditions(
    H0=0.0,
    nI=nI,
    nJ=nJ,
    I_window=IC_I_WINDOW,
    J_window=IC_J_WINDOW,
):
    I_vals = np.linspace(I_window[0], I_window[1], nI)
    J_vals = np.linspace(J_window[0], J_window[1], nJ)

    initials = []

    for I0 in I_vals:
        for J0 in J_vals:
            if abs(I0 - np.pi) < 1e-12 and abs(J0) < 1e-12:
                continue

            initials.append(np.array([I0, J0, H0], dtype=float))

    return initials


# ---------- PLOTS ----------
def add_arrow_on_orbit(ax, I, J, Istar=np.pi, Jstar=0.0, fraction=0.35):
    """Add one arrow on a numerical orbit, away from the fixed point."""
    if len(I) < 10:
        return

    # Distance to the fixed point in the projected plane
    dist = np.sqrt((I - Istar)**2 + (J - Jstar)**2)

    # We choose a point where the distance is still a fixed fraction
    # of the initial distance, instead of using the midpoint in time.
    target_dist = fraction * dist[0]

    candidates = np.where(dist > target_dist)[0]

    if len(candidates) < 2:
        return

    k = candidates[-1]

    if k >= len(I) - 1:
        k = len(I) - 2

    ax.annotate(
        "",
        xy=(I[k + 1], J[k + 1]),
        xytext=(I[k], J[k]),
        arrowprops=dict(arrowstyle="->", lw=1.0),
    )


def plot_projected_phase_portrait(
    H0=0.0,
    nI=nI,
    nJ=nJ,
    u_span=(0.0, ufinal),
    filename=None,
):
    fig, ax = plt.subplots(figsize=(8, 6))

    initial_conditions = make_initial_conditions(
        H0=H0,
        nI=nI,
        nJ=nJ,
        I_window=IC_I_WINDOW,
        J_window=IC_J_WINDOW,
    )

    for X0 in initial_conditions:
        u, sol, full_sol = solve_system(X0, u_span=u_span)

        if not full_sol.success:
            print(full_sol.message)
            continue

        I = sol[0]
        J = sol[1]

        ax.plot(I, J, lw=1.2)
        # add_arrow_on_orbit(ax, I, J)

    Hstar = np.sqrt(2.0 * (1.0 - v0) / (d * (d - 1.0)))
    ax.plot(np.pi, 0.0, marker="o", markersize=7)

    ax.set_xlabel(r"$I$")
    ax.set_ylabel(r"$\mathcal{J}$")
    ax.set_title(r"Projected phase portrait in $(I,\mathcal{J})$")
    ax.grid(True)

    # This is only the display window
    ax.set_xlim(*PLOT_I_WINDOW)
    ax.set_ylim(*PLOT_J_WINDOW)

    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=200)

    return fig, ax


def plot_time_series(X0=np.array([0, 5, 0.0]), filename=None):
    """Plot I(u), J(u), and H(u) for one initial condition."""
    u, sol, full_sol = solve_system(X0)

    if not full_sol.success:
        print(full_sol.message)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(u, sol[0], label=r"$I(u)$")
    ax.plot(u, sol[1], label=r"$\mathcal{J}(u)$")
    ax.plot(u, sol[2], label=r"$\mathcal{H}(u)$")

    ax.set_xlabel(r"$u$")
    ax.set_ylabel("value")
    ax.set_title("Dimensionless dynamical system")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=200)

    return fig, ax



# ---------- MAIN ----------
if __name__ == "__main__":
    plot_time_series(filename="time_series.png")
    # plot_projected_phase_portrait(H0=0.0, filename="phase_portrait_IJ.png")
    plt.show()