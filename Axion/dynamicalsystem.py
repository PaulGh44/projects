import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------- GLOBAL PARAMS ----------
alpha = 1.2
d = 3
v0 = 0.5
ufinal = 10
N = 1000


def Idot(I: float, J: float, H: float) -> float:
    return J / alpha


def Jdot(I: float, J: float, H: float) -> float:
    return -d * J * H + np.sin(I) / alpha


def Hdot(I: float, J: float, H: float) -> float:
    return -H**2 + J**2 / d - 2 * (v0 + np.cos(I)) / (d * (d - 1))


def system(u: float, X: np.ndarray) -> np.ndarray:
    I, J, H = X
    return np.array([
        Idot(I, J, H),
        Jdot(I, J, H),
        Hdot(I, J, H),
    ])


def solve_system(X0: np.ndarray, u_span: tuple = (0, ufinal), N: int = N):
    u_eval = np.linspace(u_span[0], u_span[1], N)
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


# ---------- Main ----------
if __name__ == "__main__":
    I0 = np.pi / 2
    J0 = 0.0
    H0 = 0.0
    X0 = np.array([I0, J0, H0])

    u, sol, full_sol = solve_system(X0)

    if not full_sol.success:
        print(full_sol.message)

    plt.figure(figsize=(12, 8))
    plt.plot(u, sol[0], label=r"$I(u)$")
    plt.plot(u, sol[1], label=r"$\mathcal{J}(u)$")
    plt.plot(u, sol[2], label=r"$\mathcal{H}(u)$")
    plt.xlabel(r"$u$")
    plt.ylabel("Values")
    plt.title("Dimensionless dynamical system")
    plt.legend()
    plt.grid()
    plt.show()