import numpy as np
from scipy.integrate import solve_ivp, simpson
from typing import Tuple, Sequence, Dict
import matplotlib.pyplot as plt
from scipy.special import hyp2f1

# ---------- STYLE ----------
plt.style.use("seaborn-v0_8")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "figure.figsize": (8, 5)
})

# ---------- GLOBAL PARAMS ----------
delta = 1e-2
ti, tf = 0.0, 10.0
N_euclid = 2000
N_lorentz = 3000
DEBUG = False  # <-- set True to enable all debug plots

# ---------- Euclidean region ----------
def f_of_z(z: float, k: int) -> float:
    return (k*(k/2+1))/(2*(1+z)**2) + k/(z**2 - 1) - k*(k+2)/(z**2 - 1)**2

def g_of_z(z: float, k: int) -> float:
    return k/(1+z) + 4*z/(z**2 - 1)

def sys_euclid(z: float, Y: Tuple[float, float], k: int) -> np.ndarray:
    chi, dchi = Y
    return np.array([dchi, -f_of_z(z, k)*chi - g_of_z(z, k)*dchi], dtype=float)

def a_1(k: int) -> float:
    return k*(k+4)/(4*(k+2))

# avoid the pole at z = -1
z0 = -1.0 + delta
a, b = z0, 0.0

def set_ini_euclid(k: int) -> np.ndarray:
    chi0  = 1.0 + a_1(k)*delta
    dchi0 = a_1(k)
    return np.array([chi0, dchi0], dtype=float)

def solve_euclid(k: int, N: int = N_euclid):
    t_eval = np.linspace(a, b, N)
    sol = solve_ivp(lambda z, Y: sys_euclid(z, Y, k),
                    (a, b), set_ini_euclid(k),
                    method="Radau", t_eval=t_eval,
                    rtol=1e-12, atol=1e-15)
    if not sol.success:
        print("Integrator failed (Euclidean):", sol.message)
    return sol

def Phifinal(z, chi, k: int):
    z   = np.asarray(z)
    chi = np.asarray(chi, dtype=complex)
    pref = (k + 2) / (2 ** ((k + 2) / 2))
    return chi * np.power(1.0 + z, k / 2) * pref

def phiE_and_dphiE_at0(sol, k: int):
    """Return φ_E(0) and dφ_E/dz(0) from (chi, chi')."""
    z_grid = sol.t
    chi    = sol.y[0]
    dchi   = sol.y[1]
    z0_chk = z_grid[-1]
    assert abs(z0_chk - 0.0) < 1e-14, "Last Euclidean grid point is not z=0."
    pref = (k + 2) / (2 ** ((k + 2) / 2))
    phi0_E = pref * chi[-1] * (1.0 + z0_chk) ** (k / 2)
    dphi0_E_z = pref * ( dchi[-1] * (1.0 + z0_chk) ** (k / 2)
                       + chi[-1]  * (k/2) * (1.0 + z0_chk) ** (k/2 - 1) )
    return complex(phi0_E), complex(dphi0_E_z)

# ---------- Lorentzian region ----------
def a_of_t(t: float) -> float:
    return np.cosh(t)

def a_dot_of_t(t: float) -> float:
    return np.sinh(t)

def f_of_t(t: float, k: int) -> float:
    a = a_of_t(t)
    return k*(k+2)/(a*a)

def g_of_t(t: float, k: int) -> float:
    return 3.0 * a_dot_of_t(t) / a_of_t(t)

# Real system: y = [Re φ, Im φ, Re φ', Im φ']
def sys_lorentz(t: float, y: np.ndarray, k: int) -> np.ndarray:
    Re_phi, Im_phi, Re_dphi, Im_dphi = y
    f = f_of_t(t, k); g = g_of_t(t, k)
    dRe_phi  = Re_dphi
    dIm_phi  = Im_dphi
    dRe_dphi = - g*Re_dphi - f*Re_phi
    dIm_dphi = - g*Im_dphi - f*Im_phi
    return np.array([dRe_phi, dIm_phi, dRe_dphi, dIm_dphi], dtype=float)

def set_ini_lorentz_from_euclid(solE, k: int) -> np.ndarray:
    phi0_E, dphi0_E_z = phiE_and_dphiE_at0(solE, k)
    # Wick rotation: d/dt = i d/dz
    phi0  = phi0_E
    dphi0 = 1j * dphi0_E_z
    return np.array([phi0.real, phi0.imag, dphi0.real, dphi0.imag], dtype=float)

def solve_lorentz_from_euclid(k: int, solE, N: int = N_lorentz):
    t_eval = np.linspace(ti, tf, N)
    y0 = set_ini_lorentz_from_euclid(solE, k)
    sol = solve_ivp(lambda t, y: sys_lorentz(t, y, k),
                    (ti, tf), y0,
                    method="Radau", t_eval=t_eval,
                    rtol=1e-12, atol=1e-15)
    if not sol.success:
        print("Integrator failed (Lorentzian):", sol.message)
    return sol

# ---------- Hypergeometric Φ₊(t) ----------
def phi_plus(t, k: int):
    """
    Φ₊(t) = exp(-3 t) * 2F1(a,b;c; -exp(-2 t)),
    with a = 3 + k, b = 3/2 + k, c = 5/2.
    """
    t = np.asarray(t, dtype=float)
    a = 3.0 + k
    b = 1.5 + k
    c = 2.5
    z = -np.exp(-2.0*t)
    return np.exp(-3.0*t) * hyp2f1(a, b, c, z)

# --- |Ψ|²(Φ1) from Σ(ℓ) ---
def psi2_of_phi1(phi1: np.ndarray, sigma_l: float) -> np.ndarray:
    """
    |Ψ^{(2)}_{(ℓ)}(Φ1)|^2 = exp( - Φ1^2 / (2 Σ(ℓ)) ).
    """
    phi1 = np.asarray(phi1, dtype=float)
    return np.exp(-0.5 * (phi1**2) / sigma_l)


# ---------- Normalize Lorentzian solution at tf ----------
def normalize_lorentz_at_tf(phiL: np.ndarray, t_arr: np.ndarray) -> tuple[np.ndarray, complex, complex]:
    """Return normalized series, complex factor c, and raw φ_L(tf)."""
    phi_tf = phiL[-1]  # last point corresponds to tf
    mag = np.abs(phi_tf)
    if mag < 1e-14:
        raise ZeroDivisionError("|phi_L(tf)| ≈ 0 : cannot normalize.")
    c = np.exp(-1j * np.angle(phi_tf)) / mag
    return c * phiL, c, phi_tf

# ---------- Core: compute Γ₊ for a given k ----------
def compute_gamma_plus(k: int, t_window: tuple[float,float]=(6.0,8.0)) -> dict:
    # Euclidean → ICs
    solE = solve_euclid(k)
    # Lorentzian
    solL = solve_lorentz_from_euclid(k, solE)
    t_arr = solL.t
    phiL  = solL.y[0] + 1j*solL.y[1]

    # Normalize Lorentzian only at tf
    phiL_norm, c_norm, phi_tf_raw = normalize_lorentz_at_tf(phiL, t_arr)

    # Window in Lorentzian region
    t1, t2 = t_window
    mask = (t_arr >= t1) & (t_arr <= t2)
    t_win = t_arr[mask]

    # Targets and basis
    y = np.imag(phiL_norm[mask])
    x = phi_plus(t_win, k)  # real for t>=0

    # Least-squares scalar Γ_+ = <y,x>/<x,x>
    num_int = simpson(y * x, x=t_win)
    den_int = simpson(x * x, x=t_win)
    Gamma_plus = num_int / den_int if den_int > 0 else np.nan

    # Final observable Sigma(ℓ) = -4 / (3 Γ_+)
    sigma_l = -4.0 / (3.0 * Gamma_plus) if np.isfinite(Gamma_plus) and abs(Gamma_plus) > 0 else np.nan

    # Optional info
    norm_y = np.sqrt(simpson(np.abs(y)**2, x=t_win))
    norm_x = np.sqrt(simpson(np.abs(x)**2, x=t_win))
    ratio  = norm_y / norm_x if norm_x > 0 else np.nan

    if DEBUG:
        # Euclidean solution
        z_arr, chi_arr = solE.t, solE.y[0]
        phiE = Phifinal(z_arr, chi_arr, k)
        plt.figure()
        plt.plot(z_arr, np.real(phiE), label='Re φ_E', linewidth=2)
        plt.plot(z_arr, np.imag(phiE), '--', label='Im φ_E', linewidth=2)
        plt.title(f"Euclidean solution φ_E(z) | k={k}")
        plt.xlabel('z'); plt.ylabel('φ_E(z)')
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()

        # Lorentzian solution (raw and normalized)
        plt.figure()
        plt.plot(t_arr, np.real(phiL), label='Re φ_L', alpha=0.6)
        plt.plot(t_arr, np.imag(phiL), '--', label='Im φ_L', alpha=0.6)
        plt.plot(t_arr, np.real(phiL_norm), label='Re φ_L (norm)', linewidth=2)
        plt.plot(t_arr, np.imag(phiL_norm), '--', label='Im φ_L (norm)', linewidth=2)
        plt.title(f"Lorentzian solution φ_L(t) (raw & normalized) | k={k}")
        plt.xlabel('t'); plt.ylabel('φ_L(t)')
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()

        # Hypergeometric Φ₊ on the same window
        t_plot = np.linspace(t1, t2, 400)
        phi_plus_vals = phi_plus(t_plot, k)
        plt.figure()
        plt.plot(t_plot, np.real(phi_plus_vals), label='Re Φ₊(t)', linewidth=2)
        plt.plot(t_plot, np.imag(phi_plus_vals), '--', label='Im Φ₊(t)', linewidth=2)
        plt.title(f"Hypergeometric Φ₊(t) with exp(-3t) on [{t1},{t2}] | k={k}")
        plt.xlabel('t'); plt.ylabel('Φ₊(t)')
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()

        # Comparison on the window
        plt.figure()
        plt.plot(t_win, y, 'o', label='Im φ_L_norm(t)', alpha=0.8)
        plt.plot(t_win, Gamma_plus * x, '-', label='Γ₊·Φ₊(t)', linewidth=2)
        plt.title(f"Comparison Im φ_L_norm vs Γ₊·Φ₊ on [{t1},{t2}] | k={k}\n"
                  f"Γ₊={Gamma_plus:.6e}, ||y||/||x||={ratio:.3e}")
        plt.xlabel('t'); plt.ylabel('value')
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()

        plt.figure()
        plt.plot(t_win, y, 'o', label='Im φ_L_norm(t)', alpha=0.8)
        plt.plot(t_win, Gamma_plus * x, '-', label='Γ₊·Φ₊(t)', linewidth=2)
        plt.title(f"Comparison Im φ_L_norm vs Γ₊·Φ₊ on [{t1},{t2}] | k={k}\n"
                  f"Γ₊={Gamma_plus:.6e},  Σ(ℓ)={sigma_l:.6e},  ||y||/||x||={ratio:.3e}")
        plt.xlabel('t'); plt.ylabel('value')
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
        plt.show()

    return {
        "k": k,
        "Gamma_plus": float(Gamma_plus),
        "Sigma_l": float(sigma_l),
        "norm_ratio": float(ratio),
        "phi_tf_raw": complex(phi_tf_raw),
        "c_norm": complex(c_norm),
        "window": (float(t1), float(t2)),
    }

# ---------- Example usage ----------
if __name__ == "__main__":
    t_window = (6.0, 8.0)

    if DEBUG:
        # --- Debug mode: single k=5 ---
        k_test = 5
        res = compute_gamma_plus(k_test, t_window=t_window)
        print(f"[DEBUG] k={k_test}: Gamma_+ = {res['Gamma_plus']:.6e}, "
              f"Sigma(k) = {res['Sigma_l']:.6e}, phi_L(tf)={res['phi_tf_raw']}")

        # Always plot |Ψ|^2 for k=5
        phi1_grid = np.linspace(-1.0, 1.0, 600)
        psi2_vals = psi2_of_phi1(phi1_grid, res["Sigma_l"])
        plt.figure(figsize=(8,5))
        plt.plot(phi1_grid, psi2_vals, linewidth=2, label=r"|Ψ|² from Σ(ℓ), k=5")
        plt.title(r"Wavefunction modulus squared $|\Psi|^2(\Phi_1)$ from Σ(ℓ), k=5")
        plt.xlabel(r"$\Phi_1$")
        plt.ylabel(r"$|\Psi|^2(\Phi_1)$")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

    else:
        # --- Production mode: k = 1..20 ---
        k_values = list(range(1, 21))
        sigma_num = []
        sigma_th  = []

        for k in k_values:
            res = compute_gamma_plus(k, t_window=t_window)
            sigma_num.append(res["Sigma_l"])
            sigma_th.append(1.0 / (2.0 * k * (k + 1.0) * (k + 2.0)))
            print(f"[k={k}]  Gamma_+ = {res['Gamma_plus']:.6e}   Sigma(k) = {res['Sigma_l']:.6e}")

        k_arr     = np.array(k_values, dtype=float)
        sigma_num = np.array(sigma_num, dtype=float)
        sigma_th  = np.array(sigma_th, dtype=float)

        # Plot Σ(k): numerical vs theoretical
        plt.figure(figsize=(8,5))
        plt.plot(k_arr, sigma_num, 'o-', label='Numerical Σ(k)', linewidth=2)
        plt.plot(k_arr, sigma_th,  '--', label='Theoretical Σ_th(k)=1/[2k(k+1)(k+2)]', linewidth=2)
        plt.title("Σ(k): numerical vs theoretical")
        plt.xlabel('k'); plt.ylabel('Σ(k)')
        plt.xticks(k_values)
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Always plot |Ψ|^2 for k=5
        try:
            idx_k5 = k_values.index(5)
            sigma_k5 = sigma_num[idx_k5]
            phi1_grid = np.linspace(-1.0, 1.0, 600)
            psi2_vals = psi2_of_phi1(phi1_grid, sigma_k5)
            plt.figure(figsize=(8,5))
            plt.plot(phi1_grid, psi2_vals, linewidth=2, label=r"|Ψ|² from Σ(ℓ), k=5")
            plt.title(r"Wavefunction modulus squared $|\Psi|^2(\Phi_1)$ from Σ(ℓ), k=5")
            plt.xlabel(r"$\Phi_1$")
            plt.ylabel(r"$|\Psi|^2(\Phi_1)$")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            plt.tight_layout()
        except ValueError:
            print("k=5 not found in k_values; skipping |Ψ|^2 plot.")

        plt.show()

