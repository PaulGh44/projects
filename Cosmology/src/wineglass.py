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
epsilon=1.25
tau_min=-np.pi
Delta_plus=3
delta = 1e-2
ti, tf = 0.0, 10.0
N_euclid_I = 2000
N_euclid_II = 2000
N_lorentz = 3000
DEBUG = False  # <-- set True to enable all debug plots


# ---------- Euclidean region I----------
#u=exp(\tau-\tau_{min})
def f_of_u(u: float, k: int) -> float:
    D=1+2*(epsilon-2)*u+u*u
    return (Delta_plus*Delta_plus)/(u*u)-(3*Delta_plus*(1-u*u))/(u*u*D) -4*k*(k+2)/(D*D)

def g_of_u(u: float, k: int) -> float:
    return (2*Delta_plus+1)/u -(3*(1-u*u))/(u*(1+2*(epsilon-2)*u+u*u))

def sys_euclid_I(u: float, Y: Tuple[float, float], k: int) -> np.ndarray:
    chi, dchi = Y
    return np.array([dchi, -f_of_u(u, k)*chi - g_of_u(u, k)*dchi], dtype=float)

def a_1(k: int) -> float:
    return (3*Delta_plus*(2-epsilon))/(Delta_plus-1)

# avoid the pole at u = 0
u0 = delta
a, b = u0, 1

def set_ini_euclid_I(k: int) -> np.ndarray:
    chi0  = (1.0 + a_1(k)*delta)
    dchi0 = a_1(k)
    return np.array([chi0, dchi0], dtype=float)

def solve_euclid_I(k: int, N: int = N_euclid_I):
    t_eval = np.linspace(a, b, N)
    sol = solve_ivp(lambda u, Y: sys_euclid_I(u, Y, k),
                    (a, b), set_ini_euclid_I(k),
                    method="Radau", t_eval=t_eval,
                    rtol=1e-12, atol=1e-15)
    if not sol.success:
        print("Integrator failed (Euclid_I):", sol.message)
    return sol

def Phifinal(u, chi, k: int):
    u   = np.asarray(u)
    chi = np.asarray(chi, dtype=float)
    return chi * np.power(u, Delta_plus)

def phiE_and_dphiE_at1(sol, k: int)->tuple[float,float]:
    """Return φ_E(1) and dφ_E/du(1) from (chi, chi')."""
    u_grid = sol.t
    chi    = sol.y[0]
    dchi   = sol.y[1]
    u0_chk = u_grid[-1]
    assert abs(u0_chk - 1.0) < 1e-14, "Last Euclidean grid point is not u=1."
    phi1_E = chi[-1] * (u0_chk) ** (Delta_plus)
    dphi1_E_u = (dchi[-1] * (u0_chk) ** (Delta_plus)
                       + chi[-1]  * (Delta_plus) * (u0_chk) ** (Delta_plus - 1) )
    return phi1_E, dphi1_E_u

# ---------- Euclidean region II----------
#d^2phi +3*adot*dphi/a-k*(k+2)*phi/a^2
def a_of_tau(tau: float) -> float:
    return epsilon+np.cos(tau)

def a_dot_of_tau(tau: float) -> float:
    return -np.sin(tau)

def f_of_tau(tau: float, k: int) -> float:
    a = a_of_tau(tau)
    return -k*(k+2)/(a*a)

def g_of_tau(tau: float, k: int) -> float:
    return 3.0 * a_dot_of_tau(tau) / a_of_tau(tau)

# Real system: The solution remains real up to the analytic continuation point t=\tau=0
def sys_euclid_II(tau: float, y: np.ndarray, k: int) -> np.ndarray:
    phi,dphi = y
    f = f_of_tau(tau, k); g = g_of_tau(tau, k)
    return np.array([dphi, -f_of_tau(tau, k)*phi - g_of_tau(tau, k)*dphi], dtype=float)

def set_ini_euclid_II_from_euclid_I(solE_I, k: int) -> np.ndarray:
    phi1_E, dphi1_E_u = phiE_and_dphiE_at1(solE_I, k)
    return np.array([phi1_E, dphi1_E_u], dtype=float)

def solve_euclid_II_from_euclid_I(k: int, solE_I, N: int = N_euclid_II):
    t_eval = np.linspace(tau_min, 0, N)
    y0 = set_ini_euclid_II_from_euclid_I(solE_I, k)
    sol = solve_ivp(lambda tau, y: sys_euclid_II(tau, y, k),
                    (tau_min, 0), y0,
                    method="Radau", t_eval=t_eval,
                    rtol=1e-12, atol=1e-15)
    if not sol.success:
        print("Integrator failed (Euclid_II):", sol.message)
    return sol

def phiE_and_dphiE_at0(sol, k: int)->tuple[float,float]:
    """Return φ_E(0) and dφ_E/du(0)"""
    tau_grid = sol.t
    phi    = sol.y[0]
    dphi   = sol.y[1]
    return phi[-1], dphi[-1]

# ---------- Lorentzian region ----------
def a_of_t(t: float) -> float:
    return epsilon+np.cosh(t)

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

def set_ini_lorentz_from_euclid_II(solE_II, k: int) -> np.ndarray:
    phi0_E, dphi0_E_tau = phiE_and_dphiE_at0(solE_II, k)
    # Wick rotation: d/dt = i d/dz
    phi0  = phi0_E
    dphi0 = 1j * dphi0_E_tau
    return np.array([phi0.real, phi0.imag, dphi0.real, dphi0.imag], dtype=float)

def solve_lorentz_from_euclid_II(k: int, solE_II, N: int = N_lorentz):
    t_eval = np.linspace(ti, tf, N)
    y0 = set_ini_lorentz_from_euclid_II(solE_II, k)
    sol = solve_ivp(lambda t, y: sys_lorentz(t, y, k),
                    (ti, tf), y0,
                    method="Radau", t_eval=t_eval,
                    rtol=1e-12, atol=1e-15)
    if not sol.success:
        print("Integrator failed (Lorentzian):", sol.message)
    return sol

# ---------- Heun Φ₊(t) ----------
# ===================== DEBUG: plots + Gamma_+ =====================

# --- base (type "plus") pour l'ajustement sur la fenêtre t∈[t1,t2]
def phi_plus(t, k: int):
    """
    Variante simple (no-boundary-like) pour le mode DEBUG:
    Φ₊(t) ≈ exp(-3 t)+a_1*exp(-4t). Suffisant pour tester Γ₊ et le pipeline de fit.
    """
    a1=-(Delta_plus*epsilon)/(Delta_plus-1)
    t = np.asarray(t, dtype=float)
    return np.exp(-3.0*t)+a1*np.exp(-4.0*t)

def normalize_lorentz_at_tf(phiL: np.ndarray) -> tuple[np.ndarray, complex, complex]:
    """Normalise φ_L(t) à t=tf pour enlever la phase/amplitude globale."""
    phi_tf = phiL[-1]
    mag = np.abs(phi_tf)
    if mag < 1e-14:
        raise ZeroDivisionError("|phi_L(tf)| ≈ 0 : cannot normalize.")
    c = np.exp(-1j * np.angle(phi_tf)) / mag
    return c * phiL, c, phi_tf

# --- helpers pour reconstruire φ sur chaque segment + combiner (sans interpolation)
def phi_from_euclid_I(solE_I):
    u = solE_I.t
    chi = solE_I.y[0]
    phi = Phifinal(u, chi, k=0)           # φ_EI(u) = u^Δ χ
    tau_I = tau_min + np.log(u)           # u = e^{τ - τ_min}
    return tau_I, phi

def phi_from_euclid_II(solE_II):
    tau = solE_II.t
    phi = solE_II.y[0]
    return tau, phi

def phi_from_lorentz(solL):
    t = solL.t
    Re_phi = solL.y[0]; Im_phi = solL.y[1]
    phi = Re_phi + 1j*Im_phi
    return t, phi

def stitch_contour(tau_I, phi_I, tau_II, phi_II, t_L, phi_L):
    lam = np.concatenate([tau_I, tau_II, t_L])
    phi = np.concatenate([phi_I,  phi_II,  phi_L])
    idx = np.argsort(lam)
    return lam[idx], phi[idx]

def compute_and_plot_debug(k: int = 5, t_window: tuple[float,float]=(6.0, 8.0)) -> dict:
    # ---------- intégrations ----------
    # (1) Euclide I
    solE_I = solve_euclid_I(k)
    tau_I, phi_I = phi_from_euclid_I(solE_I)

    # (2) Euclide II (IC depuis u=1)
    solE_II = solve_euclid_II_from_euclid_I(k, solE_I)
    tau_II, phi_II = phi_from_euclid_II(solE_II)

    # (3) Lorentz (IC : φ(0)=φ_E(0), φ'_t(0)=i φ'_τ(0))
    solL = solve_lorentz_from_euclid_II(k, solE_II)
    t_arr, phiL = phi_from_lorentz(solL)

    # ---------- normalisation à t=tf ----------
    phiL_norm, c_norm, phi_tf_raw = normalize_lorentz_at_tf(phiL)

    # ---------- calcul Γ₊ sur fenêtre ----------
    t1, t2 = t_window
    mask = (t_arr >= t1) & (t_arr <= t2)
    t_win = t_arr[mask]
    y = np.imag(phiL_norm[mask])          # cible (Imag de la solution normalisée)
    x = phi_plus(t_win, k)                # base (réelle ici)

    num = simpson(y * x, x=t_win)
    den = simpson(x * x, x=t_win)
    Gamma_plus = num / den if den > 0 else np.nan
    Sigma_l = -4.0 / (3.0 * Gamma_plus) if np.isfinite(Gamma_plus) and abs(Gamma_plus) > 0 else np.nan

    # ---------- plots DEBUG ----------
    if DEBUG:
        # (A) trois régions
        fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=False)
        axs[0].plot(tau_I, phi_I.real, label='Re φ (Euclide I)')
        axs[0].set_title('Euclide I: φ(τ)'); axs[0].set_xlabel('τ'); axs[0].set_ylabel('φ'); axs[0].legend()

        axs[1].plot(tau_II, phi_II.real, label='Re φ (Euclide II)', color='C1')
        axs[1].set_title('Euclide II: φ(τ)'); axs[1].set_xlabel('τ'); axs[1].set_ylabel('φ'); axs[1].legend()

        axs[2].plot(t_arr, phiL.real, label='Re φ (Lorentz)', color='C2')
        axs[2].plot(t_arr, phiL.imag, '--', label='Im φ (Lorentz)', color='C3')
        axs[2].set_title('Lorentz: φ(t)'); axs[2].set_xlabel('t'); axs[2].set_ylabel('φ'); axs[2].legend()
        plt.tight_layout(); plt.show()


        # (C) Comparaison sur la fenêtre pour Γ₊
        plt.figure(figsize=(9,5))
        plt.plot(t_win, y, 'o', label='Im φ_L(norm)')
        plt.plot(t_win, Gamma_plus * x, '-', label='Γ₊·Φ₊(t)')
        plt.title(f"Fit sur fenêtre [{t1},{t2}] | k={k}  —  Γ₊={Gamma_plus:.6e},  Σ={Sigma_l:.6e}")
        plt.xlabel('t'); plt.ylabel('valeur'); plt.legend(); plt.tight_layout(); plt.show()

        # (D) Graphe combiné (sans |φ|)
        lam, phi = stitch_contour(tau_I, phi_I, tau_II, phi_II, t_arr, phiL)
        plt.figure(figsize=(9,5))
        plt.plot(lam, phi.real, label='Re φ')
        plt.plot(lam, phi.imag, '--', label='Im φ')
        plt.axvline(0.0, color='k', lw=0.8, alpha=0.5)
        plt.title('Solution combinée le long du contour λ')
        plt.xlabel('λ (τ en Euclide, t en Lorentz)'); plt.ylabel('φ')
        plt.legend(); plt.tight_layout(); plt.show()

    return {
        "k": k,
        "Gamma_plus": float(Gamma_plus),
        "Sigma_l": float(Sigma_l),
        "phi_tf_raw": complex(phi_tf_raw),
        "c_norm": complex(c_norm),
        "t_window": (float(t1), float(t2)),
    }

# --- Lancement en mode DEBUG uniquement ---
# --- Lancement ---
if __name__ == "__main__":
    if DEBUG:
        # mêmes courbes que NoBoundary.py en mode debug
        res = compute_and_plot_debug(k=5, t_window=(6.0, 8.0))
        print(f"[DEBUG] k={res['k']} -> Γ₊ = {res['Gamma_plus']:.6e},  Σ = {res['Sigma_l']:.6e},  φ_L(tf) = {res['phi_tf_raw']}")
    else:
        # mode production : σ(k) en fonction de k (sans autres plots)
        k_values = list(range(1, 20))
        sigma_num = []

        for k in k_values:
            out = compute_and_plot_debug(k=k, t_window=(6.0, 8.0))
            sigma_num.append(out["Sigma_l"])
            print(f"[k={k}]  Γ₊ = {out['Gamma_plus']:.6e}   Σ(k) = {out['Sigma_l']:.6e}")

        k_arr = np.array(k_values, dtype=float)
        sigma_num = np.array(sigma_num, dtype=float)

        plt.figure(figsize=(8,5))
        plt.plot(k_arr, sigma_num, 'o-', linewidth=2, label='Σ(k) numérique (wineglass)')
        plt.title("Σ(k) en fonction de k (wineglass)")
        plt.xlabel('k'); plt.ylabel('Σ(k)')
        plt.xticks(k_values)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()
