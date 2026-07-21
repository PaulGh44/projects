import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simpson
from typing import Tuple
from numpy.typing import ArrayLike

# ---------- STYLE ----------
plt.style.use("seaborn-v0_8")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "figure.figsize": (7, 5)
})

# ---------- GLOBAL PARAMS ----------
delta = 1e-2
ti, tf = 0.0, 10.0
N_euclid = 2000
N_lorentz = 3000
t_window = (6.0, 8.0)

# ---------- Euclidean region ----------
def f_of_z(z: float, k: int) -> float:
    return (k*(k/2+1))/(2*(1+z)**2) + k/(z**2 - 1) - k*(k+2)/(z**2 - 1)**2

def g_of_z(z: float, k: int) -> float:
    return k/(1+z) + 4*z/(z**2 - 1)

def sys_euclid(z: float, Y: Tuple[float,float], k: int) -> np.ndarray:
    chi, dchi = Y
    return np.array([dchi, -f_of_z(z,k)*chi - g_of_z(z,k)*dchi], dtype=float)

def a_1(k: int) -> float:
    return k*(k+4)/(4*(k+2))

z0 = -1.0 + delta
a, b = z0, 0.0

def set_ini_euclid(k: int) -> np.ndarray:
    return np.array([1.0 + a_1(k)*delta, a_1(k)], dtype=float)

def solve_euclid(k: int):
    t_eval = np.linspace(a, b, N_euclid)
    sol = solve_ivp(lambda z,Y: sys_euclid(z,Y,k),
                    (a,b), set_ini_euclid(k),
                    method="Radau", t_eval=t_eval,
                    rtol=1e-12, atol=1e-15)
    if not sol.success:
        raise RuntimeError(f"Euclidean solver failed for k={k}")
    return sol

def Phifinal(z, chi, k: int):
    pref = (k+2) / (2**((k+2)/2))
    return chi * np.power(1.0+z, k/2) * pref

def phiE_and_dphiE_at0(sol, k: int):
    z0_chk = sol.t[-1]
    chi, dchi = sol.y[0,-1], sol.y[1,-1]
    pref = (k+2) / (2**((k+2)/2))
    phi0_E = pref * chi * (1.0+z0_chk)**(k/2)
    dphi0_E_z = pref * (dchi*(1.0+z0_chk)**(k/2)
                        + chi*(k/2)*(1.0+z0_chk)**(k/2-1))
    return complex(phi0_E), complex(dphi0_E_z)

# ---------- Lorentzian region ----------
def a_of_t(t: float) -> float: return np.cosh(t)
def a_dot_of_t(t: float) -> float: return np.sinh(t)
def f_of_t(t: float, k: int) -> float: return k*(k+2)/(a_of_t(t)**2)
def g_of_t(t: float, k: int) -> float: return 3*a_dot_of_t(t)/a_of_t(t)

def sys_lorentz(t: float, y: np.ndarray, k: int) -> np.ndarray:
    Re_phi, Im_phi, Re_dphi, Im_dphi = y
    f, g = f_of_t(t,k), g_of_t(t,k)
    return np.array([Re_dphi,
                     Im_dphi,
                     -g*Re_dphi - f*Re_phi,
                     -g*Im_dphi - f*Im_phi], dtype=float)

def set_ini_lorentz_from_euclid(solE, k: int) -> np.ndarray:
    phi0_E, dphi0_E_z = phiE_and_dphiE_at0(solE,k)
    phi0, dphi0 = phi0_E, 1j*dphi0_E_z
    return np.array([phi0.real, phi0.imag, dphi0.real, dphi0.imag], dtype=float)

def solve_lorentz_from_euclid(k: int, solE):
    t_eval = np.linspace(ti, tf, N_lorentz)
    y0 = set_ini_lorentz_from_euclid(solE,k)
    sol = solve_ivp(lambda t,y: sys_lorentz(t,y,k),
                    (ti,tf), y0, method="Radau",
                    t_eval=t_eval, rtol=1e-12, atol=1e-15)
    if not sol.success:
        raise RuntimeError(f"Lorentzian solver failed for k={k}")
    return sol

# ---------- Hypergeometric Φ₊(t) ----------
def phi_plus(t, k: int):
    return np.exp(-3.0*np.asarray(t))

# ---------- Normalization at tf ----------
def normalize_lorentz_at_tf(phiL: np.ndarray) -> tuple[np.ndarray, complex]:
    phi_tf = phiL[-1]
    c = np.exp(-1j*np.angle(phi_tf)) / np.abs(phi_tf)
    return c*phiL, c

# ---------- Core observable ----------
def compute_sigma(k: int, t_window=(6.0,8.0)) -> float:
    solE = solve_euclid(k)
    solL = solve_lorentz_from_euclid(k, solE)
    t_arr, phiL = solL.t, solL.y[0]+1j*solL.y[1]
    phiL_norm, _ = normalize_lorentz_at_tf(phiL)

    mask = (t_arr>=t_window[0]) & (t_arr<=t_window[1])
    t_win = t_arr[mask]
    y, x = np.imag(phiL_norm[mask]), phi_plus(t_win,k)

    num_int = simpson(y*x, x=t_win)
    den_int = simpson(x*x, x=t_win)
    Gamma_plus = num_int/den_int
    return -4.0/(3.0*Gamma_plus)

# ---------- Main ----------
if __name__ == "__main__":
    k_values = np.arange(1,31)
    sigma_vals = np.array([compute_sigma(k) for k in k_values])
    Delta2 = (k_values**3)/(2*np.pi**2) * sigma_vals

    # --- Fit spectral index ns only on l in [10,30] ---
fit_min, fit_max = 10, 30
mask_fit = (k_values >= fit_min) & (k_values <= fit_max)

x_fit = np.log(k_values[mask_fit])
y_fit = np.log(Delta2[mask_fit])

coeffs = np.polyfit(x_fit, y_fit, 1)
ns = coeffs[0] + 1
logC = coeffs[1]

# Uncertainty on slope from residuals over the *fit range*
residuals = y_fit - (coeffs[0]*x_fit + logC)
sigma_ns = np.std(residuals) / np.std(x_fit)

# Fitted curve (shown across full range for visualization)
Delta2_fit = np.exp(logC) * k_values**(ns - 1)

# --- Plot log-log ---
plt.figure(figsize=(7,5))
plt.loglog(k_values, Delta2, 'o-', label=r'$\Delta_\Phi^2(\ell)$ (num)')
plt.loglog(k_values, Delta2_fit, '--', 
           label=fr'fit on [{fit_min},{fit_max}]: $n_s={ns:.3f}\pm{sigma_ns:.3f}$')
plt.axvspan(fit_min, fit_max, color='grey', alpha=0.15, label='fit range')
plt.title("Dimensionless power spectrum (log-log)")
plt.xlabel(r'$\ell$'); plt.ylabel(r'$\Delta_\Phi^2(\ell)$')
plt.legend(); plt.grid(True, which="both", linestyle='--', alpha=0.7)
plt.tight_layout()

# --- Plot linear ---
plt.figure(figsize=(7,5))
plt.plot(k_values, Delta2, 'o-', label=r'$\Delta_\Phi^2(\ell)$ (num)')
plt.plot(k_values, Delta2_fit, '--', 
         label=fr'fit on [{fit_min},{fit_max}]: $n_s={ns:.3f}\pm{sigma_ns:.3f}$')
plt.axvspan(fit_min, fit_max, color='grey', alpha=0.15, label='fit range')
plt.title("Dimensionless power spectrum (linear)")
plt.xlabel(r'$\ell$'); plt.ylabel(r'$\Delta_\Phi^2(\ell)$')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()
