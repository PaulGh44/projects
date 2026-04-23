import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp

mp.dps = 20

class ComplexStructure:
    def __init__(self, func, xlim=(-2,2), ylim=(-2,2), resolution=300):
        self.func = func
        self.xlim = xlim
        self.ylim = ylim
        self.resolution = resolution

@staticmethod
def wrapped_angle_diff(a,b):
    return np.angle(np.exp(1j*(a-b)))

def sample(self):
    x = np.linspace(self.xlim[0], self.xlim[1], self.resolution)
    y = np.linspace(self.ylim[0], self.ylim[1], self.resolution)
    X, Y = np.meshgrid(x, y)

    F = np.full(X.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    finite_mask = np.zeros(X.shape, dtype=bool)

    for i in range(self.resolution):
        for j in range(self.resolution):
            z = self._safe_mpc(X[i, j], Y[i, j])
            try:
                val = self.func(z)
                val_complex = complex(val)
                if np.isfinite(val_complex.real) and np.isfinite(val_complex.imag):
                    F[i, j] = val_complex
                    finite_mask[i, j] = True
            except Exception:
                pass

    self._sample = (X, Y, F, finite_mask)
    return self._sample

def diagnostic(self):
    X, Y, F, finite_mask = self.sample()
    