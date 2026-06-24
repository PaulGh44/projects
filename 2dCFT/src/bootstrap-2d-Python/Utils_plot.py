#Packages
import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp
from Special_functions_rational import Theory


# =========================================
# Useful modules to plot a bunch of different things
# =========================================

class utils_plot:

    @staticmethod
    def plot_phase2D(func, x_min=-2, x_max=2, y_min=-2, y_max=2, resolution=100):

        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X, dtype=np.float64)

        for i in range(resolution):
            for j in range(resolution):
                z_val = func(mp.mpc(X[i, j], Y[i, j]))
                Z[i, j] = mp.arg(z_val)

        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, Z, levels=100, cmap='hsv')
        plt.colorbar(label='Phase (radians)')
        plt.title('Phase of the function in the complex plane')
        plt.xlabel('Real part')
        plt.ylabel('Imaginary part')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()

    @staticmethod
    def plot_module1D(func, name: str, x_min=-5, x_max=5, resolution=200):
        x = np.linspace(x_min, x_max, resolution)
        y = np.zeros(resolution, dtype=float)

        for i in range(resolution):
            z_val = func(mp.mpc(x[i]))
            y[i] = float(abs(z_val))

        plt.figure(figsize=(8, 6))
        plt.plot(x, y)
        plt.title('Modulus of the function')
        plt.xlabel('p')
        plt.ylabel(name)
        plt.xlim(x_min, x_max)
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_real1D(func, name: str, x_min=-3, x_max=3, resolution=200):
        x = np.linspace(x_min, x_max, resolution)
        y = np.zeros(resolution, dtype=float)

        for i in range(resolution):
            z_val = func(mp.mpc(x[i]))
            y[i] = float(mp.re(z_val))

        plt.figure(figsize=(8, 6))
        plt.plot(x, y)
        plt.title('Real part of the function')
        plt.xlabel('p')
        plt.ylabel(name)
        plt.xlim(x_min, x_max)
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_realandimag(
        Theory: Theory,
        listreal: list[float],
        listimag: list[float],
        listpoints: list[float],
        name: str = "z",
        Parameters: list[complex] = None
    ):
        if len(listreal) != len(listpoints) or len(listimag) != len(listpoints):
            raise ValueError("The lengths of the lists are not the same")

        plt.figure(figsize=(8, 6))

        plt.scatter(listpoints, listreal, label=r"$\Re(z)$", marker="o")
        plt.scatter(listpoints, listimag, label=r"$\Im(z)$", marker="x")

        plt.title(rf"Real and imaginary parts of the relative error for $(m,n)=({Theory.m},{Theory.n})$ as a function of the cross-ration $x$")
        plt.xlabel(r"$x$")
        plt.ylabel(name)
        plt.grid(True)
        plt.legend()
        plt.show()
        

    
    
    




    



    
    




    


    
    

    

    
    
