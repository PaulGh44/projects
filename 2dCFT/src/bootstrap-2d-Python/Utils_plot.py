#Packages
import matplotlib.pyplot as plt
import numpy as np


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


    
    
    




    



    
    




    


    
    

    

    
    
