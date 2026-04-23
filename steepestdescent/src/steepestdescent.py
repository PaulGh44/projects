import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
#Global variables
mu =1
l1=np.sqrt(2)/2
l2=np.sqrt(2)/2

def find_saddles(S, z0, tol=1e-4, max_iter=1000):
    """
    Find the saddles of a given phase function S using Newton's method.

    Parameters:
    S : function
        The phase function for which to find the saddles.
    z0 : complex
        Initial guess for the saddle point.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    Returns:
    z : complex
        The saddle point found.
    """
    z = z0
    for i in range(max_iter):
        S_prime = (S(z + tol) - S(z - tol)) / (2 * tol)  # Numerical derivative
        S_double_prime = (S(z + tol) - 2 * S(z) + S(z - tol)) / (tol ** 2)  # Numerical second derivative
        
        if abs(S_double_prime) < tol**2:  # Avoid division by zero
            print("Second derivative is too small, stopping iteration.")
            break
        
        z_new = z - S_prime / S_double_prime  # Newton's method update
        
        if abs(z_new - z) < tol:  # Check for convergence
            print(f"Converged to saddle point at {z_new} after {i+1} iterations.")
            return z_new
        
        z = z_new
    
    print("Maximum iterations reached without convergence.")
    return z

#Modify such function to display the analytic landscape of 1/S'(z) instead of S'(z) to better visualize the saddle points, where S'(z) is the derivative of the phase function S(z). Add the real and imaginary axis in black to the plot for better visualization of the saddle points.
def plot_saddle_landscape(S, x_range=(-2, 2), y_range=(-2, 2), resolution=100):
    """
    Plot the analytic landscape of the derivative of the phase function S'(z) in the complex plane.

    Parameters:
    S : function
        The phase function for which to plot the landscape.
    x_range : tuple
        The range of x values (real part) to plot.
    y_range : tuple
        The range of y values (imaginary part) to plot.
    resolution : int
        The number of points in each dimension for the plot.
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    # Compute the derivative S'(z) using numerical differentiation
    S_prime = (S(Z + 1e-6) - S(Z - 1e-6)) / (2 * 1e-6)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, np.abs(1/S_prime), levels=50, cmap='viridis')
    plt.colorbar(label="|1/S'(z)|")
    plt.title("Analytic Landscape of 1/S'(z)")
    plt.axhline(0, color='red', lw=0.3)  # Real axis
    plt.axvline(0, color='red', lw=0.3)  # Imaginary axis
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.grid()
    plt.show()
#Write a function that to a function f returns its analytic landscape.
def plot_analytic_landscape(f, x_range=(-2, 2), y_range=(-2, 2), resolution=100):
    """
    Plot the analytic landscape of a given function f in the complex plane.

    Parameters:
    f : function
        The function for which to plot the landscape.
    x_range : tuple
        The range of x values (real part) to plot.
    y_range : tuple
        The range of y values (imaginary part) to plot.
    resolution : int
        The number of points in each dimension for the plot.
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    F = f(Z)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, np.abs(F), levels=50, cmap='viridis')
    plt.colorbar(label="|f(z)|")
    plt.title("Analytic Landscape of f(z)")
    plt.axhline(0, color='red', lw=0.3)  # Real axis
    plt.axvline(0, color='red', lw=0.3)  # Imaginary axis
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.grid()
    plt.show()


# Demo to test the plot_saddle_landscape function with the same phase function S(z)
if __name__ == "__main__":
    def S(z):
        return -1j*mu*z+np.log(z)+np.log(np.sinh(z/2))+1j*(l1+l2)**2/(np.tanh(z/2)*2)
    def G(z):
        return np.exp(-S(z))*sp.erf(np.exp(-1j*np.pi/4)*np.sqrt(l1*l2)*np.tanh(z/2)/(np.sqrt(2)))
    
    print("Saddle points:", find_saddles(S, z0=0.7+1j))
    plot_saddle_landscape(S,x_range=(-10, 10), y_range=(-10, 10), resolution=500)
    