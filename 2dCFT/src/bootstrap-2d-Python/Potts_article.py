#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division, print_function
from Spectrum_bootstrap import *
from sympy import Rational
from Data import Table
import math, cmath

import matplotlib.pyplot as plt
plt.rc("figure", facecolor = "white")
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

from CFT import *


# # A conformal bootstrap approach to critical percolation in two dimensions
# In this notebook we perform numerical conformal bootstrap calculations for the article by Marco Picco, Sylvain Ribault and Raoul Santachiara. 

# ## Testing the code in a known case: generalized minimal models
# Let us see whether we can compute four-point functions in generalized minimal models, based on the ansatz that the spectrum of a four-point function is dictated by the fusion rules of the fields. Since all states in generalized minimal models are diagonal and degenerate, the resulting spectrum is made of finitely many diagonal states.
# 
# We choose a generic value of the central charge $c=.22+.17i$, and the degenerate field with labels $(4,3)$. We display the states that can appear in the four-point function $\left<V_{(4,3)}V_{(4,3)}V_{(4,3)}V_{(4,3)}\right>$. We first display the state labels, spins and conformal dimensions, and see that the ground state is the state $(7,5)$.
# We then display the known structure constants, normalized so that the constant for the ground state is one. 

# In[ ]:


c = .22 + .17*1j
pair = [4, 3]
spectrum = GMM_Spectrum(pair = pair, charge = Charge('c', c))
print( 'Conformal dimensions. ' )
print( '(left label) (right label) spin [(left dimension), (right dimension)]' )
spectrum.truncate(show = True)
print( 'Structure constants. ' )
print( '(label) constant' )
spectrum.display(real_part = False)


# We use the bootstrap method for deriving the structure constants. Since the spectrum has only $12$ fields, we do not truncate it. The structure constants for the $9$ lowest states are in good agreement with the known results, and the coefficient of variation of structure constants give good indications of how well they agree with the known results. 

# In[ ]:


stats = Struct_Csts_Stats(spectrum = spectrum, Nsamples = 3, L = 12, blocklevel = 10)
print( '(label) (mean structure constant) coefficient of variation' )
stats.display_cst()
print( 'computer time in seconds: ', stats.duration )


# ## The case of percolation
# Let us consider the ansatz $\mathcal{S}_{2\mathbb{Z},\mathbb{Z}+\frac12}$ at $c=0$ for the spectrum of a four-point function of fields with dimensions $\Delta_{(0,\frac12)}$. We display the results for mean structure constants and coefficients of variation as a Latex table.

# In[ ]:


c = 10**(-9)
charge = Charge('c', c)
beta = charge.get('beta')
spectrum = Another_Spectrum(beta = beta, spacings = [2, 1], shifts = [False, True])
stats = Struct_Csts_Stats(spectrum = spectrum, Nsamples = 3, L = 5.5, blocklevel = 16)
print( 'computer time in seconds: ', stats.duration )


# In[ ]:


M = 16  # number of states that we display

pretty_indices = [(Rational(pair[0]), Rational(pair[1])) for pair in stats.indices]
pretty_csts = [Table.format_result(cst, digits = 10, real = True) for cst in stats.mean_cst]
pretty_cvar = [Table.format_precision(var) for var in stats.cvar]
exact_charge = Charge('c', 0)
total_dims = [(Dimension('degenerate', pair, exact_charge).get('Delta', num = False),
              Dimension('degenerate', (pair[0], -pair[1]), exact_charge).get('Delta', num = False))
              for pair in pretty_indices]
string = ( r"\begin{array}{rrrc} (r,\quad s) & (\Delta,\quad \bar\Delta) " + 
          r" & D_{\Delta,\bar\Delta}\qquad & c_{\Delta,\bar\Delta} \\ \hline" )
for i in range(M):
    string += ( latex(pretty_indices[i]) + r" & " + latex(total_dims[i]) + r" & "
               + pretty_csts[i] + r" & " + pretty_cvar[i] + r" \\ ")
string += r" \hline \end{array}"
# print( string )
display(Math(string))


# Let us quickly generate data for the plots in the article. The precision need not be very high, and we keep only the first $7$ states.

# In[ ]:


L = 3.2
print( 'Dimensions: ' )
spectrum.truncate(L = L, show = True) 
csts = Struct_Csts(spectrum = spectrum, L = L, blocklevel = 6)
print( 'Structure constants: ' )
csts.constants()


# In[ ]:


rho_min = .001
rho_max = 1.2
N_points = 30
ylims = [.95, 1.25]
thetas = [0, math.pi/6, math.pi/4, math.pi/3]
theta_names = [r'$\theta = 0$', r'$\theta =\frac{\pi}{6}$', 
               r'$\theta =\frac{\pi}{4}$', r'$\theta =\frac{\pi}{3}$']
digits = 5

Delta = csts.field.get('Delta')   # Dimension of the field whose four-point function we compute.
rho_interval = (rho_max - rho_min) / N_points
rhos = [rho_min + i * rho_interval for i in range(N_points + 1)]
fig = plt.figure()
fig.canvas.set_window_title('A four-point function in two-dimensional percolation') 

for theta in thetas:
    values = []
    phase = cmath.exp(theta * 1j)
    for rho in rhos:
        z = rho * phase
        factor = rho ** (2 * Delta)   # Prefactor that ensures finiteness at z = 0.
        value = factor * csts.values(z, reduced = False, channel = True)
        values.append(Table.format_result(value, digits = digits, real = True))
    plt.plot(rhos, values, label = theta_names[thetas.index(theta)])

plt.gca().tick_params(axis = 'x', pad = 8)
plt.gca().tick_params(axis = 'y', pad = 8)   
plt.gca().set_ylim(ylims)
plt.legend()
plt.show()


# ## Other values of the central charge
# After critical percolation ($q=1$, $c=0$), we explore other values of the number of states $q$, equivalently of the central charge $c$. From $q$, we deduce the parameter $\beta$ as 
# $$
# \beta = \sqrt{1- \frac{1}{2\pi} \arccos\left(\frac{q}{2}-1\right)}
# $$
# Then we check that the spectrum $\mathcal{S}_{2\mathbb{Z},\mathbb{Z}+\frac12}$ is consistent for arbitrary complex central charges such that $\Re c < 13$.

# In[ ]:


def BfromQ(q):
    return math.sqrt(1 - math.acos(q/2-1)/2/math.pi)
qs = [1, 1.5, 2.5, 3]
q_names = [r'$q=1$', r'$q=1.5$', r'$q=2.5$', r'$q=3$']
L = 3.2

fig = plt.figure()
fig.canvas.set_window_title('A four-point function in the Potts model') 

for q in qs:
    beta = BfromQ(q)
    values = []
    spectrum = Another_Spectrum(beta = beta, spacings = [2, 1], shifts = [False, True])
    # print( 'Dimensions: ' )
    # spectrum.truncate(L = L, show = True) 
    csts = Struct_Csts(spectrum = spectrum, L = L, blocklevel = 6)
    csts.constants()
    Delta = csts.field.get('Delta') # Dimension of the field whose four-point function we compute.
    for rho in rhos:
        factor = rho ** (2 * Delta)   # Prefactor that ensures finiteness at z = 0.
        value = factor * csts.values(rho, reduced = False, channel = True)
        values.append(Table.format_result(value, digits = digits, real = True))
    plt.plot(rhos, values, label = q_names[qs.index(q)])

plt.gca().tick_params(axis = 'x', pad = 8)
plt.gca().tick_params(axis = 'y', pad = 8)   
plt.gca().set_ylim(ylims)
plt.legend()
plt.show()


# In[ ]:


c = 3 + 2 * 1j
charge = Charge('c', c)
beta = charge.get('beta')
spectrum = Another_Spectrum(beta = beta, spacings = [2, 1], shifts = [False, True])
stats = Struct_Csts_Stats(spectrum = spectrum, L = 5, blocklevel = 10)
print( '(label) (mean structure constant) coefficient of variation' )
stats.display_cst()
print( 'computer time in seconds: ', stats.duration )


# ## A failed extrapolation from $c=1$
# 
# At $c=1$ we know that the spectrum $\mathcal{S}_{2\mathbb{Z},\mathbb{Z}}$ is consistent, and correctly describes the $u$-channel behaviour of the four-point function whose $s$- and $t$-channel spectrum is $\mathcal{S}_{2\mathbb{Z},\mathbb{Z}+\frac12}$. We will now see that it is inconsistent for $c\neq 1$, while becoming less and less so as $c\to 1$ i.e. $\beta \to 1$. 
# 
# Notice that for $\beta \to 1$ some coefficients of variations remain high for states that share the same dimension. Such states occur due to the identity $(m, n) = (n, m)$ that is valid at $c=1$. So for example $(2, 0) = (0, 2)$ coincide at $c=1$, but are counted as two different states in $\mathcal{S}_{2\mathbb{Z},\mathbb{Z}}$.
# 
# These calculations are first done using the naive regularization of divergent conformal blocks, where infinite terms are simply ignored. We then check that the logarithmic regularization does not solve the problem.

# In[ ]:


betas = [.61, .81, .95, .99, .9999, .999999]
for beta in betas:
    print( 'beta = ', beta )
    spectrum = Another_Spectrum(beta = beta, spacings = [2, 1], shifts = [False, False])
    stats = Struct_Csts_Stats(spectrum = spectrum, L = 5, blocklevel = 10)
    print( '(label) (mean structure constant) coefficient of variation' )
    stats.display_cst()
    print( 'computer time in seconds: ', stats.duration )


# In[ ]:


beta = .81
spectrum = Another_Spectrum(beta = beta, spacings = [2, 1], shifts = [False, False], deg = True)
"""
The option 'deg = True' switches on the logarithmic regularization, including adding the relevant
diagonal states to the spectrum. By convention, the first index of such a diagonal state is
negative. For example, (-2, 1) is the diagonal state that appears in the logarithmic
regularization of the non-diagonal states (2, -1) and (2, 1).
"""
stats = Struct_Csts_Stats(spectrum = spectrum, L = 5, blocklevel = 10)
print( '(label) (mean structure constant) coefficient of variation' )
stats.display_cst()
print( 'computer time in seconds: ', stats.duration )


# ## Another consistent spectrum
# 
# For $\Re c<13$, we find that the spectrum $\mathcal{S}_{2\mathbb{Z}+1,\mathbb{Z}}$ is consistent for a four-point function of fields with dimensions $\Delta_{(0,\frac12)}$. 

# In[ ]:


c = 3 + 2 * 1j
charge = Charge('c', c)
beta = charge.get('beta')
spectrum = Another_Spectrum(beta = beta, spacings = [2, 1], shifts = [True, False])
stats = Struct_Csts_Stats(spectrum = spectrum, L = 5, blocklevel = 10)
print( '(label) (mean structure constant) coefficient of variation' )
stats.display_cst()
print( 'computer time in seconds: ', stats.duration )

