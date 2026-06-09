#!/usr/bin/env python
# coding: utf-8

# # Conformal bootstrap approach to Liouville theory
# In this notebook we numerically check crossing symmetry in Liouville theory and generalized minimal models.

# ## Liouville theory with $c\notin]-\infty,1]$
# Let us numerically check the identity
# $$
# \int_{\frac{Q}{2}+i\mathbb{R}} d\alpha\ C_{\alpha_1,\alpha_2,Q-\alpha} C_{\alpha,\alpha_3,\alpha_4} \mathcal{F}^{(s)}_\alpha(z) \mathcal{F}^{(s)}_\alpha(\bar z) = \int_{\frac{Q}{2}+i\mathbb{R}} d\alpha\ C_{\alpha_2,\alpha_3,Q-\alpha} C_{\alpha,\alpha_1,\alpha_4} \mathcal{F}^{(t)}_\alpha(z) \mathcal{F}^{(t)}_\alpha(\bar z)
# $$
# where $c=1+6Q^2$ is the central charge, $\alpha_1,\cdots,\alpha_4$ are four momentums, $C_{\alpha_1,\alpha_2,Q-\alpha}$ is a DOZZ structure constant, and $\mathcal{F}^{(s)}_\alpha(z)$ is an $s$-channel conformal block. A momentum $\alpha$ can alternatively be parametrized by $P$ such that $\alpha = \frac{Q}{2}+iP$.

# In[ ]:


from __future__ import division, print_function
from CFT import *
from Correlators import *
from Data import *
from IPython.display import display, Math
import mpmath


# In[ ]:


c = 15 + 4*1j             # the central charge
Ps = [1.3, 1.01, .45, .22]   # the momentums
z = .3 + .2*1j            # the position
N = 6                     # a cutoff parameter that controls the precision of blocks

display(Math('c = '+ str(c)))


# In[ ]:


b = Charge('c', c).get('b')   # the coupling constant
q = mpmath.qfrom(m = z)       # the elliptic nome 

s_channel = FourPoint(b = b, Ps = Ps, Nmax = N).value(q = q)
t_channel = FourPoint(t_channel = True, b = b, Ps = Ps, Nmax = N).value(q = q)

display(Math(r's{\rm -channel}:  ' + str(s_channel)))
display(Math(r't{\rm -channel}:  ' + str(t_channel)))


# In[ ]:


M = 20
zs = [.01 + i*1.0/M for i in range(M)]   # many values of the position z

""" We compute the four-point functions for all these values, 
using the high-level function 'Data'. 
"""
data = Data(spline = 30, x = zs, b = b, Ps = Ps, Pcutoff = 3, 
              Nmax = [0, 2, 4], reduced = True)


# In[ ]:


graph = Graph(data, Nmax = [2, 4])

graph.show()


# ## Liouville theory with $c\leq 1$
# Same thing with the structure constant $\hat C$ instead of $C$. We use the variable $\beta= ib$, which is real if $c\leq 1$.

# In[ ]:


c = .237    # a central charge less than one

Ps = [1.3, 1.01, .45, .22]   # the momentums
z = .3 + .2*1j            # the position
N = 6                     # a cutoff parameter that controls the precision of blocks

U = 60      # a cutoff parameter that controls the precision of structure constants

display(Math('c = '+ str(c)))


# In[ ]:


beta = Charge('c', c).get('beta')   # the real coupling constant
q = mpmath.qfrom(m = z)             # the elliptic nome 

s_channel = Z4(beta = beta, Ps = Ps, Nmax = N, Ucutoff = U).value(q = q)
t_channel = Z4(t_channel = True, beta = beta, Ps = Ps, Nmax = N, Ucutoff = U).value(q = q)

display(Math(r's{\rm -channel}:  ' + str(s_channel)))
display(Math(r't{\rm -channel}:  ' + str(t_channel)))


# ## Generalized minimal model

# In[ ]:


cs = [.3 + .8 * 1j, -.5 + .7 * 1j, 3 - .1 * 1j, .789, -4.56]   # a number of values of c
pairs = [[2, 3], [3, 2], [4, 2], [3, 3]]           # the labels of four degenerate fields 
z = 0.23
N = 10

""" We compute the four-point functions in both channels, 
using the high-level function 'Data'. 
"""
datas = []
for c in cs:
    charge = Charge('c', c)
    fields = [Dimension('degenerate', pair, charge) for pair in pairs]
    datas.append(Data(theory = 'GMM', fields = fields, x = [z], Nmax = N))


# In[ ]:


""" We display the results. """
table = Table(datas, parameter_values = cs, parameter = 'c', digits = 7)
table.display()


# ## Kac tables of minimal models
# The necessary code is freely available on GitHub. In addition to computing four-point functions, it can do more basic things, such as displaying Kac tables of minimal models.

# In[ ]:


print( Charge('minimal', [4, 3]).KacTable() )   # the Ising model


# In[ ]:


pair = [10, 7]
print( Charge('minimal', pair).KacTable() )      # another minimal model


# In[ ]:




