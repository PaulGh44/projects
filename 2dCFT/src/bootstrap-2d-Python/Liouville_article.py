#!/usr/bin/env python
# coding: utf-8

# This notebook displays sample calculations which support the claims of the article "Liouville theory with a central charge less than one", by Sylvain Ribault and Raoul Santachiara.

# In[ ]:


from __future__ import print_function
from CFT import *
from Data import *
import math
from IPython.display import display, Math         


# # 1 Introduction

# # 2 Liouville theory in the conformal bootstrap approach

# ## 2.1 Definition and spectrum of Liouville theory

# Entering a value for the central charge, and converting to other notations, including $b$ and $Q$ such that 
# $$
# c = 1 + 6Q^2 \quad , \quad Q= b+ \frac{1}{b}
# $$

# In[ ]:


c = 27.123
charge1 = Charge('c', c)
b = charge1.get('b')
print( 'b =' , b )
charge1.display()

b = 2.1 * 1j
charge2 = Charge('b', b)
charge2.display()


# Entering the conformal dimension of a primary field, and converting to other notations, including $\alpha$ and $P$ such that 
# $$
# \Delta = \alpha(Q-\alpha) = \frac{Q^2}{4} + P^2
# $$

# In[ ]:


c = 12.123
Delta = 1.1
primary = Dimension('Delta', Delta, Charge('c', c))
primary.display()


# ## 2.2 Crossing symmetry equations for correlation functions

# Large $\Delta$ asymptotics of conformal blocks:
#  $$\mathcal{F}^{(s)}_\alpha(\alpha_i|z_i) \underset{|\Delta| \to \infty}{\sim} (16q)^{\Delta}\quad \text{with} \quad |q| < 1$$
#  To check this, we will study the ratio of the two sides as $\Delta \to \infty$, and see if it has a constant limit.

# In[ ]:


""" We give values to the momentums P_i and the position q. """
c = 32.1
momentums = [0.3, .123, 0.79, 0.82]
q = 0.1

""" We build a numerical block, which is a function of Delta.
This depends on a cutoff Nmax, which is defined in the Appendix. 
"""
dimensions = [Dimension('P', P, Charge('c', c)) for P in momentums]
block = Block(dims = dimensions, Nmax = 2)
numerical_block = BlockNum(block, q)

""" We choose a number of values of Delta. """
Deltas = [.01 + i**2 * .15 for i in range(20)]

""" We compute the corresponding values of the ratio. """
ratio = [(numerical_block.value(Delta) * (16*q)**(-Delta)).real
         for Delta in Deltas]

""" The rest is plotting. """
plt.figure('Asymptotics of a conformal block')
plt.scatter(Deltas, ratio, marker = 'o', color = 'blue')
plt.title('Large $\Delta$ asymptotics of a conformal block')
plt.ylabel('ratio')
plt.xlabel('dimension $\Delta$')
plt.legend(loc = 'upper right')
plt.show()    


# ## 2.3 Solving the crossing symmetry equations

# Asyptotic behaviour of the function $\Upsilon_{b}$:
# $$\log \Upsilon_b(\tfrac{Q}{2}  + iP) \underset{P\to \infty}{=} -P^2\log|P| +\frac32 P^2 + o(P^2)
# $$
# We reformulate this as $\underset{P\to\infty}{\lim} g(P) = \frac32$, where 
# $$
# g(P) =  \frac{1}{P^2} \log \Upsilon_b(\tfrac{Q}{2}  + iP) + \log|P| 
# $$
# which we proceed to check directly.

# In[ ]:


""" We compute some values of our function g. """
b = 2.1
Ps = [1.2 + i for i in range(20)]
gs = [cmath.log(Upsilons(b = b, Plist = [[P, True]]).integral())/P**2 + cmath.log(P) 
             for P in Ps]

""" The rest is plotting. """
plt.figure('Asymptotics of the Upsilon function')
plt.scatter(Ps, gs, marker ='o', color ='blue')
plt.title(r'Large $P$ asymptotics of $\Upsilon_b(\frac{Q}{2}+iP)$')
plt.ylabel('function $g(P)$')
plt.xlabel('momentum $P$')
plt.legend(loc='upper right')
plt.show()


# Let us check crossing symmetry of DOZZ - Liouville theory for complex central charges. 
# This amounts to checking the equality
# $$
# \int_{\frac{Q}{2}+i\mathbb{R}} d\alpha_s\ C^\text{DOZZ}(\alpha_1,\alpha_2,Q-\alpha_s) C^\text{DOZZ}(\alpha_s,\alpha_3,\alpha_4)\, 
# \Big|\mathcal{F}^{(s)}_{\alpha_s}(\alpha_i|z_i)\Big|^2
# \\
# =
# \int_{\frac{Q}{2}+i\mathbb{R}} d\alpha_t\ C^\text{DOZZ}(\alpha_2,\alpha_3,Q-\alpha_t) C^\text{DOZZ}(\alpha_t,\alpha_4,\alpha_1)\, 
# \Big|\mathcal{F}^{(t)}_{\alpha_t}(\alpha_i|z_i)\Big|^2
# $$
# We start with an example.

# In[ ]:


""" We choose values for b, the four momentums, and the cross-ratio x. """
b1 = 1.3428
P1s = [1.3, 1.01, .45, .22]
x1 = 0.363

""" We build the corresponding objects, and display the resulting central charge and conformal dimensions. """
charge1 = Charge('b', b1)
display(Math(('c=' + '{:.3f}'.format(float(charge1.get('c'))))))
Deltas = map(lambda x: float('{:.3f}'.format(x)), 
             [float(Dimension('P', P, charge1).get('Delta')) for P in P1s])
display(Math(r'\Delta_i =' + str(Deltas)))

""" We compute the four-point functions in both channels for our chosen parameters. 
We also specify a number of parameters of the numerical evaluation, as explained in the Appendix.
The parameter which controls the precision is the cutoff Nmax in the computation of the conformal blocks.
""" 
data1 = Data(theory = 'Liouville', b = b1, Ps = P1s, x = [x1], Nmax = [0, 2, 4, 6], Pcutoff = 3, spline = 60)


# In[ ]:


""" We display the results. We can choose how many digits we keep. """
Table(data1, digits = 3).display()


# We now check crossing symmetry for a number of values of the central charge. 
# 

# In[ ]:


""" We choose several values of c, the four momentums, and the cross-ratio x. We also choose Nmax. """
c2s = [36.74, 17.55, 2.12, 3 + 4*1j, .5 + 2*1j, -1 + 2.1*1j]
P2s = [1.3, 1.01, .45, .22]
x2 = 0.26
Nmax2 = 6

""" We compute the four-point functions in both channels. """
b2s = [complex(Charge('c', c).get('b')) for c in c2s]
data2s = [Data(theory = 'Liouville', b = b, Ps = P2s, x = [x2], Nmax = Nmax2, Pcutoff = 3) for b in b2s]


# In[ ]:


""" We display the results. Correlation functions can be complex: we display the real parts, but the relative
difference between the channels is computed from the complex values. """
table2 = Table(data2s, parameter_values = c2s, parameter = 'c', digits = 5)
table2.display()

""" For use in the article, we print the Latex source of the table, without computation times. """
print( Table(data2s, parameter_values = c2s, parameter = 'c', digits = 5, time = False).tables[0][0] )


# # 3 The problem with $c\leq 1$

# ## 3.1 The three-point structure constant

# We plot the three-point structure constant $C^{c\leq 1}$ 
# as a function of the momentums, for real values of the momentums.
# Since $C^{c\leq 1}$ has poles, we actually plot
# $C^{c\leq 1}(P_1, P_2, P_3+i\epsilon)$ as a function of $P_3$, and the poles manifest themselves as peaks: the smaller $\epsilon$, the sharper the peaks.

# In[ ]:


""" We choose the parameters. """
beta = 1.103
P1, P2 = 0.72, 0.95
epsilon = 5*10**(-3)

""" We also choose the range of values of P3, the number of values of P3. """
Pmin, Pmax = P1 - 0.1, P2 + 0.8
points = 200

""" We compute the values of P3 and of the three-point structure constant. """
Ps = [Pmin+ (Pmax - Pmin) * i / points + 1j * epsilon for i in range(points)]
Cs = [Z3(beta = beta, Ps = [P1, P2, P]).upsilons.integral().real for P in Ps]

""" The rest is plotting. """
plt.figure('Momentum dependence of the three-point structure constant')
plt.plot(Ps, Cs, color ='red')
plt.title(r'$C^{c\leq 1}(P_1=%.2f, P_2=%.2f, P_3+i\epsilon)$ as a function of $P_3$' %(P1, P2))
plt.ylabel(r'structure constant')
plt.xlabel(r'$P_3$')
plt.show()


# We numerically test our equation which involves structure constants at two different values of the central charge, parametrized by $\beta$ and $\beta' = \frac{\beta}{\sqrt{2}}$.
# To do this we adopt the notations of Hadasz, Jaskolski and Suchanek, which differ the notations in our article. In these notations, the relation becomes
# $$ 
# \frac{16^{-4P^2}C_{(\beta)}^{c\leq 1}(P,P_0,-P)}{C_{(\beta')}^{c\leq 1}\left(\frac{1}{4\beta'},-\frac{1}{4\beta'},\sqrt{2}P\right)C_{(\beta')}^{c\leq 1}\left(\frac{1}{4\beta'},\frac{P_0}{\sqrt{2}},-\sqrt{2}P\right)}=g_{\beta}(P_0)
# $$
# where
# $$
# g_{\beta}(P_0) = \mu_\beta 2^{(-\beta^{-1}+\frac{3}{2}\beta-2 P_0) P_0}\frac{\Upsilon_{\beta}\left(\frac{1}{2}(\beta+\beta^{-1})+P_0\right)}{\Upsilon_{\beta}\left(\frac{1}{2}\beta^{-1}+P_0\right)}
# $$
# and 
# $$
# \mu_\beta = 2^{\frac{7}{4}-2\beta^{-2}-\frac{1}{2}\beta^{2}}\beta^{-2+4 \beta^{-2}}\gamma(\beta^{-2})^2\left(\frac{\Upsilon_{\beta}(\beta)}{\Upsilon_{\frac{\beta}{\sqrt{2}}}(\frac{\beta}{\sqrt{2}})}\right)^2 \frac{1}{\Upsilon_{\beta}(\frac{1}{2}\beta)}
# $$

# In[ ]:


""" We choose values for the parameters beta, P, P0. """
beta = 0.3
P0 = 0.19
P = 0.51 + 0.19*1j

""" We compute the ratio of structure constants. """
betap = beta / cmath.sqrt(2)
p = 1.0/4/betap
constants = ( (16)**(-4*P**2) * Z3([-P, P0, P], beta).upsilons.integral() 
         / Z3([-p, p, cmath.sqrt(2)*P], betap).upsilons.integral()
         / Z3([p, P0/cmath.sqrt(2), -cmath.sqrt(2)*P], betap).upsilons.integral() )

""" We compute the right-hand side of the relation. """
def U(beta, x):   
    return Upsilons(b = beta, Plist = [[1j * ((beta + beta**(-1))/2 - x), True]]).integral()
mu = ( 2**(7.0/4 - 2*beta**(-2) - beta**2/2) * beta**(-2 + 4*beta**(-2))
      * mpmath.gamma(beta**(-2))**2 / mpmath.gamma(1 - beta**(-2))**2
      * U(beta, beta)**2 / U(betap, betap)**2 / U(beta, beta/2) )
g = ( mu * 2**((-1.0/beta + 3.0/2*beta - 2*P0) * P0) 
     * U(beta, (beta + beta**(-1))/2 + P0)/U(beta, 1.0/2/beta + P0) )

""" We print the two sides. They should be equal. """
print( constants )
print( g )


# ## 3.2 Why timelike Liouville theory does not exist

# ## 3.3 Proposal for the spectrum and correlation functions

# We check that our integral on $\mathbb{R}+i\epsilon$ does not depend on $\epsilon$. 
# Numerically, this is only true provided $\epsilon$ is neither too small, nor too large.

# In[ ]:


""" We choose some parameters. """
c = 0.5645
Pi = [0.32, 0.71, .45, .22]
x = .4

""" We choose a number of values of epsilon. """
epsilons = [2, 1, 0.1, 0.01, 0.001]

""" We compute the s-channel four-point function and print the results. """
beta = float(Charge('c', c).get('beta'))
q = complex(mpmath.qfrom(m= x))
for epsilon in epsilons:
    fourpoint= Z4(t_channel = False, Ps = Pi, Ucutoff = 30, Nmax = 4, 
                  beta = beta, epsilon = epsilon).value(q)
    print(  (" epsilon = %.4f, integral = %.5f + i %.5f " %(epsilon, fourpoint.real, fourpoint.imag)) ) 


# We plot the behaviour of the integrand of the four-point function,
# $$
# \text{Integrand} = C^{c\leq 1}(P_1,P_2,P_s) C^{c\leq 1}(-P_s,P_3,P_4)\, 
# \Big|\mathcal{F}^{(s)}_{P_s}(P_i|z_i)\Big|^2
# $$
# when the $s$-channel momentum $P_s$ varies along the integration line 
# $$
# P_s = P + i\epsilon \quad , \quad P \in \mathbb{R}
# $$
# The imaginary shift of the integration line is designed to avoid the poles of the integrands.
# We draw the positions of the poles as blue lines for the poles from the structure constants, and green lines for the poles from the conformal blocks, and label the poles as 
# $$
# P_{(m, n)} = m \frac{\beta}{2} + n \frac{\beta^{-1}}{2}
# $$
# We find peaks of the integrand near the positions of the poles.
# The absence of apparent peaks at certain poles of the conformal blocks, because the residues are too small.

# In[ ]:


""" We choose the various parameters: beta, the momentums P_1, P2, P3, P4, the cross-ratio x. """
beta = 1.103
Pi = [0.12, 0.7, 0.72, 0.95]
x = 0.4

""" We choose the cutoff Nmax for the computation of conformal blocks, the shift of the 
integration line.
"""
Nmax = 8
epsilon = 3*10**(-3)

""" We choose an interval of values of the s-channel momentum Ps, a number of points, and
maximum values of m, n for the poles which we display. """
Pmin, Pmax = 0., 1.2
Npoint = 100
n_max, m_max = 2, 2

""" We compute the integrand. """
block = Block(dims = [Dimension('P', P, Charge('beta', beta)) for P in Pi], Nmax = Nmax)
q = complex(mpmath.qfrom(m=x))
block_num = BlockNum(block, q = q)
block_num_cjg = BlockNum(block, q = q.conjugate())
Ps = [Pmin + (Pmax - Pmin)*i/Npoint + 1j * epsilon for i in range(Npoint)]
Integrand = [( Z3(beta = beta,  Ps = [Pi[0], Pi[1], P]).upsilons.integral()
              * Z3(beta = beta,  Ps = [Pi[2], Pi[3], -P]).upsilons.integral()
              * block_num.value(P**2) * block_num_cjg.value(P**2) ).real for P in Ps]

""" The rest is plotting, and saving the figure. """
plt.figure('Integrand of a four-point function')
plt.plot(Ps, Integrand, color ='red')
for m in range(-m_max, m_max + 1):
    for n in range(-n_max, n_max + 1):
        if m != 0 and n != 0:
            pole = m * beta / 2 + n / beta / 2
            pole_rel = (pole - Pmin) / (Pmax - Pmin)
            if 0 < pole_rel < 1:
                plt.axvline(x = pole, color = 'green' if m * n < 0 else 'blue', label = 'test')
                plt.annotate('(%.d, %.d)'%(m, n), xy = (pole_rel + .003, .015), xycoords = 'axes fraction')
plt.ylabel('Integrand')
plt.xlabel('$P$')
# plt.savefig('figure_integrand.pgf', bbox_inches = 'tight', pad_inches = 0)
plt.show()


# We now test crossing symmetry in Liouville theory with $c\leq 1$.

# In[ ]:


""" We choose several values of c, the four momentums, and the cross-ratio x. We also choose Nmax. """
c3s = [0.873, 0.564, 0.241, -1.237, -3.751]
P3s = [0.32, 0.71, .45, .22]
x3 = 0.27
Nmax3 = 6

""" We compute the four-point functions in both channels. """
beta3s = [float(Charge('c', c).get('beta')) for c in c3s]
data3s = [Data(theory = 'Analytic', beta = beta, Ps = P3s, x = [x3], Nmax = Nmax3, Pcutoff = 3) 
          for beta in beta3s]


# In[ ]:


""" We display the results. """
table3 = Table(data3s, parameter_values = c3s, parameter = 'c', digits = 5)
table3.display()

""" For use in the article, we print the Latex source of the table, without computation times. """
print( Table(data3s, parameter_values = c3s, parameter = 'c', digits = 5, time = False).tables[0][0] )


# We plot the $x$-dependence of a $c\leq 1$ Liouville four-point correlation function.

# In[ ]:


""" Physical parameters. """
Ps = [0.18, -0.2, 0.71, -0.43]
beta = 1.103

""" Number of spline points, and Nmax. """
spline = 300
Nmax = 2

""" Number and positions of points in the plot. """
N = 40
xs = [i*1.0/N for i in range(1, N)]

""" We compute the data. """
data4 = Data(theory = "Analytic", spline = spline, x = xs, beta = beta, 
             Nmax = Nmax, Ps = Ps, reduced = False)


# In[ ]:


""" We display the data as a graph. """
graph = Graph(data4)
# graph.show(save = 'figure_fourpoint.pgf')
graph.show()


# # 4 Non-analytic theories at rational values of c

# ## 4.1 Non-analytic Liouville theory

# We plot the three-point structure constant of non-analytic Liouville theory $C^\text{non-analytic}$ as a function of the momentum.
# This structure constant is the product of the smooth structure constant $C^{c\leq 1}$ (which we also plot) with the factor $\sigma\in \{0,1\}$. 
# The smooth structure constant has poles, which we transform into peaks by shifting the values of the momentum slightly away from the real line.
# So we actually plot
# $$
# \Re \left(C^{c\leq 1}(P_1, P_2, P + i\epsilon)\right) \sigma(P_1, P_2, P)
# $$
# as a function of $P\in\mathbb{R}$.
# We also indicate the positions of the poles $P\in \frac{1}{2\sqrt{pq}}$ as vertical lines. 
# We can check that $C^\text{non-analytic}$ always vanishes at these points.

# In[ ]:


""" Physical variables. """
p, q = 7, 5
P1, P2 = 0.2, 0.45

""" Shift. """
epsilon = .01

""" Interval of values of P, and number of values. """
Pmin, Pmax = 0.4, 1.2
Npoints = 500

""" Some preliminary computations, including the intervals where sigma is one. """
c = Charge('minimal', [p, q]).get('c')
beta = Charge('c', c + 10**(-11)).get('beta')
period = 1.0 / math.sqrt(p * q)
intervals = R4.extend([R4.interval(P1 + period / 2, P2, period)], period, [Pmin, Pmax])

""" Computing the three-point function. """
Ps = [Pmin + (Pmax - Pmin)*i / Npoints for i in range (Npoints)]
C_analytic = [Z3(beta = beta, Ps = [P1, P2, P + 1j*epsilon]).upsilons.integral().real for P in Ps]
C_non_analytic = []
for index, P in enumerate(Ps):
    sigma = 0
    for interval in  intervals:
        if interval[0] < P < interval[1]:
            sigma = 1    
    C_non_analytic.append(C_analytic[index] * sigma)

""" The rest is plotting. """
plt.figure('Structure constants of (non-)analytic Liouville theory')
plt.plot(Ps, C_analytic, color ='blue', label = 'analytic')
plt.plot(Ps, C_non_analytic, color ='red', label = 'non-analytic')
for n in range(int(Pmin * 2.0 / period) + 1, int(Pmax * 2.0 / period) + 1):
    pole = n * period / 2
    plt.axvline(pole, color = 'green')
# plt.title(r'Structure constants as functions of the momentum, with $\beta^2 = \frac{%d}{%d}$' %(p,q))
plt.ylabel('structure constants')
plt.xlabel('momentum')
plt.legend(loc='upper right')
# plt.savefig('figure_non_analytic.pgf', bbox_inches = 'tight', pad_inches = 0)
plt.show()  


# We directly check crossing symmetry in non-analytic Liouville theory.

# In[ ]:


""" We choose several values of c, the four momentums, and the cross-ratio x. We also choose Nmax. """
pqs = [[1, 4], [3, 5], [2, 3], [3, 4], [1, 1]] 
P5s = [0.22, 0.37, 0.28, 0.12]
x5 = 0.23
Nmax5 = 4

""" We compute the four-point functions in both channels. """
charges = [Charge('minimal', pair) for pair in pqs]
data5s = []
for charge in charges:
    data5s.append(Data(theory = 'Rational', charge = charge, Ps = P5s, x = [x5], Nmax = Nmax5))


# In[ ]:


""" We display the results. """
c5s = [latex(charge.get('c')) for charge in charges]
table5 = Table(data5s, parameter_values = c5s, parameter = 'c', digits = 7)
table5.display()

""" For use in the article, we print the Latex source of the table, without computation times. """
print( Table(data5s, parameter_values = c5s, parameter = 'c', digits = 7, time = False).tables[0][0] )


# ## 4.2 Other plausible theories

# We dismiss the non-analytic theories which are built from pairs of integers $(p, q)$ that are not coprime.
# We use values of the parameters for which crossing symmetry was achieved to a precision of the order $10^{-5}$ or better in non-analytic Liouville theory.

# In[ ]:


""" We choose several pairs of integers, the four momentums, and the cross-ratio x. We also choose Nmax. """
pqs_bad = [[2, 2], [2, 4], [3, 3]] 
P6s = [0.22, 0.37, 0.28, 0.12]
x6 = 0.23
Nmax6 = 4

""" We compute the four-point functions in both channels. """
charges = [Charge('minimal', pair) for pair in pqs_bad]
data6s = [Data(theory = 'Rational', charge = charge, Ps = P6s, x = [x6], Nmax = Nmax6) 
          for charge in charges]

""" We display the results. """
c6s = [latex((pq[0],pq[1])) for pq in pqs_bad]
table6 = Table(data6s, parameter_values = c6s, parameter = '(p, q)', digits = 7)
table6.display()


# We study an integral over disjoint unions as a function of a momentum.
# We see that the $s$- and $t$-channel agree or not depending on the momentum.

# In[ ]:


""" Choosing parameters. """
p, q = 1, 1
P7s = [0.22, 0.47, 0.31]
Npoints = 200
lastP = [i / (Npoints + .01) for i in range (1, Npoints)]
x7 = 0.23
Nmax7 = 4

""" Computing the four-point functions. """
charge = Charge('minimal', [1, 1])
data7s = [Data(theory = 'Rational', disjoint = True, 
               charge = charge, Ps = P7s + [P], x = [x7], Nmax = Nmax7)  for P in lastP]


# In[ ]:


# Table(data7s, parameter_values = lastP, parameter = 'P_4', digits = 6).display()

""" Retrieving four-point functions from the Data object. """
s_values = [data.results[0][0][0].real for data in data7s]
t_values = [data.results[0][0][1].real for data in data7s]

""" The rest is plotting. """
plt.figure('Disjoint four-point function')
plt.plot(lastP, s_values, color ='blue', label = '$s$-channel')
plt.plot(lastP, t_values, color ='red', label = '$t$-channel')
plt.ylabel('four-point integrals')
plt.xlabel('momentum $P_4$')
axes = plt.gca()
axes.set_ylim([-2, 2])
plt.legend(loc = 'upper left')
# plt.savefig('figure_disjoint.pgf', bbox_inches = 'tight', pad_inches = 0)
plt.show()  


# # Conclusion

# We test crossing symmetry in generalized minimal models. 
# In this case $N_\text{max}$ often has to be quite large for a good agreement to be found. 
# This is not problematic, because the calculations are quite fast, since we have an $s$-channel sum instead of an integral.

# In[ ]:


""" We choose values of c, four degenerate fields, the cross-ratio x, and Nmax. """
c8s = [.3 + .8 * 1j, -.5 + .7 * 1j, 3 - .1 * 1j, 28.12, .789, -4.56]
pq8s = [[2, 3], [3, 2], [4, 2], [3, 3]]
# pq8s = [[1, 2]] * 4 
x8 = 0.23
Nmax8 = 16
# The case c = 28.12 with pqs = [[2, 3], [3, 2], [4, 2], [3, 3]] is problematic for unknown
# reasons. Maybe the involved numbers are too large or small, and precision is lost.

""" We compute the four-point functions in both channels. """
data8s = []
for c8 in c8s:
    charge = Charge('c', c8)
    fields = [Dimension('degenerate', pair, charge) for pair in pq8s]
    data8s.append(Data(theory = 'GMM', fields = fields, x = [x8], Nmax = Nmax8))


# In[ ]:


""" We display the results. """
table8 = Table(data8s, parameter_values = c8s, parameter = 'c', digits = 7)
table8.display()


# In[ ]:




