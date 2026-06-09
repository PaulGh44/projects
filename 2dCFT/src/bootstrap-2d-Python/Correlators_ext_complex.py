#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division, print_function

import numpy, math, cmath
np = numpy
import mpmath, scipy.special
import operator
from cmath import sin, sinh, exp
from scipy.interpolate import InterpolatedUnivariateSpline


from Auxiliary_classes import complex_quadrature, numerical
from CFT import Charge, Dimension
from Blocks import Block, BlockNum 


# # Upsilon functions

# The Upsilon function $\Upsilon_b(P)$ is defined as 
# $$
# \Upsilon_b(P) = \exp \int_0^\infty \frac{dt}{t} \left[\frac{\sin^2 Pt}{\sinh bt\sinh\frac{t}{b}} - P^2 e^{-2t}\right]
# $$
# which is valid if $|\Im P |\leq \frac{Q}{2}$ with $Q = b + \frac{1}{b}$. For momentums outside this strip, we should use shift equations such as
# $$
# \frac{\Upsilon_b(P-ib)}{\Upsilon_b(P)} = b^{-b^2-2ibP}\gamma(\tfrac{bQ}{2} + ibP)
# $$
# In the case $b=1$ we have 
# $$
# \Upsilon_1(iP) = G(1+P)G(1-P)
# $$
# where we use Barnes' $G$-function.
# 
# There is an alternative representation as the infinite product
# $$
# \Upsilon_b(P) = e^{\lambda_b P^2} \prod_{m,n=0}^\infty F\left(\frac{P^2}{\left(\frac{Q}{2}+mb+nb^{-1}\right)^2}\right) \qquad \mathrm{where} \qquad F(x) = (1+x)\exp(-x)
# $$
# where we define
# $$
# \lambda_b = \int_0^\infty \frac{dt}{t} \left[\frac{t^2}{\sinh bt\sinh\frac{t}{b}} - e^{-2t}\right]
# $$
# The factor $e^{\lambda_b P^2}$ plays no role in most calculations. We neglect it by default, while providing
# a static method for computing it. We have $\lambda_1 = 1 + \gamma$ where $\gamma$ is the Euler-Mascheroni constant.
# 
# The product representation can be improved by approximating the neglected factors with (the exponential of) an integral, according to the formula
# $$
# \Upsilon_b(P) \sim \prod_{m=0}^M\prod_{n=0}^N \left(1+\frac{P^2}{y_{m,n}^2}\right) 
# \times
# e^{ - \sum_{m=0}^M\sum_{n=0}^N \frac{P^2}{y_{m,n}^2} - Pb\sum_{m=0}^M \varphi'\left(\frac{y_{m,N+\frac12}}{P}\right) - Pb^{-1}\sum_{n=0}^N\varphi'\left(\frac{y_{M+\frac12,n}}{P}\right) + P^2 \varphi\left(\frac{y_{M+\frac12,N+\frac12}}{P}\right) }
# $$
# where we define $y_{m,n} = \frac{Q}{2}+mb+nb^{-1}$ and 
# $$
# \varphi(y) = \tfrac32 + \tfrac12 (y^2-1)\log(1+\tfrac{1}{y^2}) + iy\log\tfrac{y+i}{y-i}
# \\
#  \varphi'(y) = y\log(1+\tfrac{1}{y^2}) +\tfrac{1}{y} +i\log\tfrac{y+i}{y-i}
# $$

# In[ ]:


class Upsilons:
    """ Computing products and ratios of Upsilon functions. """

    def __init__(self, b = 1, Plist = [[1, True]]):
        """ We need the parameter b, and lists of Ps for numerator 
        (True) or denominator (False) Upsilon functions. """ 
        self.b = b
        self.Plist = Plist

    @staticmethod
    def phi(y):
        return 3/2 + (y**2 - 1)/2 * cmath.log(1 + 1/y**2) + 1j*y*cmath.log((y+1j)/(y-1j))

    @staticmethod
    def phip(y):
        return y * cmath.log(1 + 1/y**2) + 1 / y + 1j*cmath.log((y+1j)/(y-1j))

    def product(self, L = 20, improved = True, normalized = False):
        """ Computes the product, where m and n are bounded by integer cutoffs M and N, chosen so that 
        we compute O(L^2) factors and so that Mb = Nb^{-1}.
        If improved, includes the approximation of the neglected factors. 
        If normalized, agrees with the integral representation.
        """

        b = self.b
        binv = 1 / b
        q = ( b + binv ) / 2      

        M = int(L/abs(b)) + 1
        N = int(L*abs(b)) + 1

        Plist = [pair for pair in self.Plist if pair[0] != 0]  
        """ We remove the zeros from the list, as they give trivial contributions and nevertheless
        cause errors if improved. """
        Plists = [[pair[0]**2, pair[1]] for pair in Plist]

        prod = 1
        exponent = 0

        for m in range(M + 1):
            for n in range(N + 1):
                y = q + m*b + n*binv
                x = 1 / y**2
                exponent += x
                for pair in Plists:
                    factor = 1 + pair[0] * x
                    prod = prod * factor if pair[1] else prod / factor

        expFactor = 0
        for pair in Plists:
            expFactor = expFactor + pair[0] if pair[1] else expFactor - pair[0]
        exponent *= expFactor

        if improved:

            addterm = 0
            for m in range(M + 1):
                y = q + m*b + (N+1/2)*binv
                for pair in Plist:
                    P = pair[0]
                    term = P * Upsilons.phip(y / P)
                    addterm = addterm + term if pair[1] else addterm - term
            addterm *= b 
            exponent += addterm

            addterm = 0
            for n in range(N + 1):
                y = q + (M+1/2)*b + n*binv
                for pair in Plist:
                    P = pair[0]
                    term = P * Upsilons.phip(y / P)
                    addterm = addterm + term if pair[1] else addterm - term
            addterm *= binv
            exponent += addterm

            y = q + (M+1/2)*b + (N+1/2)*binv
            for pair in Plist:
                term = Upsilons.phi(y / pair[0])
                exponent = exponent - term if pair[1] else exponent + term

        if normalized:
            shift = 0
            for pair in Plists:
                shift = shift + pair[0] if pair[1] else shift - pair[0]
            exponent -= shift * Upsilons.lambda_b(b)

        return prod * cmath.exp(- exponent)

    @staticmethod
    def lambda_b(b, cutoff = 10):
        """ Computes the number lambda_b which controls 
        the discrepancy between the product and integral representations. """
        def integrand(t):
            return (t**2 / cmath.sinh(b*t) / cmath.sinh(t/b) - cmath.exp(-2*t))/t  
        return complex_quadrature(integrand, 0, cutoff)[0]    

    def integral(self, cutoff = 100):
        """ Directly computes the integral over t. Uses recursion relations if need be. """
        result = 1
        for pair in self.Plist:
            factor = Upsilons.upsilon(pair[0], self.b)
            result = result * factor if pair[1] else result / factor
        return result      

    @staticmethod 
    def upsilon(P, b, cutoff = 100):
        """ Computes the Upsilon function, using recursion relations if need be. """
        bs = [b, -b, 1/b, -1/b]
        for oneb in bs:
            if oneb.real > b.real:
                b = oneb
        Q = b + 1/b
        q = Q.real/2
        alpha = Q/2 + 1j*P

        def integrand(t):
            return ((cmath.sin(P*t))**2 / cmath.sinh(b*t) / cmath.sinh(t/b) - P**2*cmath.exp(-2*t))/t  

        if P.imag >= q:
            return numerical(Upsilons.upsilon(P - 1j*b, b, cutoff) * mpmath.gamma(1-b*alpha) 
                           / mpmath.gamma(b*alpha) * b**(2*b*alpha - 1))
        elif P.imag <= -q:
            return numerical(Upsilons.upsilon(P + 1j*b, b, cutoff) * mpmath.gamma(b*(alpha-b))
                           / mpmath.gamma(1-b*(alpha-b)) * b**(1-2*b*(alpha - b)))
        else:    
            return cmath.exp(complex_quadrature(integrand, 0, cutoff)[0]) 


from mpmath import gamma
class Upsilonprime:
    """ calcuates 1 / derivatives of the upsilon function
    at alpha = - m b - n / b, as function of alpha"""
    def __init__(self, b, mmax = 10, nmax = 10):
        self.b=b
        Q = b + 1 / b
        def smallgamma(z):
            try:
                res =  gamma(z) / gamma(1 - z)
                return numerical(res)
            except ValueError:
                return 0.j
        upsb = Upsilons(b, [[(b - Q/2) / 1.j, True]]).integral()
        #print( upsb )
        table = np.zeros((mmax, nmax), dtype = complex)
        table[0,0] = 1.  /  upsb
        for j in range(0, mmax):
            x = -(j * b)
            if j > 0:
                table[j,0] = numerical(table[j - 1, 0] * (b ** (1 - 2 * b * x)) * smallgamma(b * x))
            for k in range(1, nmax):
                x = - k / b - j * b
                table[j,k] = numerical(table[j, k - 1] * (b ** (2 * x / b - 1)) * smallgamma(x / b))
        self.table = table



# # Liouville theory

# ## Reflection coefficient

# The reflection coefficient associated to a field is 
# $$
# R = - \frac{\Gamma(b(2\alpha-Q)) \Gamma(b^{-1}(2\alpha-Q))}{\Gamma(b(Q-2\alpha)) \Gamma(b^{-1}(Q-2\alpha))}
# $$
# 

# In[ ]:


class Reflection:
    """ Liouville reflection coefficient. """

    def __init__(self, field):

        alpha = field.get('alpha')
        b = field.charge.get('b')
        x = 2*alpha - b - 1/b
        self.value = - ( scipy.special.gamma(b*x) * scipy.special.gamma(x/b) 
                    / scipy.special.gamma(-b*x) / scipy.special.gamma(-x/b) )


# ## Three-point correlation function

# The Liouville three-point function is given by the DOZZ formula,
# $$
# C(P_1, P_2, P_3) = \frac{\prod_{i=1}^3 \Upsilon_b(2P_i -i\frac{Q}{2})}{\prod_{\pm\pm}\Upsilon_b(P_1\pm P_2 \pm P_3)}
# $$

# In[ ]:


class ThreePoint:
    """ Liouville three-point function. """

    def __init__(self, Ps, b, smart = -1):
        """ We encode the three-point function as an Upsilons object.
            when smart = 0,1,2, only the Ps[smart] dependent factors will be calcuated. """
        Q = b + 1/b
        Plist = ([[2*P - 1j * Q/2, True] for P in ([Ps[smart]] if smart in [0,1,2] else Ps)] 
                 + [[P, False] for P in Z3.Psums(Ps)])
        self.upsilons = Upsilons(b, Plist)
class ThreePointInt:
    """ Liouville three-point function, without the two external Upsilon(2alpha) 's """

    def __init__(self, Ps, b):
        """ We encode the three-point function as an Upsilons object.
            when smart = 0,1,2, only the Ps[smart] dependent factors will be calcuated. """
        Q = b + 1/b
        Plist = ([[2*Ps[2] - 1j * Q/2, True]] 
                 + [[P, False] for P in Z3.Psums(Ps)])
        self.upsilons = Upsilons(b, Plist)


# In[ ]:





# ## Four-point correlation function

# We consider four fields, parametrized by real momentums $P_1,P_2,P_3,P_4$. We want to compute the corresponding four-point function $Z$ in Liouville theory, by using its decomposition in some channel,
# $$
# Z = \prod_{i=1}^4 \frac{\Upsilon_b(2P_i-i\frac{Q}{2})}{\Upsilon_b(2P_i)} \int_0^\infty dP\ K(P) |B(P)|^2
# $$
# where $K(P)$ is a combination of structure constants, $B(P)$ is a conformal block, where complex conjugation only acts on the position $x$. In the case of the $s$-channel, we have
# $$
# K_s(P) = P^2 \exp \int_0^\infty \frac{dt}{t}\frac{4\varphi(t)}{\sinh bt\sinh b^{-1}t }
# $$
# where we used
# $$
# \varphi(t) = 4S(1-S)\sinh^2 \tfrac12(b-\tfrac{1}{b})t-2S^2
# +S\left(2\sum_{i=1}^4S_i-4S_1S_2-4S_3S_4\right)-(S_1-S_2)^2-(S_3-S_4)^2
# $$
# where $S_i=\sin^2P_it$ and $S=\sin^2Pt$.

# In[ ]:


class FourPoint:
    """ Liouville four-point function. <\prod_i V_{Q/2 + ip_i}(z_i)>, z_i = (0, x, 1, \infty)"""


    def __init__(self, t_channel = False, Ps = [i + .5 for i in range(4)], b = math.sqrt(2) - 1j,
                 Nmax = 4, Pcutoff = 10, Kcutoff = 40,
                 reduced = False, spline = 0, normalized = False, useK2 = 0):
        """ We need the values of b, four momentums P, the order Nmax of the blocks' expansion, 
        two integration cutoffs, and the indication whether to use reduced blocks. 
        We also give the number of points in the interpolating spline
        for K(P): if this number is zero, then no interpolation is used. The interpolating
        spline dispenses us from computing K(P) again whenever we change N or q. For precision computations
        the spline should be about 2000. For graphical display, much lower values such as 100 may be enough.
        There is an option to include the normalization prefactor, which is absent by default.
        useK2: 0 - use K(P), -1: use K2(P) with prefactor, 2:use K2(P) without prefactor.
        the K(P) trick works only if |imag(P)| < Q / 4, so will switch to K2 automatically.
        """
        useK2 = 0
        for P in Ps:
            if abs(P.imag) > (b + 1 / b).real / 4:
                useK2 = 2
                break 
        print( "use K2", useK2 )
        self.useK2 = useK2
        self.b = b
        self.t_channel = t_channel
        self.Kcutoff = Kcutoff
        self.Pcutoff = Pcutoff
        self.charge = Charge('b', b)        
        self.Ps = Ps
        self.computeBlock(Nmax)
        self.Nmax = Nmax
        self.reduced = reduced
        self.prefactor = 1.
        self.normalized = normalized
        self.upsprime = Upsilonprime(b, 5, 5)
        if normalized and useK2 in [0, 2]:
            if useK2 == 0:
                Plist = [[2*P - 1j * (b+1/b)/2, True] for P in Ps] + [[2*P, False] for P in Ps]
            if useK2 == 2:
                Plist = [[2*P - 1j * (b+1/b)/2, True] for P in Ps]
            self.upsilons = Upsilons(b, Plist)
            self.prefactor = self.upsilons.integral()
            if useK2 == 0:
                self.prefactor *= (self.upsprime.table[0,0] ** (-2)) * 4
        self.spline = spline
        self.useSpline = spline > 0            
        if self.useSpline:
            self.Kspline = FourPoint.complex_spline([i*Pcutoff/spline for i in range(spline+1)], 
                                                    [self.K, self.K2, self.K21][self.useK2])
        self.polelist = self.getcrossedpoles()
        self.CCdiscrete = self.getCCdiscrete(self.polelist) 
    def computeBlock(self, N):
        """ Computing the relevant Block. This method is here so that we can change N without
        reconstructing the object.
        """       
        self.block = Block([Dimension('P', self.Ps[i], self.charge) for i in range(4)], N, self.t_channel)

    @staticmethod
    def complex_spline(Pvals, K):
        """ Computing spline functions for real and imaginary parts of a function K
        on argument values Pvals. """
        Kvals = [K(Pval) for Pval in Pvals]
        Kreals = [[Kval.real for Kval in Kvals], [Kval.imag for Kval in Kvals]]
        return [InterpolatedUnivariateSpline(Pvals, Kreals[0]), 
                InterpolatedUnivariateSpline(Pvals, Kreals[1])]
    def getallpoles(self, nmax = 10, mmax = 10):
        """We look for all poles in Ps of the structure constant """
        Ps = self.Ps
        if self.t_channel:
            Ps = [Ps[1], Ps[2], Ps[3], Ps[0]]
        b = self.b
        Q = b + 1. / b
        R = np.array([[b * k + j / b for k in range(nmax)] for j in range(mmax)])
        Poles = np.zeros((16, nmax, mmax), dtype = complex)
        for series in range(16):
            PA, PB = (Ps[0], Ps[1]) if series < 8 else (Ps[2], Ps[3])
            signs = [1,1,1]
            s1 = series
            for k in range(3):
                signs[k] = -1 if s1 % 2 == 1 else 1
                s1 = s1 // 2
            Poles[series, :, :] = (R + Q / 2) * 1.j * signs[0] + PA * signs[1] + PB * signs[2] 
        return Poles

    def getcrossedpoles(self):
        """return the poles that have crossed the spectrum. Return a list of (p, k, j, series), 
           where p is the p-position of the pole, k, j are the coordinate k b + j / b in the lattice
           and series gives the mupsilon that produces the pole. """
        res = []
        Ps = self.Ps
        if self.t_channel:
            Ps = [Ps[1], Ps[2], Ps[3], Ps[0]]
        b = self.b
        Q = b + 1. / b
        for series in range(16):
            #the lowest bit is \pm P_int, the next two are \pm P_1, P_2, /comp 
            PA, PB = (Ps[0], Ps[1]) if series < 8 else (Ps[2], Ps[3])
            signs = [1,1,1]
            s1 = series
            for k in range(3):
                signs[k] = -1 if s1 % 2 == 1 else 1
                s1 = s1 // 2
            leading = (Q / 2) * 1.j * signs[0] + PA * signs[1] + PB * signs[2]
            borne = -leading.imag * signs[0] 
            if borne < 0: 
                continue
            else: 
                print( "crossing in series", series, 'signs:', signs )
            for j in range(int(borne / b.real) + 1):
                jb = borne - j * b
                for k in  range(int(jb.imag * b.real) + 1):
                    pos = (j * b + k / b + Q / 2) * 1.j * signs[0] + signs[1] * PA + signs[2] * PB
                    res.append((pos, j, k, series))
        return res
    def getCCdiscrete(self, polelist):
        """calculate the product of structure constants at the crossed poles."""
        Ps = self.Ps
        b = self.b
        Q = b + 1. / b
        if self.t_channel:
            Ps = [Ps[1], Ps[2], Ps[3], Ps[0]]
        res = []
        for (pos, j, k, series) in polelist:
            Plists = [[pos * 2 + Q / 2.j, True], [- pos * 2 + Q / 2.j, True]]
            signs = [[1, 1],[-1, 1],[1, -1],[-1, -1]]
            for s in range(8):
                if s ==  series // 2:
                    continue
                (Pa, Pb) = (Ps[0], Ps[1]) if s < 4 else (Ps[2], Ps[3])
                newp = -Pa * signs[s % 4][0] - Pb * signs[s % 4][1] + pos
                Plists.append([newp, False])
            prod = Upsilons(b, Plists).integral()
            res.append(2 * np.pi * self.upsprime.table[j,k] * prod)
        return res 
    def K(self, P, *args):
        """ We compute the combination of structure constants. """
        Ps = self.Ps
        if self.t_channel:
            Ps = [Ps[1], Ps[2], Ps[3], Ps[0]]

        def integrand(t):
            Ss = [sin(Ps[i]*t)**2 for i in range(4)]
            S = sin(P*t)**2
            sh = sinh((self.b - 1/self.b)/2*t)**2
            phi = (4*S*(1-S)*sh - 2*S**2 + S*(2*(Ss[0]+Ss[1]+Ss[2]+Ss[3]) 
                                        - 4*Ss[0]*Ss[1] - 4*Ss[2]*Ss[3]) - (Ss[0]-Ss[1])**2 - (Ss[2]-Ss[3])**2)
            return 4*phi/t/sinh(self.b*t)/sinh(t/self.b)

        return P**2*exp(complex_quadrature(integrand, 0, self.Kcutoff)[0])
    def K2(self, P):
        Ps = self.Ps
        if self.t_channel:
            Ps = [Ps[1], Ps[2], Ps[3], Ps[0]]
        return (ThreePoint([Ps[0], Ps[1], P], self.b).upsilons.integral() * 
                ThreePoint([Ps[2], Ps[3], -P], self.b).upsilons.integral())
    def K21(self, P):
        Ps = self.Ps
        if self.t_channel:
            Ps = [Ps[1], Ps[2], Ps[3], Ps[0]]
        return (ThreePointInt([Ps[0], Ps[1], P], self.b).upsilons.integral() * 
                ThreePointInt([Ps[2], Ps[3], -P], self.b).upsilons.integral())

    def value(self, q = .05):
        """ Computes the numerical value of the four-point function. The integrand decreases quickly at large
        P so the cutoff can be quite small. Anyway, computing K(P >= 12) is problematic. (Warning about number
        of subdivisions from integration over t.) However this value of Pcutoff might have to depend on the 
        parameters b and Ps. We take an asymmetric range (0, Pcutoff) because the integrand is an even 
        function of P.   
        Discrete fusion contritbutions added. 
        """

        blockNum = BlockNum(self.block, q)
        blockNum_c = BlockNum(self.block, q.conjugate())
        if self.useSpline:
            kfunc = lambda P: self.Kspline[0](P) + self.Kspline[1](P)*1j
        else: 
            kfunc = [self.K, self.K2, self.K21][self.useK2]
        integrand = lambda P: numerical(kfunc(P) * blockNum.value(P**2, self.reduced) 
                                        * blockNum_c.value(P**2, self.reduced))
        """ We apply the method 'numerical' because 'mpmath' can produce objects of the 
        'mpc' and 'mpf' classes which are not recognized as complex numbers and floats 
        respectively by the 'quad' integration method.
        """
        res = complex_quadrature(integrand, 0, self.Pcutoff)[0]

        for i in range(len(self.polelist)):
            pos = self.polelist[i][0]
            res += (.5 *  self.CCdiscrete[i] * blockNum.value(pos**2, self.reduced) 
                       * blockNum_c.value(pos**2, self.reduced))
            # .5 because we integrate over P = 0 to +infty, so we take the half of the residue    
        if self.normalized:
            res *= self.prefactor

        return res
    def value_discrete(self, q = .05):
        """return the list of discrete contributions to 4p function."""
        blockNum = BlockNum(self.block, q)
        blockNum_c = BlockNum(self.block, q.conjugate())
        res = [] 
        for i in range(len(self.polelist)):
            pos = self.polelist[i][0]
            contribution = (self.CCdiscrete[i] * blockNum.value(pos**2, self.reduced)
                            * blockNum_c.value(pos**2, self.reduced))
            if self.normalized:
                contribution *= self.prefactor
            res.append(contribution)
        return res


# # Generalized minimal models

# ## Three-point function

# The three-point structure constant of a generalized minimal model is given by
# $$
# C_{\{(r_i,s_i)\}} = \frac{(-1)^{r_0+s_0+1}b^{2(r_0-s_0)}}{P_{r_0}(b^2)P_{s_0}(b^{-2})Q_{r_0,s_0}(b)} 
# \prod_{i=1}^3 \frac{P_{r_i-1}(b^2)P_{s_i-1}(b^{-2}) Q_{r_i-1,s_i-1}(b)}{P_{r_0-r_i}(b^2)P_{s_0-s_i}(b^{-2})Q_{r_0-r_i,s_0-s_i}(b)}
# $$
# where we introduced the integers
# $$
# r_0 = \frac{r_1+r_2+r_3-1}{2} \quad , \quad s_0 = \frac{s_1+s_2+s_3-1}{2}
# $$
# and the functions 
# $$
# P_r(x) = \prod_{i=1}^r \gamma(1+ix) \quad ,\quad Q_{r,s}(b) = \prod_{i=1}^r \prod_{j=1}^s(ib+jb^{-1})^2
# $$
# where $\gamma(x) =\frac{\Gamma(x)}{\Gamma(1-x)}$.
# 
# The conditions for the three fields to be intertwined by fusion are 
# $$
# r_0, s_0 \in \mathbb{N} \quad , \quad \left\{\begin{array}{l} r_i \leq r_0 \\ s_i \leq s_0 \end{array} \right.
# $$

# In[ ]:


class Parray:
    """ An array of values of P_r(x) for various values of r. """

    def __init__(self, x):
        """ We need only specify x. """ 

        self.x = x
        self.values = [1]

    def get(self, r):
        """ Given an integer r, lengthens the lists of values if appropriate, and returns
        the desired value. """

        value = self.values[-1]

        for i in range(len(self.values), r+1): 
            value *= scipy.special.gamma(1+i*self.x) / scipy.special.gamma(-i*self.x)
            self.values.append(value)

        return self.values[r] 

class Qarray:
    """ An array of values of Q_{r,s}(x) for various values of r, s. """

    def __init__(self, x):
        """ We need only specify x. """

        self.x = x
        self.values = [[1]]

    def get(self, r, s):
        """ Given r and s, lengthens the list of values as appropriate, and returns the 
        desired value. """

        for i in range(len(self.values)):
            line = self.values[i]
            value = line[-1]
            for j in range(len(line), s+1):
                for k in range(1, i+1):
                    value *= (k*self.x + j/self.x)**2
                line.append(value)

        line = list(self.values[-1])    

        for i in range(len(self.values), r+1):
            for j in range(len(line)):
                for k in range(1, j+1):
                    line[j] *= (i*self.x + k/self.x)**2
            self.values.append(list(line))

        return self.values[r][s]

class GMM3:
    """ Three degenerate fields, and the corresponding three-point structure constant. 
    We also determine whether the fields are intertwined by fusion. That r0 and s0 are 
    integer is not explicitly tested. 
    """

    def __init__(self, fields, minimal = False, forced = True):
        """ We need an array of three degenerate fields. We should also indicate 
        whether we really want to compute the three-point function if the fields 
        are not intertwined by fusion. And we have the option to normalize as in 
        minimal models.
        """

        pairs = [field.get('degenerate') for field in fields]
        r0 = 0
        s0 = 0
        for pair in pairs:
            (r, s) = pair
            r0 += r
            s0 += s
        r0 = (r0 - 1)//2
        s0 = (s0 - 1)//2

        I = True
        for pair in pairs:
            (r, s) = pair
            I = I and r <= r0 and s <= s0
        self.canFuse = I            

        if I or forced:

            charge = fields[0].charge
            b = charge.get('b')
            Ps = Parray(b**2)
            Pinvs = Parray(1/b**2)
            Qs = Qarray(b)

            C = (-1)**(r0 + s0 + 1) * b**(2*(r0-s0)) / Ps.get(r0) / Pinvs.get(s0) / Qs.get(r0, s0)
            for pair in pairs:
                (r, s) = pair
                C *= Ps.get(r-1) * Pinvs.get(s-1) * Qs.get(r-1, s-1)
                C /= Ps.get(r0-r) * Pinvs.get(s0-s) * Qs.get(r0-r, s0-s)
            if minimal:
                C *= cmath.sqrt( Reflection(Dimension('degenerate', [1, 1], charge)).value )
                for field in fields:
                    C /= cmath.sqrt( Reflection(field).value ) 

            self.value = numerical(C)



# ## Four-point function

# In[ ]:


class GMM4:
    """ Computing the four-point function. """

    def __init__(self, fields, t_channel = False,  Nmax = 4, reduced = False):
        """ We need an array of four degenerate fields. """

        self.t_channel = t_channel
        self.original_fields = fields
        self.Nmax = Nmax
        self.computeBlock(Nmax)
        self.fields = [fields[1], fields[2], fields[3], fields[0]] if t_channel else fields
        self.reduced = reduced

        prop = []
        for field in self.fields[0].fuse(self.fields[1]):
            if GMM3([field, self.fields[2], self.fields[3]], False, False).canFuse:
                prop.append(field)
        self.propagating = prop

    def computeBlock(self, N):
        """ Computing the relevant Block. This method is here so that we can change N without
        reconstructing the object.
        """       
        self.block = Block(self.original_fields, N, self.t_channel) 

    def value(self, q = .05):
        """ The value of the four-point function. """
        total = 0
        blockNum = BlockNum(self.block, q)
        blockNum_c = BlockNum(self.block, q.conjugate())

        for field in self.propagating:

            factor = ( GMM3([field, self.fields[0], self.fields[1]]).value 
                      * GMM3([field, self.fields[2], self.fields[3]]).value / Reflection(field).value )
            delta = field.get('delta')
            total += factor * blockNum.value(delta, self.reduced) * blockNum_c.value(delta, self.reduced)

        return numerical(total)

    def getFields(self):
        """ The list of propagating fields. """
        return [field.get('degenerate') for field in self.propagating]



# # Conformal field theory with $c<1$

# ## Three-point structure constants

# We compute the three-point structure constant
# $$
# C(P_1,P_2,P_3) = 
# \frac{\prod_{\pm\pm}\Upsilon_\beta(i(P_1\pm P_2 \pm P_3))}{\prod_{i=1}^3\Upsilon_\beta(i(q'+2P_i))}
# $$
# where we defined
# $$
# \beta = ib \quad , \quad q' = \frac12(\beta^{-1}-\beta)
# $$
# In the statistical physics application, the normalized three point structure constants have the extra normalization factor $A(\beta)\prod_{i=1}^3 B(\beta, P_i) $ where 
# $$
# A(\beta)=\beta^{-1-\beta^2+\beta^{-2}}\frac{\sqrt{\gamma(\beta^{-2}-1)\gamma(\beta^2)}}{\Upsilon_{\beta}(iq')}
# $$
# $$
# B(\beta, P) = \sqrt{\frac{\Upsilon_\beta(i(q'+2P))}{\Upsilon_\beta(i(q'-2P))}} = \beta^{4\beta^{-1}P-\beta^{-2}-1}\sqrt{\gamma(4\beta^{-1}P) \gamma(4\beta^{-1}P -\beta^{-2})}
# $$

# In[ ]:


class Z3:
    """ Computing the three-point function. """

    def __init__(self, Ps, beta): 
        """ We need a list of three fields. """

        self.beta = beta
        self.Ps = Ps
        self.qp = (1/beta - beta)/ 2
        Plist = [[-1j * P, True] for P in Z3.Psums(Ps)] + [[-1j * (self.qp + 2*P), False] for P in Ps]
        self.upsilons = Upsilons(beta, Plist)

    @staticmethod
    def Psums(Ps):
        """ We compute four combinations of momenums. """
        return [Ps[0] + Ps[1] + Ps[2], Ps[0] - Ps[1] + Ps[2], Ps[0] + Ps[1] - Ps[2], - Ps[0] + Ps[1] + Ps[2]]

    def upsilons(self):
        """ We return an Upsilon object, which can then be evaluated in various ways. """
        return self.upsilons

    @staticmethod
    def overall_factor(beta):
        """ We compute the factor A(beta). """
        result = 1   
        result *= Upsilons(beta, [[1j*(1/beta - beta)/2, False]]).integral()
        result *= beta**(beta**(-2) - beta**2 -1)
        result *= cmath.sqrt(mpmath.gamma(beta**2) / mpmath.gamma(1 - beta**2) * mpmath.gamma(beta**(-2) - 1)
                             / mpmath.gamma(2 - beta**(-2)))        
        return result

    @staticmethod
    def momentum_factor(beta, P):
        """ We compute the momentum-dependent normalization. """
        (p1, p2) = (4*P/beta, 4*P/beta - 1/beta**2)
        result = beta**(p1 - 1/beta**2 - 1)
        result *= cmath.sqrt(mpmath.gamma(p1) / mpmath.gamma(1 - p1) * mpmath.gamma(p2) / mpmath.gamma(1 - p2))
        return result

    def normalized_value(self):
        """ Computes the value of the three-point function, as normalized for statistical physics
        applications.
        """
        result = self.upsilons.integral()
        result *= Z3.overall_factor(self.beta)
        for P in self.Ps:
            result *= Z3.momentum_factor(self.beta, P)
        return result


# ## Four-point functions

# We compute four-point functions with $c<1$ i.e. $\beta\in \mathbb{R}$. In the $s$-channel, this is 
# $$
# Z = \int_{\mathcal{C_\epsilon}} dP\ C(P_1, P_2, P) C(-P, P_3, P_4)\ |B_s(P)|^2
# $$
# for a contour $\mathcal{C}_\epsilon = \mathbb{R} + i\epsilon$.
# 
# By default we neglect $P_i$-dependent prefactors, and use the product formula for Upsilon functions. 
# To obtain the statistical physics normalization, we should multiply with the normalization factor
# $$
# \frac{1}{\prod_{i=1}^4 \Upsilon_\beta(i(q'+2P_i))} \cdot A(\beta)^2 \prod_{i=1}^4 B(\beta, P_i) \cdot e^{\lambda_b(6q'^2 + 8q'\sum_{i=1}^4 P_i)}
# $$
# where the first factor contains the neglected $P_i$-dependent prefactors, the second factor brings us to the statistical physics normalization, and the third factor accounts for our use of products instead of integrals for the Upsilon functions.

# In[ ]:


class Z4:
    """ Computing the four-point function. """

    def __init__(self, t_channel = False, Ps = [i + .5 for i in range(4)], beta = math.sqrt(2) - 1, 
                 Nmax = 4, Pcutoff = 10, Ucutoff = 20, reduced = False, spline = 0, epsilon = .1, 
                 normalized = False):
        """ We need the values of beta, four momentums P, the order Nmax of the blocks' expansion, 
        cutoffs for the P integration and the Upsilon product formula, and the indication whether to 
        use reduced blocks. 
        We also give the number of points in the interpolating spline: if this number is zero, then no 
        interpolation is used. For precision computations the spline should be about 2000. 
        For graphical display, much lower values such as 100 may be enough.
        We also give a parameter epsilon for the position of the integration line.
        We indicate whether the result should be normalized for statistical physics applications.
        """  

        self.beta = beta
        self.qp = (1/beta - beta)/2
        self.t_channel = t_channel
        self.Ucutoff = Ucutoff
        self.Pcutoff = Pcutoff
        self.charge = Charge('beta', beta)        
        self.Ps = Ps
        self.Nmax = Nmax
        self.computeBlock(Nmax)
        self.reduced = reduced
        self.epsilon = epsilon

        self.useSpline = spline > 0
        if self.useSpline:
            Pvals = [i*Pcutoff/spline for i in range(-spline, spline+1)]
            self.Kspline = FourPoint.complex_spline(Pvals, self.K)   

        self.normalized = normalized
        if normalized:
            upsilons = [[-1j * (self.qp + 2*P), False] for P in Ps]
            factor = Upsilons(beta, upsilons).product(Ucutoff)
            factor *= Z3.overall_factor(beta)**2
            for P in Ps:
                factor *= Z3.momentum_factor(beta, P)
            factor *= cmath.exp(Upsilons.lambda_b(beta) * (6 * self.qp**2 + 8 * self.qp * sum(Ps)))
            self.factor = factor                

    def computeBlock(self, N):
        """ Computing the relevant Block. This method is here so that we can change N without
        reconstructing the object.
        """       
        self.block = Block([Dimension('P', self.Ps[i], self.charge) for i in range(4)], N, self.t_channel)

    def K(self, p0):
        """ We compute the prefactor. We need to shift p at this point, so that we can compute the spline. """
        Ps = self.Ps
        p = self.shift(p0)
        if self.t_channel:
            Ps = [Ps[1], Ps[2], Ps[3], Ps[0]]
        Psums = Z3.Psums([Ps[0], Ps[1], p]) + Z3.Psums([Ps[2], Ps[3], -p])
        Plist = [[-1j * P, True] for P in Psums] + [[-1j * (self.qp + 2*P), False] for P in [p, -p]]
        upsilons = Upsilons(self.beta, Plist)
        # return upsilons.integral()
        return upsilons.product(self.Ucutoff)

    def shift(self, P):
        """ Shifts P away from the real line. """
        return P + 1j * self.epsilon

    def value(self, q = .05):
        """ Computes the numerical value of the four-point function. The integrand decreases quickly at large
        P so the cutoff can be quite small. 
        """

        blockNum = BlockNum(self.block, q)
        blockNum_c = BlockNum(self.block, q.conjugate())

        def integrand(P):
            if self.useSpline:
                K = self.Kspline[0](P) + self.Kspline[1](P)*1j
            else:
                K = self.K(P)
            Delta = self.shift(P)**2
            return numerical(K * blockNum.value(Delta, self.reduced) * blockNum_c.value(Delta, self.reduced))
            """ We apply the method 'numerical' because 'mpmath' can produce objects of the 
            'mpc' and 'mpf' classes which are not recognized as complex numbers and floats 
            respectively by the 'quad' integration method.
            """
        result = complex_quadrature(integrand, -self.Pcutoff, self.Pcutoff)[0]      
        if self.normalized:
            result *= self.factor
        return result


# # Rational central charge

# Let us have 
# $$
# \beta^2 = \frac{q}{p}
# $$
# We consider the three-point structure constants 
# $$
# \tilde{C}(P_1,P_2,P_3) = C(P_1, P_2, P_3) \sigma
# $$
# where $C$ is the usual, smooth three-point structure constant, and
# $$
# \sigma = 1_E =
# \left\{
# \begin{array}{l} 
# 1 \quad \text{if} \quad 
# s(P_1+P_2+P_3)s(P_1+P_2-P_3)s(P_1-P_2+P_3)s(-P_1+P_2+P_3) > 0
# \\
# 0 \quad \text{otherwise}
# \end{array}
# \right.
# $$
# with 
# $$
# s(x) = \sin \pi \left(\sqrt{pq} x + \tfrac12(p-q)\right)
# $$
# The periodic function $\sigma$ is the characteristic function of the set $E$, which is a union of intervals. In a four-point function we have a factor $\sigma\sigma' = 1_{E\cap E'}$, and we call the corresponding integral the intersection integral. We also consider the xor integral with the factor $1_{E\sqcup E'}$, which would appear if we considered three-point function of the type $C(P_1, P_2, P_3)(1-\nu + \nu \sigma)$ for some number $\nu$. 
# 
# Let us examine how we compute $\sigma$ as a function of $P_1$ in the case $p=q=1$. We call $P_-, P_+$ the two smallest numbers of the set of four numbers $\{\pm P_2 \pm P_3 \operatorname{mod} 1\} \subset (0,1)$. Then, as a function of $P_1$, $\sigma$ obeys
# $$
# P_1\in (0, 1) \ \text{and} \ \sigma(P_1) = 1 \quad \Leftrightarrow \quad P_1 \in (P_-, P_+) \cup (1-P_+, 1-P_-)
# $$
# Then let us consider a product of two three-point functions, with intervals $(P_-, P_+)$ and $(P'_-, P'_+)$. Let us order these numbers so that $\{P_-, P_+, P'_-, P'_+\} = \{a, b, c, d\}$ with $a<b<c<d$. Then
# $$
# P_1\in (0, 1) \ \text{and} \ \sigma(P_1) \neq \sigma'(P_1)\quad \Leftrightarrow \quad P_1 \in (a, b) \cup (c, d) \cup (1-d, 1-c) \cup (1-b, 1-a)
# $$
# and 
# $$
# P_1 \in (0, 1) \ \text{and} \ \sigma(P_1)=\sigma'(P_1) = 1 \quad \Leftrightarrow \quad P_1 \in 
# \left\{\begin{array}{ll} 
# \emptyset & \text{if}\ \  \operatorname{sign}(P_\pm - P'_\pm) = \text{constant}
# \\ 
# (b, c) \cup (1-c, 1-b) & \text{else} \end{array}\right.
# $$
# 

# ## Four-point function

# In[ ]:


class R4:
    """ Computing the four-point function. """

    def __init__(self, t_channel = False, Ps = [1/4.37, 1/11.73, 1/3.81, 1/12.46], 
                 charge = Charge('minimal', [1, 2]), eta = .00001274, Nmax = 4, 
                 Pcutoff = 2.25, Ucutoff = 20, reduced = False, epsilon = .1, barnes = False, disjoint = False):
        """ We need a minimal central charge, four momentums P, the order Nmax of the blocks' expansion, 
        cutoffs for the P integration and the Upsilon product formula, and the indication whether to 
        use reduced blocks. 
        We also give a parameter epsilon for the position of the integration line, and
        the deviation eta of the central charge from its rational value. 
        If c = 1 we can use Barnes' G-function for computing structure constants, but this is slower.
        We indicate whether we want to compute the integral on an intersection (default) or disjoint union.
        """  

        self.beta = charge.get('beta')
        self.qp = (1/self.beta - self.beta)/ 2
        self.charge = Charge('beta', self.beta + 1j * eta)
        (self.p, self.q) = charge.get('minimal')
        self.even = self.p - self.q % 2 == 0
        self.barnes = self.p == self.q and barnes
        self.factor = 1 / math.sqrt(self.p * self.q)
        self.t_channel = t_channel
        self.Ucutoff = Ucutoff
        self.Pcutoff = Pcutoff       
        self.Ps = Ps
        self.Nmax = Nmax
        self.computeBlock(Nmax)
        self.reduced = reduced
        self.epsilon = epsilon
        self.disjoint = disjoint

    def computeBlock(self, N):
        """ Computing the relevant Block. This method is here so that we can change N without
        reconstructing the object.
        """       
        self.block = Block([Dimension('P', self.Ps[i], self.charge) for i in range(4)], N, self.t_channel)

    @staticmethod 
    def upsilon(P):
        return mpmath.barnesg(1 + P) * mpmath.barnesg(1 - P)

    def Ksmooth(self, P):
        """ We compute the analytic factor of the three-point function. """
        Ps = self.Ps
        if self.t_channel:
            Ps = [Ps[1], Ps[2], Ps[3], Ps[0]]
        Psums = Z3.Psums([Ps[0], Ps[1], P]) + Z3.Psums([Ps[2], Ps[3], -P])
        if self.barnes:
            result = 1 
            for P1 in Psums:
                result *= R4.upsilon(P1)
            result /= R4.upsilon(2*P)**2
            return result
        Plist = [[-1j * P1, True] for P1 in Psums] + [[-1j * (self.qp + 2*P1), False] for P1 in [P, -P]]
        upsilons = Upsilons(self.beta, Plist)  
        return upsilons.product(self.Ucutoff)

    @staticmethod
    def interval(P2, P3, factor = 1):
        """ We compute the first interval where the non-analytic factor does not vanish. """
        (P2, P3) = (P2.real, P3.real)
        return list(numpy.sort(map(lambda x: x % factor, [P2 + P3, P2 - P3, P3 - P2, - P2 - P3]))[0:2])

    @staticmethod
    def intersections(interval1, interval2, display = False):
        """ Computes the xor and intersection of the two intervals, as a list of four bounds,
        plus two boolean which are True if the xor and intersections are respectively empty. 
        """
        diffs = map(lambda x: x > 0, [i1 - i2 for i1 in interval1 for i2 in interval2])
        bounds = interval1 + interval2
        bounds = numpy.sort(bounds)
        if display:
            R4.display([interval1, interval2, bounds], sort = False)        
        empty_intersection = diffs[0] == diffs[1] and diffs[0] == diffs[2] and diffs[0] == diffs[3]
        empty_xor = bounds[0] == bounds[1] and bounds[2] == bounds[3]
        return [bounds, empty_xor, empty_intersection]

    @staticmethod
    def extend(interval_list, period, interval):
        """ Extends interval_list into interval modulo period and reflection. """
        reflected = [[period - inter[1], period - inter[0]] for inter in interval_list]
        shifts = range(int(interval[0] / period), int(interval[1] / period))
        return [[inter[0] + i*period, inter[1] + i*period] 
                for inter in interval_list + reflected for i in shifts]

    @staticmethod
    def display(interval_list, sort = True):
        """ Orders the interval list, truncates the bounds, and prints. """
        if sort: 
            interval_list.sort(key = lambda interval: interval[0])
        print( [[round(bound, 3) for bound in interval] for interval in interval_list] )

    def value(self, q = .02):
        """ Computes the numerical value of the four-point function. The non-smooth terms are computed
        as sums of integrals over the intervals where they do not vanish.
        """

        blockNum = BlockNum(self.block, q)
        blockNum_c = BlockNum(self.block, q.conjugate())

        Ps = self.Ps
        period = self.factor
        if self.t_channel:
            Ps = [Ps[1], Ps[2], Ps[3], Ps[0]]
        if not self.even:
            Ps = [Ps[0] + period / 2, Ps[1], Ps[2], Ps[3] + period / 2]
        inters = R4.intersections(R4.interval(Ps[0], Ps[1], period), R4.interval(Ps[2], Ps[3], period))  
        bounds = inters[0]
        interval = [0, self.Pcutoff]
        xor = [] if inters[1] else R4.extend([[bounds[0], bounds[1]], [bounds[2], bounds[3]]], period, interval)
        intersection = [] if inters[2] else R4.extend([[bounds[1], bounds[2]]], period, interval)

        def integrand(P):
            Delta = P**2 
            return numerical(self.Ksmooth(P) * blockNum.value(Delta, self.reduced) 
                           * blockNum_c.value(Delta, self.reduced))

        intervals = xor if self.disjoint else intersection
        # self.xor_int = 2*sum([complex_quadrature(integrand, inter[0], inter[1])[0] for inter in xor])
        # self.intersection_int = 2*sum([complex_quadrature(integrand, inter[0], inter[1])[0] for inter in intersection])
        # self.smooth_int = complex_quadrature(lambda P: integrand(P + 1j * self.epsilon), -self.Pcutoff, self.Pcutoff)[0]
        return 2*sum([complex_quadrature(integrand, inter[0], inter[1])[0] for inter in intervals])


# In[ ]:


""" Test of crossing symmetry for Liouville for c \notin (-\infty,1)"""
b = .31 + 0.32j
Q = b + 1. / b
Ps = [0.2, 0.123, 0.3,0.17]
crossratio =  0.32
sch = FourPoint(b = b, Ps = Ps, Nmax = 8, normalized = True)
tch = FourPoint(b = b, Ps = Ps, t_channel=True,  Nmax = 8, normalized = True)
print( "crossing sym.", sch.value(mpmath.qfrom(m=crossratio)), tch.value(mpmath.qfrom(m= crossratio)) )
#ls = sch.value_discrete() 
#print( "discrete contributions", ls )
#polelist = sch.getcrossedpoles()
#sch.getCCdiscrete(polelist)
#tch = FourPoint(b = b, Ps = Ps, t_channel=True,  Nmax = 8, normalized = True)
#sch.value(mpmath.qfrom(m=.5)), tch.value(mpmath.qfrom(m=.5))
#ls = sch.value_discrete() 
#print( "discrete contributions", ls )
#import matplotlib.pyplot as plt
#poles = sch.getallpoles(3,3)
#print( poles )
#allpoles = poles.flatten()
#for j in range(16):
#    plt.plot(poles[j,:,:].flatten().real, poles[j,:,:].flatten().imag, "+-")
#plt.plot([-1, 2], [0,0], color = "black")
#plt.plot([-1, 8], [Q / 2,0], color = "black")
#plt.show()
#sch.getcrossedpoles()


# In[ ]:





# In[ ]:




