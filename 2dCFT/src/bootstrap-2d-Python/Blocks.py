#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division, print_function

from Auxiliary_classes import Formal, LatexTable, Series, numerical
from CFT import Charge, Dimension

from IPython.display import display, Math
from sympy import latex
import cmath, math, mpmath


# # Conformal blocks
# 
# We define and compute four-point $s$-channel conformal blocks. A block has parameters and variables. The parameters are the central charge $c$, the conformal dimensions of the four fields $\delta_i$, and the maximum order $N_{max}$ of the series expansion. The variables are the $s$-channel conformal dimension $\delta$, and the cross-ratio $x$. In addition to $x$ it is useful to use the variable $q$ such that
# $$
# x = \frac{\theta_2(q)^4}{\theta_3(q)^4}
# $$
# where 
# $$
# \theta_3(q) = \sum_{n\in\mathbb{Z}} q^{n^2} \quad , \quad \theta_2(q) = 2q^\frac14\sum_{n=0}^\infty q^{n(n+1)}
# $$
# 
# ## Blocks
# 
# For $mn\leq N_{max}$ we compute 
# $$
#  R_{m,n} = -\frac{1}{2}\frac{1}{D_{mn}} \prod_{r\overset{2}{=} 1-m}^{m-1} \prod_{s\overset{2}{=}1-n}^{n-1} \sqrt{(\delta_2-\delta_1)^2 -2\delta_{r,s}(\delta_1+\delta_2) + (\delta_{r,s})^2} 
# \sqrt{(\delta_3-\delta_4)^2 -2\delta_{r,s}(\delta_3+\delta_4) + (\delta_{r,s})^2}
# $$
# where we do not actually take square roots, because each factor appears twice, except the $(r,s)=(0,0)$ factor which is however a perfect square. The normalization factor is 
# $$
# D_{m,n} = -mn \prod_{r=1}^{m-1} r^2b^2 \left(r^2b^2 - \frac{n^2}{b^2}\right) 
# \prod_{s=1}^{n-1} \frac{s^2}{b^2}\left(\frac{s^2}{b^2} - m^2b^2\right)
# \prod_{r=1}^{m-1} \prod_{s=1}^{n-1} \left(r^2b^2 -\frac{s^2}{b^2} \right)^2
# $$
# Notice that if $\delta_1 = \delta_{r_1,s_1}$ and $\delta_2 = \delta_{r_2,s_2}$, then 
# $$
# \left\{\begin{array}{l} m\in |r_1-r_2|+1+2\mathbb{N} 
# \\ n \in |s_1-s_2| + 1+2\mathbb{N} \end{array} \right. \quad 
# \Rightarrow \quad R_{m,n} = 0
# $$
# and similarly if the fields with numbers $3$ and $4$ are degenerate.
# We then compute the coefficients
# $$
# C^N_{m,n} = R_{m,n}\left(\delta_{N-mn,0} + \sum_{m'n'\leq N-mn} \frac{C^{N-mn}_{m',n'}}{\delta_{m,-n}-\delta_{m',n'}} \right)
# $$
# These are the coefficients of the nontrivial factor
# $$
# H(q,\delta) = 1 + \sum_{N=1}^{N_{max}} \sum_{mn\leq N} C_{m,n}^N \frac{(16q)^N}{\delta-\delta_{m,n}}
# $$
# This makes sense if the four fields are degenerate and $\delta = \delta_{m, n}$, due to the double zero of $R_{m,n}$ and therefore $C_{m,n}^N$. 
# The $s$-channel conformal block for the correlation function $\langle V_1(x) V_2(0) V_3(\infty) V_4(1) \rangle$ is then 
# $$
#  F^{(s)}_{\delta}(x) =  x^{E_0} (1-x)^{E_1} \theta_3(q)^{-4E_2} 
# (16q)^{\delta} H(q,\delta)
# $$
# where we use the exponents 
# $$
# E_0 = -\delta_1-\delta_2-\frac{c-1}{24} \quad , \quad E_1 = -\delta_1-\delta_4-\frac{c-1}{24} \quad , 
# \quad E_2 = \delta_1+\delta_2+\delta_3+\delta_4+\frac{c-1}{24} 
# $$
# When writing the $s$-channel block, we split the channel-independent prefactor from the rest of the block. The $t$-channel block is 
# $$
# F^{(t)}_{\delta}(x) = (1-x)^{\delta_2+\delta_3-\delta_1-\delta_4} \left. F^{(s)}_{\delta}(1-x) \right|_{(1234)\rightarrow (2341)}
# $$
# 

# In[ ]:


class Block:
    """ A conformal block, viewed as a function of the cross-ratio x, and delta. """

    def __init__(self, dims, Nmax = 4, t_channel = False):
        """ We need the maximum order Nmax, four conformal dimensions, and the channel. 
        The central charge will be deduced from the dimensions: they should all 
        have the same, although we do not check it.
        """

        self.Nmax = Nmax
        self.t_channel = t_channel
        if t_channel:
            dims = [dims[1], dims[2], dims[3], dims[0]] 

        self.deltas = [dim.get('delta') for dim in dims]
        charge = dims[0].charge
        self.charge = charge
        self.B = charge.get('B')

        self.Delta = [[Dimension('degenerate', [r, s], charge).get('delta') 
                       for s in range(-Nmax, Nmax+1)] 
                          for r in range(Nmax+1)]

        """ If pairs of fields are degenerate, then the vanishing of elements of R allows us to 
        impose restrictions on pairs of indices in certain sums. We build a list of 
        allowed pairs [m, n], classified according to their products m*n, taking these 
        restrictions into account. 
        """
        restrictions = []
        for pair in [[0, 1], [2, 3]]:
            if dims[pair[0]].isDegenerate and dims[pair[1]].isDegenerate:
                (r1, s1) = dims[pair[0]].get('degenerate')
                (r2, s2) = dims[pair[1]].get('degenerate')
                restrictions.append([abs(r1 - r2), abs(s1 - s2)])  
        pairs = []
        for N in range(1, Nmax+1):
            pairs.append([])
        for m in range(1, Nmax+1):
            for n in range(1, Nmax//m+1):
                OK = True
                for restriction in restrictions:
                    r = m - restriction[0]
                    s = n - restriction[1]
                    if r > 0 and s > 0 and r % 2 == 1 and s % 2 == 1:
                        OK = False 
                if OK:
                    pairs[m*n-1].append([m,n])
        self.pairs = pairs                       

        self.R = Formal.matrix("R", Nmax)
        self.computeR()            

        self.C = [Formal.matrix("C^"+latex(N), Nmax) for N in range(1, Nmax+1)]    
        self.computeC() 

        """ Now we will compute the exponents for the block's prefactor. """
        S = charge.get('S')
        E0 = -self.deltas[2] - self.deltas[3] - S if t_channel else (-self.deltas[0] 
                                                                     - self.deltas[1] - S)
        self.E = [E0, -self.deltas[0] - self.deltas[3] - S, 
                  self.deltas[0] + self.deltas[1] + self.deltas[2] + self.deltas[3] + S]

    @staticmethod
    def threeSym(a,b,c):
        """ Computes the permutation-invariant quadratic combination. """
        return (a-b)**2 - 2*c*(a+b) + c**2

    @staticmethod
    def norm(B, m, n):
        """ Computes the normalization factor of R. """
        norm = - n*m
        for r in range(1, m):
            norm *= r**2 * (r**2 * B**2 - n**2)
        for s in range(1, n):
            norm *= s**2 * (s**2 - m**2 * B**2) / B**2
        for r in range(1, m):
            for s in range(1, n):
                norm *= (r**2*B - s**2/B)**2
        return norm

    def computeR(self):
        """ Computes the coefficients Rmn for m*n less or equal than Nmax. """
        N = self.Nmax
        factors = [[0 for s in range(-N, N+1)] for r in range(N)]
        deltas = self.deltas
        for r in range(N):
            for s in range(N // (r+1) + 1):
                if (r == 0 and s == 0):
                    factors[r][s+N] = (deltas[1] - deltas[0]) * (deltas[2] - deltas[3])
                else:
                    delta = self.Delta[r][s+N]
                    factors[r][s+N] = self.threeSym(deltas[0], deltas[1], delta)
                    factors[r][s+N] *= self.threeSym(deltas[2], deltas[3], delta)
                    delta = self.Delta[r][N-s] 
                    factors[r][N-s] = self.threeSym(deltas[0], deltas[1], delta)
                    factors[r][N-s] *= self.threeSym(deltas[2], deltas[3], delta)     
        for pairsList in self.pairs:
            for pair in pairsList:
                m = pair[0]
                n = pair[1]
                R = 1
                for r in range((m+1)%2, m, 2):
                    if r == 0:
                        for s in range((n+1) % 2, n, 2):
                            R *= factors[r][s+N]
                    else:        
                        for s in range(1-n, n, 2):
                            R *= factors[r][s+N]
                self.R[m-1][n-1] = R / Block.norm(self.B, m, n) / -2       

    def computeC(self):
        """ Recursively computes the coefficients C^N_{m,n} of the block's expansion 
        as a function of q and delta. 
        """
        Nmax = self.Nmax
        for N in range(1, Nmax+1):
            for L in range(N):
                for pair in self.pairs[L]:
                    m = pair[0]
                    n = pair[1]
                    M = N - m*n
                    C = 1
                    if M != 0:
                        C = 0
                        for K in range(M):
                            for pair in self.pairs[K]:
                                p = pair[0]
                                q = pair[1]
                                C += self.C[M-1][p-1][q-1] / (self.Delta[m][Nmax-n] 
                                                              - self.Delta[p][Nmax+q])
                    C *= self.R[m-1][n-1]         
                    self.C[N-1][m-1][n-1] = C 

    def series(self, delta, full = True):
        """ Computes the expansion of the block as a series in x or 1-x, depending on the 
        channel, after dropping the x or 1-x power prefactor. If full = False, only returns 
        a series in q, without prefactors and without converting to the variable x. 
        """        
        Nmax = self.Nmax

        coefs = Series([1])
        for N in range(1, Nmax + 1):
            coef = 0
            for L in range(N):
                for pair in self.pairs[L]:
                    m = pair[0]
                    n = pair[1]
                    if delta - self.Delta[m][Nmax+n] == 0:
                        print('Imminent division by zero with indices ', m, n)
                    coef += self.C[N-1][m-1][n-1] / (delta - self.Delta[m][Nmax+n])
            coefs.append(coef)

        if full:
            qExp = Series.qExp(Nmax + 1)
            qp = qExp.exp().product(Series.monomial(1))
            q = qp.multiply(1/16)
            prefactor = Series.binomial(Nmax + 1, self.E[1]).product(qExp.multiply(delta).exp())
            prefactor = prefactor.product(Series.theta(q).power(-4*self.E[2]))
            coefs = coefs.of(qp).product(prefactor) 

        return coefs

if __name__ == '__main__':

    c = Charge('beta', 1.2+.1*1j)
    dims =[Dimension('P',0.23+ .11*1j, c), Dimension('P', 3.43, c), Dimension('P', 0.13, c),
           Dimension('P', 1.3, c)]
    block = Block(Nmax=3, dims = dims)
    LatexTable(block.Delta)        
    term = block.series(.41).value[3]
    print( term )


# ## Hypergeometric blocks

# We study conformal blocks that contribute to the four-point function $\langle V_{(2,1)}(x) V_{\alpha_1}(0) V_{\alpha_2}(\infty) V_{\alpha_3}(1) \rangle$. These blocks are 
# $$
# F^{(s)}_+(x) = x^{b\alpha_1} (1-x)^{b\alpha_3} F(A,B,C,x)
# \\ 
# F^{(s)}_-(x)  = x^{b(Q-\alpha_1)} (1-x)^{b\alpha_3} F(A-C+1,B-C+1,2-C,x)
# \\
# F^{(t)}_+(x) = x^{b\alpha_1} (1-x)^{b\alpha_3} F(A,B,A+B-C+1,1-x)
# \\
# F^{(t)}_-(x) = x^{b\alpha_1} (1-x)^{b(Q-\alpha_3)} F(C-A,C-B,C-A-B+1,1-x)
# $$
# where we defined
# $$
# A = \tfrac12 + b(\alpha_1+\alpha_3-Q) + b(\alpha_2-\tfrac{Q}{2})  \\
# B = \tfrac12 + b(\alpha_1+\alpha_3-Q) - b(\alpha_2-\tfrac{Q}{2})  \\
# C = 1 + b(2\alpha_1-Q) 
# $$      
# We can compute such blocks either from the explicit formula, or from Zamolodchikov's recursion.

# In[ ]:


class HypergeometricBlock:
    """ An hypergeometric conformal block. """

    def __init__(self, Nmax, alphas, charge, channel = True, t_channel = False):
        """ Need three alphas, the charge, and a boolean which corresponds to the channel label.
        """

        b = charge.get('b')
        Q = charge.get('Q') 
        dims = [Dimension('degenerate', [2, 1], charge)]
        for i in range(3):
            dims.append(Dimension('alpha', alphas[i], charge))
        self.block = Block(dims, Nmax, t_channel)        

        if t_channel:
            alphas = [alphas[2], alphas[1], alphas[0]]
        if not channel:
            alphas[0] = Q - alphas[0]

        A = 1/2 + b*(alphas[0] + alphas[2] - Q) + b*(alphas[1] - Q/2)
        B = 1/2 + b*(alphas[0] + alphas[2] - Q) - b*(alphas[1] - Q/2)
        C = 1 + b*(2*alphas[0] - Q) 
        self.hypseries = Series.hypergeometric(Nmax + 1, A, B, C).product(
            Series.binomial(Nmax + 1, b*alphas[2]))
        self.series = self.block.series(Dimension('alpha', alphas[0] - b/2, charge).get('delta'))


if __name__ == '__main__':

    """ Let us check the agreement of the two computations. """

    """ First we do a numerical check. """
    hyp1 = HypergeometricBlock(10, [.1, .2, .3], Charge('b', math.sqrt(2) - 1), True, True)
    hyp1.series.display()
    hyp1.hypseries.display()



# ## Numerical computation of conformal blocks
# When define the value of a block at its pole $\delta=  \delta_{r,s}$
# as
# $$
# F^\text{reg}_{\delta_{r,s}}(x) + \log(q) R_{r,s}F_{\delta_{-r,s}}(x)
# $$
# where $F^\text{reg}_{\delta_{r,s}}(x)$ is the sum of the terms that do not diverge. 
# It is then understood that we should allow a contribution of $F_{\delta_{-r,s}}(x)$ witn an arbitrary coefficient.
# The logarithmic correction, and the contribution of $F_{\delta_{-r,s}}(x)$, can be omitted if $R_{r,s}=0$, which happens in particular if the fusion rules are obeyed. 
# They can also be omitted if $rs>N_{max}$.

# In[ ]:


class BlockNum:
    """ A conformal block, viewed as a function of delta, after giving a numerical value to q. """

    def __init__(self, block, q = .05):
        """ We need a block, and a complex value for q. """

        self.block = block
        Nmax = block.Nmax
        self.Nmax = Nmax
        self.t_channel = block.t_channel
        self.original_q = q    # Stored for the method regularized_combination
        self.q = cmath.exp(math.pi**2 / cmath.log(q)) if self.t_channel else q
        self.Delta = block.Delta
        self.x = mpmath.mfrom(q = self.q)    # In a t-channel block this is actually 1-x.
        thetaThree = mpmath.jtheta(3, 0, self.q)
        E = block.E
        """ We compute the modulus of the prefactor, taking the modulus of geometric
        quantities in order to anticipate their contribution to full correlation functions. We also 
        compute the reduced prefactor which appears if we drop the channel-independent piece.
        """
        self.prefactor =  (self.x)**E[0] * (1-self.x)**E[1] * thetaThree**(-4*E[2]) 
        # self.reducedPrefactor = (abs(-cmath.log(self.q)) / math.pi)**E[2]
        self.reducedPrefactor = thetaThree**(-4*E[2])  

        """ We now compute the residues of H(q, delta) at delta = delta_{m,n}. """
        self.C = Formal.matrix("C", Nmax)
        for pairsList in self.block.pairs:
            for pair in pairsList:
                m = pair[0]
                n = pair[1]
                C = 0
                for N in range(m*n, Nmax+1):
                    C += block.C[N-1][m-1][n-1] * (16*self.q)**N
                self.C[m-1][n-1] = C   

    def value(self, delta, reduced = True):
        """ Computes the value of the conformal block for a given delta or a given Dimension.
        By default we include only the channel-independent prefactor. 
        """
        pair0 = [0, 0]
        if isinstance(delta, Dimension):
            if delta.hasNullVector:
                pair0 = delta.get('degenerate')
                pair0 = [abs(pair0[0]), abs(pair0[1])]
            delta = delta.get('delta')
        H = 1
        Nmax = self.Nmax
        for pairsList in self.block.pairs:
            for pair in pairsList:
                m = pair[0]
                n = pair[1]
                if m != pair0[0] or n != pair0[1]:
                    H += self.C[m-1][n-1] / (delta - self.Delta[m][Nmax+n])
                else:
                    H += self.C[m-1][n-1] * cmath.log(self.q)
        if reduced:
            H *= self.reducedPrefactor
        else:
            H *= self.prefactor
        H *= (16*self.q)**delta
        return numerical(H)


# In[ ]:


if __name__ == '__main__':    
    """ Here we check the agreement between the series expansion and the numerical calculation. 
    The series converges much slower, and we need N = 15 to get the correct third digit. The numerical 
    calculation gives 15 correct digits for N = 12.
    """

    delta = .317
    deltas = [.1, .2, .3, .4] 
    charge = Charge('b', .5 - .1*1j)
    dims = [Dimension('delta', delta, charge) for delta in deltas]

    for N in range(4):        
        block = Block(dims, N, True)
        num = BlockNum(block)
        res1 = block.series(delta).compute(num.x, True) * (num.x)**(delta + block.E[0])
        res2 = num.value(delta, False)
        print( N, abs(res1/res2 - 1), res1, res2 )

    """ We check how the agreement depends on x. Not suprisingly, it is good for small x and gets worse 
    as x increases in the s-channel. (Situation reversed in the t-channel.)
    """

    N = 10    
    block = Block(dims, N, True)
    for i in range(1, 10):
        num = BlockNum(block, i*.01)
        res1 = block.series(delta).compute(num.x, True) * (num.x)**(delta + block.E[0])
        res2 = num.value(delta, False)
        print( num.x, abs(res1/res2 - 1), res1, res2 )     


# In[ ]:


if __name__ == '__main__':    
    """ We compute some numerical values, for comparing with other code. """
    c = Charge('beta', 1.2+.1*1j)
    print(c.get('c'))
    dims =[Dimension('Delta',0.23+ .11*1j, c), Dimension('Delta', 3.43, c), 
           Dimension('Delta', 0.13, c), Dimension('Delta', 1.3, c)]
    block = Block(Nmax=14, dims = dims, t_channel = True)
    num = BlockNum(block)
    for Delta in [-.15 + j/10 for j in range(10)]:
        dim = Dimension('Delta', Delta, c)
        delta = dim.get('delta')
        print(Delta, num.value(delta, False))


# In[ ]:




