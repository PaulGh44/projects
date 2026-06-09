#!/usr/bin/env python
# coding: utf-8

# # Testing exacts ansatzes for spectrums of four-point functions in two-dimensional CFT

# In[ ]:


from __future__ import division, print_function
import numpy, math, cmath, mpmath, time
import matplotlib.pyplot as plt

from CFT import *
from Blocks import Block, BlockNum
from Correlators import Reflection, GMM3


# ## General principles
# Let $V_0$ be a field with dimension $\Delta_0$ in a model with central charge $c$. Our aim is to test ansätze for the OPE $V_0V_0$, by considering crossing symmetry of the four-point function $Z(z)=\left<V_0(0)V_0(z)V_0(1)V_0(\infty)\right>$. We express an ansatz as a discrete set $A$ of fields with dimensions $(\Delta,\bar{\Delta})$, where $\bar{\Delta}-\Delta \in \mathbb{Z}$. If $D_{\Delta,\bar{\Delta}}$ are the corresponding structure constants, we define 
# $$
# Z^{(i)}(z) = \sum_{(\Delta,\bar{\Delta})\in \mathcal S} D_{\Delta, \bar{\Delta}} \mathcal{F}_\Delta^{(k)}(z)\mathcal{F}_{\bar{\Delta}}^{(k)}(\bar z) \qquad (k\in\{s,t\})
# $$
# and we want to solve the crossing symmetry equation for $D_{\Delta,\bar{\Delta}}$,
# $$
# Z^{(s)}(z) = Z^{(t)}(z)
# $$
# This is a linear system of infinitely many equations (due to the dependence on $z$) and discretely many unknowns. Let us truncate it to a finite linear system. 
# 
# For any integer $N$ we define the truncated ansatz $\mathcal S(N)$ as the $N$ fields with the lowest total dimensions $\Delta+\bar{\Delta}$, in particular $\mathcal S(0)$ is the ground state. If there are several fields with the same total dimensions, only certain values of $N$ may be reasonable, which we parametrize as $N(L)$ for $L$ a number. We make the normalization assumption that the structure constant for the ground state is one.
# 
# We then randomly choose $N$ point $z_j$ in the square $[0.5-\kappa, 0.5+\kappa]\times [-\kappa, \kappa]$, whose distances to the real line and to each other are at least 
# $$
# \epsilon = \frac{\kappa}{\sqrt{N+1}}
# $$
# Let $D_{\Delta,\bar{\Delta}}(N)$ be the unique solution of the truncated system
# $$
# \sum_{(\Delta,\bar{\Delta})\in \mathcal S(N)} D_{\Delta, \bar{\Delta}}(N) \mathcal{F}_\Delta^{(s)}(z_j)\mathcal{F}_{\bar{\Delta}}^{(s)}(\bar z_j) = \sum_{(\Delta,\bar{\Delta})\in \mathcal S(N)} D_{\Delta, \bar{\Delta}}(N) \mathcal{F}_\Delta^{(t)}(z_j)\mathcal{F}_{\bar{\Delta}}^{(t)}(\bar z_j)
# $$
# If our ansatz $\mathcal S$ is correct, we expect that $D_{\Delta,\bar{\Delta}}(N)$ does not depend much on the choice of the points $z_j$, and converges when $N$ becomes large. 

# In[ ]:


class Points:
    """ A collection of static methods for taking care of the geometry of test points. """

    @staticmethod
    def newpoint(kappa):
        """ Generating a random point in a square centered at .5, of size 2*kappa. """
        x = numpy.random.uniform(0.5-kappa, 0.5+kappa)
        y = numpy.random.uniform(-kappa, +kappa)
        return x + y*1j

    @staticmethod
    def generate(N, kappa):
        """ Generating N random points in a square centered at .5, of size 2*kappa. """
        epsilon = kappa / math.sqrt(N+1)  # the minimum distance to each other and to the real line.
        points = []
        while len(points) < N:
            point = Points.newpoint(kappa)
            if abs(point.imag) < epsilon:
                continue
            for other_point in points:
                if abs(point - other_point) < epsilon:
                    break
            else:   
                points.append(point)
        return points   

    @staticmethod
    def vandermonde(points):
        """ Computing the Vandermonde determinant of a collection of points. """
        return abs(numpy.linalg.det(numpy.vander(points)))

    @staticmethod
    def plot(points):
        """ Plotting complex points. """
        points = numpy.array(points)
        plt.scatter(points.real,points.imag, color ='red')
        plt.show()  

#if __name__ == '__main__':        
#    points = Points.generate(15, .3)
#    Points.plot(points)


# ## Generalized Minimal Models as test cases
# Valuable examples are provided by Generalized Minimal Models, which involve finitely many diagonal fields.
# Let $V_0=V_{(r,s)}$ a degenerate field, then the OPE $V_0V_0$ involves $rs$ degenerate fields. 

# In[ ]:


class GMM_Spectrum:
    """ The spectrum of a Generalized Minimal Model, together with its truncations. """

    def __init__(self, charge = Charge('c', .129), pair = [4,3]):
        """ We need a central charge c<1, and the indices of the field V_0. """

        self.charge = charge
        self.beta = self.charge.get('beta')
        self.field = Dimension('degenerate', pair, self.charge)

        dims = self.field.fuse(self.field)
        self.fields = Field.sort([Field(dim, diagonal = True) for dim in dims])
        self.csts = []
        for field in self.fields:
            dim = field.dims[0]
            self.csts.append(GMM3([dim, self.field, self.field]).value**2/Reflection(dim).value)
        normalization = self.csts[0]
        self.csts = [cst / normalization for cst in self.csts]

    def truncate(self, N = 0, show = False):
        """ Returns the left and right (shifted) dimensions of the first N+1 fields. """
        if N == 0 or N > len(self.fields):
            N = len(self.fields)
        fields = self.fields[:N]
        if show:
            for field in fields:
                field.display()
        return fields

    def display(self, N = 0, real_part = False):
        """ Displays field labels and the associated structure constants. """
        if N == 0 or N > len(self.fields):
            N = len(self.fields)
        for i in range(N):
            cst = self.csts[i]
            if real_part:
                cst = cst.real
            print( self.fields[i].dims[0].get('degenerate'), cst )

if __name__ == '__main__':  
    GMM_test = GMM_Spectrum(pair = [3,2])
    for field in GMM_test.truncate(12):
        field.display()
    GMM_test.display()



# ## Non-diagonal spectrums
# Let us consider the non-diagonal field $\Phi_{m,n}$ with the conformal dimensions 
# $$
# (\Delta,\bar\Delta ) = (\Delta_{m,n},\Delta_{m,-n})
# $$
# where
# $$
# \Delta_{m,n} = \frac{c-1}{24} + \frac14 \left(m\beta-\frac{n}{\beta}\right)^2
# $$
# These fields come in pairs $(\Phi_{m,n},\Phi_{-m,n})$ if $mn\neq 0$. They are bosonic for $mn\in\mathbb{Z}$ and fermionic for $mn\in \mathbb{Z}+\frac12$.
# We consider spectrums of the type 
# $$
# S_{X, Y} = \bigoplus_{m\in X}\bigoplus_{n\in Y} \Phi_{m,n}
# $$
# where the sets $X,Y$ are of the type $a\mathbb{Z} + b$ with $b\in\{0,\frac{a}{2}\}$ so that $X=-X$ and $Y=-Y$. To get a cutoff on such spectrums, we choose a number $\sigma$ and consider the pairs $(m,n)\in [0, \frac{\sigma}{\sqrt{|\Re\beta^2|}}]\times [0, \frac{\sigma}{\sqrt{|\Re\frac{1}{\beta^2}|}}]$. In this rectangle we select the pairs that also obey $\left|\Re(m^2\beta^2+\frac{n^2}{\beta^2})\right|\leq \sigma^2$.

# In[ ]:


class Another_Spectrum:
    """ A non-diagonal ansatz for the spectrum. """

    def __init__(self, beta = 1.0931, spacings = [2, 1], shifts = [False , True], 
                 deg = False, diag = False, clip = 0):
        """ We specify the spacings and shifts of m and n respectively, which belong to sets 
        of the type spacing * Z + shift. The only allowed shift is half the spacing, so that 
        the resulting set is even. 
        The 'deg' variable says whether to include diagonal fields that correspond to null
        vectors of degenerate fields. 
        The 'diag' variable says whether to include diagonal degenerate fields of the type
        (1, n), including the identity field.
        The 'clip' variable allows removing the first few fields from the spectrum.
        """

        self.spacings = spacings
        self.shifts = shifts
        self.deg = deg
        self.diag = diag
        self.clip = clip
        self.beta = beta
        self.charge = Charge('beta', beta)
        self.field = Dimension('degenerate', (0, 1/2), self.charge)

    def truncate(self, L = 4, show = False):
        sigma = L
        sigmas = [math.sqrt(abs((self.beta**2).real)), math.sqrt(abs((self.beta**(-2)).real))]
        ranges = []
        for i in range(2):
            shift = self.spacings[i]/2 if self.shifts[i] else 0
            if shift == int(shift):
                shift = int(shift)     # We need to know whether we have integers
            ranges.append([j + shift for j in range(0, int(sigma/sigmas[i] + 1 - shift), 
                                                        self.spacings[i])])
        fields = []
        for m in ranges[0]:
            for n in ranges[1]:
                field = Field(Dimension('degenerate', (m, n), self.charge), non_diagonal = True)
                if field.total_dimension.real < sigma**2/2:
                    if m != 1 or n != 0 or not self.diag:     # Avoid having (1, 0) twice.
                        fields.append(field)     
                    if m != 0 and n != 0:
                        fields.append(Field(Dimension('degenerate', (m, -n), self.charge), 
                                            non_diagonal = True))
                    if self.deg and field.dims[0].hasNullVector:
                        descendent = Field(Dimension('degenerate', (-m, n), self.charge), 
                                           diagonal = True)
                        if descendent.total_dimension.real < sigma**2/2:
                            fields.append(descendent)
        if self.diag:
            for n in range(int(2/sigmas[1]**2+sigma/sigmas[1]+1)):
                field = Field(Dimension('degenerate', (1, n), self.charge), diagonal = True)
                if field.total_dimension.real < sigma**2/2:
                    fields.append(field)
        fields = Field.sort(fields)
        fields = fields[self.clip:]
        if show:
            for field in fields:
                field.display()
        return fields

if __name__ == '__main__':

    beta = .9
    spec = Another_Spectrum(beta = beta)
    dims = spec.truncate(5, True)
    spec = Another_Spectrum(beta = .9 + .1*1j, spacings = [2, 1], shifts = [True, False])
    dims = spec.truncate(5, True)
    spec = Another_Spectrum(beta = beta, shifts = [False, False], deg = True, diag = True)
    dims = spec.truncate(5, True)


# Let us consider the spectrum where $m$ is integer while $n$ is a rational number such that $mn$ is an even integer,
# $$
# S = \bigoplus_{m\in\mathbb{Z}} \bigoplus_{M\in\mathbb{Z}} \Phi_{m, \frac{2M}{m}}
# $$

# In[ ]:


class Rational_Spectrum:
    """ Another non-diagonal ansatz. """

    def __init__(self, beta = .9, deg = True, diag = True):
        """ The 'deg' variable says whether to include diagonal fields that correspond to null
        vectors of degenerate fields. 
        The 'diag' variable says whether to include diagonal degenerate fields of the type
        (1, n), including the identity field.
        """

        self.deg = deg
        self.diag = diag
        self.beta = beta
        self.charge = Charge('beta', beta)
        self.field = Dimension('degenerate', (0, 1/2), self.charge)

    def truncate(self, L = 4, show = False):

        sigma = L
        sigmas = [math.sqrt(abs((self.beta**2).real)), math.sqrt(abs((self.beta**(-2)).real))]
        fields = []
        for m in range(1, int(sigma/sigmas[0] + 1)):
            for N in range(int(sigma/sigmas[1]*m/2) + 1):
                n = 2*N/Rational(m)
                if n == int(n):
                    n = int(n)
                field = Field(Dimension('degenerate', (m, n), self.charge), non_diagonal = True)
                if field.total_dimension.real < sigma**2/2:
                    if m != 1 or n != 0 or not self.diag:     # Avoid having (1, 0) twice.
                        fields.append(field)     
                    if m != 0 and n != 0:
                        fields.append(Field(Dimension('degenerate', (m, -n), self.charge), 
                                            non_diagonal = True))
                    if self.deg and field.dims[0].hasNullVector:
                        descendent = Field(Dimension('degenerate', (-m, n), self.charge), 
                                           diagonal = True)
                        if descendent.total_dimension.real < sigma**2/2:
                            fields.append(descendent)
        for n in range(int(sigma/sigmas[1] + 1)):     # special treatment of the case m = 0
            fields.append(Field(Dimension('degenerate', (0, n), self.charge), diagonal = True))
        if self.diag:
            for n in range(int(2/sigmas[1]**2+sigma/sigmas[1]+1)):
                field = Field(Dimension('degenerate', (1, n), self.charge), diagonal = True)
                if field.total_dimension.real < sigma**2/2:
                    fields.append(field)
        fields = Field.sort(fields)
        if show:
            for field in fields:
                field.display()
        return fields

if __name__ == '__main__':

    spec = Rational_Spectrum()
    dims = spec.truncate(3, True)


# In[ ]:


class Spectrum_Union:
    """ Union of several spectrums. """

    def __init__(self, spectrums):
        """ We need a list of spectrums with the same central charge and field. """

        self.field = spectrums[0].field
        self.charge = spectrums[0].charge
        self.spectrums = spectrums

    def truncate(self, L = 4, show = False):

        fields = []
        for spectrum in self.spectrums:
            fields = fields + spectrum.truncate(L)
        fields = Field.sort(fields)
        if show:
            for field in fields:
                field.display()
        return fields


if __name__ == '__main__':

    beta = .611
    spectrum1 = Another_Spectrum(beta = beta, shifts = [False, False], deg = True)
    spectrum2 = Another_Spectrum(beta = beta, spacings = [4, 1], shifts = [False, True])
    spectrum3 = Another_Spectrum(beta = beta, spacings = [2, 2], 
                                 shifts = [True, False], diag = True)
    spectrum = Spectrum_Union([spectrum1, spectrum2, spectrum3])
    spectrum.truncate(L = 3, show = True)


# ## Dotsenko's spectrum at $c=0$
# In his [recent article](https://arxiv.org/abs/1606.09162), Dotsenko finds only one channel in the fully permutation-symmetric four-point function, corresponding to the diagonal $(2,0)$ field. This field is however degenerate, with the column of singular vectors 
# $$
# (2, 0) - (4, 0) - (8, 0) - (10, 0) - (14, 0) - (16, 0) - \cdots
# $$
# If we therefore write the corresponding representation as an infinite sum of Verma modules, then the spectrum becomes non-diagonal, and can be characterized in terms of fields $\Phi_{m,n}$ with 
# $$
# n\in 2\mathbb{Z} \quad , \quad m - \frac{n}{2} \in 2\mathbb{Z} \quad , \quad m\notin 3 \mathbb{Z}
# $$

# In[ ]:


class Dotsenko_Spectrum:

    def __init__(self):

        self.beta = math.sqrt(2/3) + 10**(-9)
        self.charge = Charge('beta', self.beta)
        self.field = Dimension('delta', 1/16/self.beta**2, self.charge)

    def truncate(self, L = 4, show = False):

        sigma = L
        sigmas = [math.sqrt(abs((self.beta**2).real)), math.sqrt(abs((self.beta**(-2)).real))]

        fields = []
        for n in range(0, int(sigma/sigmas[1]), 2):
            for m in range(int(n/2) % 2, int(sigma/sigmas[0]), 2):
                if int(m/3) != m/3:
                    field = Field(Dimension('degenerate', (m, n), self.charge), 
                                  non_diagonal = True)
                    if field.total_dimension.real < sigma**2/2:
                        fields.append(field)
                        if m != 0 and n != 0:
                            fields.append(Field(Dimension('degenerate', (m, -n), self.charge), 
                                            non_diagonal = True))
        fields = Field.sort(fields)
        if show: 
            for field in fields:
                field.display()
        return fields

if __name__ == '__main__':

    spec = Dotsenko_Spectrum()
    spec.truncate(9, True)


# ## Determining the structure constants

# In[ ]:


class Struct_Csts:
    """ A determination of the structure constants. """

    def __init__(self, spectrum = GMM_Spectrum(), L = 2, blocklevel = 12, 
                 kappa = .22, show = False):

        self.spectrum = spectrum
        self.blocklevel = blocklevel
        self.kappa = kappa
        self.fields = spectrum.truncate(L, show = show)
        self.indices = [field.dims[0].get('degenerate') for field in self.fields]
        self.N = len(self.fields) - 1

        self.field = spectrum.field
        dims = [spectrum.field for i in range(4)]
        self.block_s = Block(Nmax = self.blocklevel, dims = dims)
        self.block_t = Block(t_channel = True, Nmax = self.blocklevel, dims = dims)

    def blocks(self, z, reduced = True, channel = None):
        """ Computes the needed values of conformal blocks. Computes both channels
        if None, only the s-channel if True, and only the t-channel if False.        
        """

        q = mpmath.qfrom(m = z)
        if channel or channel is None:
            block_s =   BlockNum(block = self.block_s, q = q)
            block_s_c = BlockNum(block = self.block_s, q = q.conjugate())
        if not channel or channel is None:
            block_t  = BlockNum(block = self.block_t, q = q)
            block_t_c = BlockNum(block = self.block_t, q = q.conjugate())
        values = []

        for field in self.fields:
            value = []
            if channel or channel is None:
                value.append(block_s.value(field.dims[0], reduced) 
                             * block_s_c.value(field.dims[1], reduced))
            if not channel or channel is None:
                value.append(block_t.value(field.dims[0], reduced)
                             *block_t_c.value(field.dims[1], reduced))
            if channel is not None:
                value = value[0]
            values.append(value)

        return values

    def constants(self, test = False):
        """ Computing the structure constants by imposing crossing symmetry. Also allows to 
        test whether the supposedly correct structure constants that may come with the spectrum 
        indeed solve the system. """

        matrix = []
        f = []        
        points = Points.generate(self.N, self.kappa)

        for point in points:

            values = self.blocks(point)
            diffs = [value[0] - value[1] for value in values]
            f.append(- diffs.pop(0))
            matrix.append(diffs)

        csts = [1] + numpy.linalg.solve(matrix, f).tolist()  # Includes the first constant.
        if test: 
            for i in range(self.N):
                print( csts[i], self.spectrum.csts[i], abs(1 - csts[i]/self.spectrum.csts[i]) )
        self.csts = csts
        return csts

    def decompose(self, struct_csts, flip = False):
        """ Decomposing the s-channel four-point function computed from another Struct_Csts
        object, into t-channel conformal blocks. 
        """

        matrix = []
        f = []
        points = Points.generate(self.N + 1, self.kappa)
        for point in points: 
            matrix.append(self.blocks(point, channel = False))
            f.append(struct_csts.values(point, flip = flip, channel = True))
        csts = numpy.linalg.solve(matrix, f).tolist()
        self.csts = csts
        return csts

    def values(self, z, reduced = True, flip = False, channel = None):
        """ Computes the value of the four-point function in both channels at a given point. 
        Computes both channels if None, only the s-channel if True, and only the t-channel 
        if False. Can flip signs of terms with odd spins.
        """

        blocks = self.blocks(z, reduced, channel)       

        channels = [0, 1] if channel is None else [0]
        values = [0 for i in channels]
        for i in range(self.N + 1):
            for j in channels:
                block = blocks[i][j] if channel is None else blocks[i]
                term = self.csts[i] * block
                if flip and int(self.fields[i].spin) % 2 == 1:
                    term = - term
                values[j] += term

        if channel is not None:
            values = values[0]

        return values 

    def display(self):
        """ Displaying fields and constants.""" 

        for i in range(len(self.fields)):
            print( self.fields[i].dims[0].get('degenerate'), self.csts[i] )



if __name__ == '__main__':

    spectrum = Another_Spectrum(beta = .412)
    struct = Struct_Csts(spectrum = spectrum, L = 4, show = True)
    struct.constants()
    print( struct.values(.2) )
    print( struct.values(.2, flip = True) )
    struct2 = Struct_Csts(spectrum = spectrum, L = 4)
    print( 'Original bootstrap results: ' )
    struct.display()
    print( 'Reconstruction: ' )
    struct2.decompose(struct)
    struct2.display()


# In[ ]:


class Struct_Csts_Stats:
    """ Computing four-point functions for various random drawings of sample points,
    and studying the statistics of the results. If a Struct_Csts object is given, then
    we decompose the four-point function of that object. """

    def __init__(self, Nsamples = 10, z = .1, struct_csts = None, flip = False, **kwargs):

        self.data = []
        self.constants = []
        time0 = time.clock()
        for i in range(Nsamples):
            correl = Struct_Csts(**kwargs)
            if struct_csts is None:
                correl.constants()
            else:
                correl.decompose(struct_csts, flip = flip)
            if i == 0:
                self.indices = correl.indices
            self.data.append(correl.values(z))
            self.constants.append(correl.csts)
        self.mean = numpy.mean(self.data, 0)
        self.std = numpy.std(self.data, 0)
        self.mean_cst = numpy.mean(self.constants, 0)
        self.std_cst = numpy.std(self.constants, 0)
        self.duration = time.clock() - time0
        self.cvar = []
        for i in range(len(self.indices)):
            self.cvar.append(self.std_cst[i] / abs(self.mean_cst[i]))

    def display_cst(self):

        for i in range(len(self.indices)):
            print( self.indices[i], self.mean_cst[i], self.cvar[i] )


