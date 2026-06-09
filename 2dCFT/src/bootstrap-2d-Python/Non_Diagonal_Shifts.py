#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division, print_function
from sympy import Rational, re, im
import math, cmath, time

from CFT import *
from Correlators import *
from Spectrum_bootstrap import *
from Data import Table
import numbers
from Auxiliary_classes import numerical
import numbers


# # Shift equations and four-point functions in non-diagonal CFT
# In this notebook we define classes to compute four-point functions and structure constants in a non-diagonal CFT.
# 
# The structure constants are computed by recursively using the shift equations of the analytic conformal bootstrap.
# 
# Four-point functions are computed using the non-diagonal spectrum $\mathbb{S}_{2\mathbb{Z}, \mathbb{Z}+\frac{1}{2}}$. 
# 
# ##  Notations
# 
# We write the central charge in terms of a parameter $\beta$ such that
# \begin{align}
# c = 1-6\left(\beta - \frac{1}{\beta}\right)^2\, .
# \end{align}
# Diagonal fields $V^D_{P}$ can have arbitrary mommentums $P=\bar P$.
# Non-diagonal fields $V^N_{(r,s)}$ are labelled by two half-integer numbers $r,s$, and their left and right momentums are
# \begin{align}
# P &= P_{(r,s)} = \frac{1}{2}\left(r\beta - \frac{s}{\beta}\right)\, ,\\
# \bar{P} &= P_{(r,-s)}\, .
# \end{align}
# 
# The central charge is assumed to obey $\Re{c}<13$.
# 
# ##  Shift equations and four-point structure constants
# 
# The ratios that appear in shift equations corresponding to the degenerate field are $V_{\langle 2,1 \rangle}$ are:
# \begin{align}
# \rho\left(V_1|V_2, V_3 \right) = -(-1)^{2s_2} \frac{\Gamma(-2\beta P_1)\Gamma(-2\beta \sigma_1\bar{P}_1)}{\Gamma(2\beta P_1)\Gamma(2\beta \sigma_1\bar{P}_1)} \prod_{\pm \pm}\frac{\Gamma(\frac{1}{2} + \beta P_1 \pm \beta P_2 \pm \beta P_3)}{(\frac{1}{2} - \beta \sigma_1 \bar{P}_1 \pm \beta \bar{P}_2 \pm \beta \bar{P}_3)}\, ,
# \end{align}
# and 
# \begin{align}
# \rho(V) = -\frac{\Gamma(-2\beta P)\Gamma(-2\beta \sigma\bar{P})}{\Gamma(2\beta P)\Gamma(2\beta \sigma\bar{P})}  \frac{\Gamma(\beta^2+2\beta P)\Gamma(1-\beta^2+2\beta P)}{\Gamma(\beta^2-2\beta \sigma \bar{P})\Gamma(1-\beta^2-2\beta \sigma \bar{P})}\, .
# \end{align}
# 
# The shift equations for the four-point structure constants are
# \begin{align}
# \frac{D(V^{+}_s)}{D(V^{-}_s)} = \frac{\rho \left(V_s|V_1,V_2 \right)\rho \left(V_s|V_3,V_4 \right)}{\rho\left( V_s \right)}\, .
# \end{align}
# 
# We also implement the dual shift equations, corresponding to the degenerate field $V_{\langle 1,2 \rangle}$.
# 
# 
# ## Normalizations
# 
# Computing structure constants using the shift equations makes it necessary to choose a starting point for the recursions. We can arbitrarily fix structure constants corresponding to the lowest dimension to be $1$, i.e:
# 
# \begin{align}
# D\left( V^N_{\left( 0,\frac{1}{2} \right)} \right) = 1\, .
# \end{align}
# 
# However, comparing $s$-and $t$-channel computations requires that we determine the relative normalization between both channels. This is given by
# 
# \begin{align}
# \frac{D_{2341}\left( V^N_{\left( 0,\frac{1}{2} \right)} \right)}{D_{1234}\left( V^N_{\left( 0,\frac{1}{2} \right)} \right)} = (-1)^{\sum_{i = 1}^{4}\text{Spin}_i } \prod_{j=0}^{m-1} \prod_{k=0}^{n-1} \left[ \frac{\tilde{\rho} \left( V^N_{\left( r_2, \, s_2 -\delta_s (2j+1) \right) }, V^N_{\left( 0,\frac{1}{2} \right)} , V_3 \right) } {\tilde{\rho}\left( V^N_{\left( r_2, \, s_2 -\delta_s (2j+1) \right) }, V^N_{\left( 0,\frac{1}{2} \right)}, V_1 \right) }\right]^{\delta_s}
# \times
# \left[ \frac{\rho \left( V^N_{\left( r_2 -\delta_r (2k+1) , \, s_4 \right) }, V^N_{\left( 0,\frac{1}{2} \right)}, V_3 \right) } {\rho\left( V^N_{\left( r_2 -\delta_r (2k+1) , \, s_4 \right) }, V^N_{\left( 0,\frac{1}{2} \right)}, V_1 \right) }\right]^{\delta_r} \, , 
# \end{align}
# 
# where 
# 
# \begin{align}
# s_2 - s_4 &= 2 \delta_s m, \quad m\in \mathbb{Z}_{\geq 0}, \\
# r_2 - r_4 &= 2 \delta_r n, \quad n\in \mathbb{Z}_{\geq 0}\, .
# \end{align}
# 
# Thus, when computing four-point functions we normalize the first $s$-channel structure constant as $D_{1234}\left( V^N_{\left( 0,\frac{1}{2} \right)} \right) = 1$, and multiply the $t$-channel computation by the factor above.
# 
# ### Ordering of the fields
# 
# Four-point functions will be computed correctly only if diagonal and non-diagonal fieds alternate, i.e. if they are of the types
# \begin{align}
# \left\langle  V^D_1 V^N_2 V^D_3 V^N_4 \right\rangle \, ,\quad \left\langle  V^N_1 V^D_2 V^N_3 V^D_4 \right\rangle\, .
# \end{align}
# 
# These are the cases where the $s$- and $t$-channel spectrums are both $\mathbb{S}_{2\mathbb{Z}, \mathbb{Z}+\frac{1}{2}}$.
# 
# ## Four-point structure constants

# In[ ]:


class ND_csts:
    """ Computing the structure constants of a given four point function.  """

    def __init__(self, ext_fields):
        """ext_fields: LIST of four FIELDS. All should have the same central charge.
           The value of the central charge is taken form the first field. """ 

        self.fields = ext_fields

        self.Ps = [field.dims[0].get('P') for field in ext_fields ]
        self.bPs = [field.dims[1].get('P') for field in ext_fields]

        self.charge = self.fields[0].dims[0].charge
        self.beta = self.charge.get('beta')

    @staticmethod
    def rho(Vs, dual = False, full = True):
        """Compute rho and rho-tilde.
        Vs: LIST containing three or a single field (for rho(V)).
        dual = If True, changes beta --> -1/beta in the arguments of the gammas, giving rho-tilde.
        deg = If True, generates rho(V). 
        full = If True, computes prefectors de
        Use: By default, give three fields and get rho(V1|V2,V3).
        dual = True --> get rho-tilde(V1|V2,V3).
        Degenerate rho's: Set deg = True, and give only one field. The
        degenerate momemtums are created within rho. """


        beta = Vs[0].dims[0].charge.get('beta') # We don't use self.beta because it's a static method.

        deg = (len(Vs) == 1)

        sigma = 1

        if dual:
            """Inverts beta to get rho tilde. If the shifted field is non-diagonal,
            changes the sign of sigma.
            We determine if the field is diagonal by comparing the left and right Dimension objects
            (not the value, the instance of the class.)"""
            beta = -1/beta
            if Vs[0].non_diagonal:
                sigma = -1

        if deg:
            Pdeg = Dimension(variable = 'degenerate', value = (2,1), charge = Charge('beta', beta )).get('P')

            L_P = [Vs[0].dims[0].get('P'), Pdeg, Vs[0].dims[0].get('P')]
            R_P = [Vs[0].dims[1].get('P'), Pdeg, Vs[0].dims[1].get('P')]
        else:
            L_P = [V.dims[0].get('P') for V in Vs]
            R_P = [V.dims[1].get('P') for V in Vs]


        """Performing the computation. This takes a few steps:
        1 - Compute the factor depending on all momentums
        2 - if full
        Determine the sign factor.
        Compute the factor depending only on the shifted field.
        3 - return product of computed factors.
        """

        prod = 1
        for i in [+1,-1]:
            for j in [+1,-1]:
                prod *= (scipy.special.gamma(1/2 + beta* L_P[0] + i * beta*L_P[1]+ j * beta*L_P[2])
                         /scipy.special.gamma(1/2 - sigma *beta* R_P[0] + i * beta*R_P[1]+ j * beta * R_P[2]))

        if full:
            """Computes sign and normalization factor"""
            if deg:
                sign =1
            elif dual:
                sign = Vs[1].signs[0]
            else:
                sign = Vs[1].signs[1]

            norm = - (scipy.special.gamma(-2*beta*L_P[0])*scipy.special.gamma(-2*beta*sigma *R_P[0]) 
                   /scipy.special.gamma(2*beta*L_P[0]) /scipy.special.gamma(2*beta*sigma *R_P[0]) )

            return sign * norm * prod
        else:
            return prod

    def relative_norm(self):
        """Relative normalization between s and t channels."""

        D0 = Dimension(variable = 'degenerate', value = (0,1/2), charge = Charge('beta', self.beta))
        V0 = Field(dim = D0, non_diagonal = True)

        nd_fields = [] # a list of the non-diag fields
        d_fields = [] # a list of the diag fields
        for f in self.fields:
            if f.non_diagonal == True:
                nd_fields.append(f)
            else:
                d_fields.append(f)


        (r2, s2) = nd_fields[0].dims[0].get('degenerate') 
        (r4, s4) = nd_fields[-1].dims[0].get('degenerate')


        """Testing if a reflection is necessary for the shifts in s."""
        if (s2-s4) % 2 != 0:
            (r4, s4) = (-r4, -s4)

        dr = 1 if r2-r4 >= 0 else -1
        ds = 1 if s2-s4 >= 0 else -1

        m = int(abs(s2-s4)/2)
        n = int(abs(r2-r4)/2)

        s_factor = 1
        r_factor = 1

        for j in range(m):
            sj = s2 - ds*(2*j+1)

            Dj = Dimension(variable = 'degenerate', value = (r2, sj), charge = Charge('beta', self.beta))
            Vj = Field(dim = Dj, non_diagonal = True)

            s_factor *= (ND_csts.rho([Vj, V0, d_fields[-1]], dual = True, full = False)
                         /ND_csts.rho([Vj, V0, d_fields[0]], dual = True, full = False) )**ds

        for k in range(n):
            rk = r2 - dr*(2*k+1)

            Dk = Dimension(variable = 'degenerate', value = (rk, s4), charge = Charge('beta', self.beta))
            Vk = Field(dim = Dk, non_diagonal = True)

            r_factor *= (ND_csts.rho([Vk, V0, d_fields[-1]], full = False)
                         /ND_csts.rho([Vk, V0, d_fields[0]], full = False) )**dr

        spin = nd_fields[0].spin + nd_fields[-1].spin

        return numerical( (-1)**spin *r_factor * s_factor )

    def Shift(self, pair, variable = 'r'):
        """Takes the indices of a field, and a variable as a string. 
        Returns D(r+1,s)/D(r-1,s) or D(r,s+1)/D(r,s-1)."""

        Drs = Dimension(variable = 'degenerate', value = pair, charge = Charge('beta', self.beta))
        Vrs = Field(dim = Drs, non_diagonal = True)

        if variable == 'r':                         
            return (ND_csts.rho([Vrs, self.fields[0], self.fields[1]])
                    *ND_csts.rho([Vrs, self.fields[2], self.fields[3]])
                   /ND_csts.rho([Vrs]) )

        if variable == 's':    
            return (ND_csts.rho([Vrs, self.fields[0], self.fields[1]], dual = True)
                    *ND_csts.rho([Vrs, self.fields[2], self.fields[3]], dual = True)
                   /ND_csts.rho([Vrs], dual = True) )

    def value(self, pair, path = True):
        """Calculating the D(r,s)/D(0,1/2), for a single choice of indices.
        If path is True, first the shift in r is done. If False, first we do s."""

        cst = 1
        (r, s) = pair

        sign_r = 1 if r >= 0 else -1
        sign_s = 1 if s >= 0 else -1

        s0 = 1/2 if (s-1/2) % 2 == 0 else -1/2
        s_start = int(s0 + sign_s/2)
        s_end = int(s -sign_s/2)

        if path == True:
            for rstep in range(sign_r*1, r,sign_r*2):
                cst *=self.Shift((rstep, s0), 'r')**sign_r

            for sstep in range(s_start, s_end, sign_s*2):
                cst *=self.Shift((r, sstep + sign_s/2), 's')**sign_s
        else:
            for sstep in range(s_start, s_end, sign_s*2):
                cst *=self.Shift((0, sstep + sign_s/2), 's')**sign_s
            for rstep in range(sign_r*1, r ,sign_r*2):
                cst *=self.Shift((rstep, s), 'r')**sign_r

        return cst

    def make_table(self, bounds, tex = False, tex_cmd = False, real = True):
        """ Takes (positive) bounds for (r,s) and builds the table of structure
        constants. It also gives two lists with all the values of r and s, to use
        as reference when looking for constants in the table.
        tex = If True, builds a tex table and displays it.
        tex_cmd = Prints the Tex command.
        real = Keeps only the real values of the structure constants."""

        (rbound,sbound) = bounds
        s_end = int(sbound +1/2)

        table = []
        rowp = [1] # Starting point (0,1/2)
        rowm = [1] # Starting point (0,-1/2)

        r_ref = [0] #Initial value for r indices.
        s_ref = [-1/2,1/2] # Initial values for s indices.
        for r in range(1, rbound, 2):

            rowm.insert(0, rowm[0]/self.Shift((-r,-1/2), 'r'))
            rowm.append(rowm[-1]*self.Shift((r,-1/2), 'r'))

            rowp.insert(0, rowp[0]/self.Shift((-r,1/2), 'r'))
            rowp.append(rowp[-1]*self.Shift((r,1/2), 'r'))

            r_ref.insert(0,-r-1)
            r_ref.append(r+1)

        table.extend([rowm, rowp])

        for s in range(1, s_end):

            new_row_up = []
            new_row_down = []

            for j in range(0,rbound+1):
                new_row_down.append(table[1][j]/self.Shift((-rbound+2*j,-s+1/2 ), 's'))
                new_row_up.append(table[-2][j]*self.Shift((-rbound+2*j,s-1/2), 's') )


            table.insert(0,new_row_down)
            table.append(new_row_up)

            s_ref.insert(0, -s-1/2)
            s_ref.append(s+1/2)

        """making the tex table """
        if tex or tex_cmd:

            pretty_csts = []
            for row in table:
                pretty_row = []
                for element in row:
                    if abs(element) < 10**(-4):
                        pretty_row.append(Table.format_exp('{:.8g}'.format(element)))
                    else:
                        pretty_row.append(Table.format_result(element, digits = 12, real = real) )
                pretty_csts.append(pretty_row)

            tbl_st = r'\begin{array}{|c|'

            for i  in r_ref:
                tbl_st += r'c'

            tbl_st += r'| } \hline '+ '\n' + r's \backslash r ' + '\n'

            for r in r_ref:
                tbl_st += ' & {: d}'.format(r)

            tbl_st += r'\\ \hline ' + '\n'

            for i in range(len(s_ref)):
                tbl_st += latex(Rational(s_ref[i])) 
                for j in range(len(r_ref)):
                    tbl_st += r' &' + pretty_csts[i][j]

                tbl_st += r'\\ ' + '\n'
            tbl_st += r'\hline ' + '\n' + r'\end{array}'

            if tex:
                display(Math(tbl_st))

            if tex_cmd:
                print( tbl_st )

        return [table,r_ref,s_ref]


# ## Four-point functions

# In[ ]:


class ND4:
    """Calculates 4 point functions with the spectrum (2Z, Z+1/2) in the s- or t-channel."""

    def __init__(self, ext_fields, L = 5, blocklevel = 12):
        """ext_fields: LIST of four Fields. 
        L = Truncation parmeter of the spectrum
        blocklevel = Truncation paramenter of the conformal blocks.
        """

        self.beta = ext_fields[0].dims[0].charge.get('beta')
        self.ext_fields = ext_fields
        self.L = L
        self.blocklevel = blocklevel

        """Building the internal spectrum and it's truncation. """    
        self.spectrum = Another_Spectrum(self.beta)
        self.int_fields = self.spectrum.truncate(self.L)

        """Building blocks"""
        dims_l = [f.dims[0] for f in self.ext_fields] # Left dims
        dims_r = [f.dims[1] for f in self.ext_fields] # Right dims

        """s-blocks"""
        self.s_blocks = [Block(Nmax = self.blocklevel, dims = dims_l),
                        Block(Nmax = self.blocklevel, dims = dims_r)]
        """t-blocks"""
        self.t_blocks = [Block(t_channel = True, Nmax = self.blocklevel, dims = dims_l),
                        Block(t_channel = True, Nmax = self.blocklevel, dims = dims_r)]

        """Finding maximal indices fot the constants table"""

        r_vals = [f.dims[0].get('degenerate')[0] for f in self.int_fields]
        s_vals = [f.dims[0].get('degenerate')[1] for f in self.int_fields]

        self.rmax = max(r_vals)
        self.smax = max(s_vals)

        self.csts = self.constants() #Table of s- and t-ch constants, and normalization.

    def constants(self):
        """Makes a table with the constants, and references.
        Computes the relative normalization between s and t channel"""

        """s-channel constants and normalization"""
        s_csts = ND_csts(self.ext_fields)

        N = s_csts.relative_norm() # Relative normalization
        s_table = s_csts.make_table((self.rmax, self.smax)) #table of s-constants

        """t-channel constants"""

        t_csts = ND_csts([self.ext_fields[1],self.ext_fields[2],
                        self.ext_fields[3], self.ext_fields[0]])
        # The permutation chosen for t corresponds to the one on the Block class. 

        t_table = t_csts.make_table((self.rmax, self.smax)) #table of t-constants


        return [s_table[0],t_table[0], s_table[1],s_table[2], N] # s-constants, t-constants, ref_r, ref_s

    def blocks(self, z, reduced = True, channel = None):
        """ Computes the needed values of conformal blocks. Computes both channels
        if None, only the s-channel if True, and only the t-channel if False.
        """

        q = mpmath.qfrom(m = z)
        if channel or channel is None:
            block_s_l = BlockNum(block = self.s_blocks[0], q = q)
            block_s_r = BlockNum(block = self.s_blocks[1], q = q.conjugate())
        if not channel or channel is None:
            block_t_l = BlockNum(block = self.t_blocks[0], q = q)
            block_t_r = BlockNum(block = self.t_blocks[1], q = q.conjugate())
        values = []

        for field in self.int_fields:
            value = [field.dims[0].get('degenerate')] #Reference indices.
            if channel or channel is None:
                value.append(block_s_l.value(field.dims[0], reduced) 
                         * block_s_r.value(field.dims[1], reduced))
            if not channel or channel is None:
                value.append(block_t_l.value(field.dims[0], reduced)
                             *block_t_r.value(field.dims[1], reduced))

            values.append(value)

        return values

    def values(self, z, channel = None, norm = True):

        blocks = self.blocks(z = z, channel = channel) #List of indices and block values.

        table = self.csts[:4] # Tables of constants and references

        N = self.csts[4] # t-channel normalization

        channels = [0, 1] if channel is None else [0]
        values = [0 for i in channels]

        for data in blocks:
            """Determine the position of the constant in the table. Same for both channels."""
            index_r = table[2].index(data[0][0]) #Determining r index
            index_s = table[3].index(data[0][1]) #Determining s index

            if channel or channel is None:
                values[0] += table[0][index_s][index_r] * data[1] #s-channel values
            if not channel or channel is None:
                values[-1] += table[1][index_s][index_r] * data[-1] #t-channel values

        if norm:
            if not channel or channel is None:
                values[-1] *= N # multiplying by the t-s normalization

        if channel is not None:
            values = values[0] #If only one channel, returns a number instead of a list.

        return values


# ##  An auxiliary class for building arguments of four-point functions

# In[ ]:


class four_point_args:
    """A class for building arguments of four-point functions. 
    Takes a central charge and a list of fields or paramenters to build them, and creates a 
    list of fields. The method channel allows to permute them.
    """

    def __init__(self, beta, *args):
        """ The first argument is the parameter Beta of the central charge. The rest can be:
        Field: Just add it to a list.
        Number: Creates a diagonal field with this value of Delta, and the given central charge.
        Adds it to the list.
        Tuple (r,s, string): (r,s) are the indices of a field. The string should be 'd' or 'n'
        specifying if the field created is diagonal or not. Adds the field to the list.

        If given two arguments, it duplicates them to create a list of two fields of the type
        1212.
        The created list has the fields ordered in the same way as the arguments. 
        """
        self.args = args
        self.beta = beta
        self.charge = Charge('beta', beta)

        if len(args) not in [2,4]:
            print( 'WARNING: Wrong number of arguments' )

        fields = []

        for a in self.args:

            if isinstance(a, Field):
                """If it gets a Field, it adds it to the list."""
                fields.append(a)
            elif isinstance(a, numbers.Number):
                """If it gets a Number, creates a diagonal field of this Delta."""
                d = Dimension(variable = 'Delta', value = a, charge = self.charge)
                fields.append(Field(dim = d, diagonal = True))
            elif isinstance(a, tuple):
                """If it gets a tuple, it creates a diag or non-diag field with the first
                 two values of the tuple as indices"""
                d = Dimension(variable = 'degenerate', value = (a[0],a[1]), charge = self.charge)
                if a[2] =='d':
                    fields.append(Field(dim = d, diagonal = True))
                elif a[2] =='n':
                    fields.append(Field(dim = d, non_diagonal = True))

        if len(fields) == 2:
            fields.extend(fields)

        self.fields = fields

    @staticmethod
    def permute(sigma, source):
        """Takes a string of numbers between 1 and 4 and a list, and gives a new list
        with elements permuted.
        sigma : A STRING of the type 1234 or 4132 or any permutation. 
        source: The LIST whose elements we want to permute """

        permuted = []
        for i in list(sigma):
            permuted.append(source[int(i)-1])
        return permuted

    def channel(self, perm):
        """Takes a string as an argument, and returns the list of 
        fields according to it. The string can be 's', 't', 'u' or
        any permutation of '1234'.
        """
        channels = {'s' : '1234', 't': '2341', 'u': '1342'}
        # The permutation chosen for t corresponds to the one on the Block class. 

        if perm in channels:
            return four_point_args.permute(channels[perm], self.fields)
        else:
            return four_point_args.permute(perm, self.fields)


# # Tests
# 
# 
# ## Recovering numerical bootstrap results
# 
# We compute four-point structure constants in the case of $c=0$ and four $(0,\frac12)$ fields (two diagonal, two non-diagonal). These can be compared with earlier numerical bootstrap results.

# In[ ]:


if __name__ == '__main__':

    """Computing the values of certain structure constants. Results using the methods 'value'
    and 'make_table' of the class ND_csts.
    """
    c = 0
    C = Charge('c', c)
    beta0 = C.get('beta')

    Champs = four_point_args(beta0, (0,1/2,'d'), (0,1/2,'n'))

    pairs = [(0,1/2), (2,1/2), (0,3/2), (2,3/2), (0,5/2), (4,1/2), (2,5/2), (4,3/2), (0,7/2)]

    Csts = ND_csts(Champs.channel('s'))

    """Results using the method 'value' . """
    for pair in pairs:
        print( str(pair), Csts.value(pair) )


    """Table in tex format"""
    Csts.make_table((4,5/2), tex = True, tex_cmd = True ) 


# ## Crossing symmetry of four-point functions

# In[ ]:


if __name__ == '__main__':

    """Case of c=0"""

    c = 0
    eps = 10**(-9)
    C = Charge('c', c+eps)
    beta0 = C.get('beta')

    Champs = four_point_args(beta0, (0,1/2,'d'), (0,1/2,'n')).channel('s')

    F = ND4(Champs, L = 8, blocklevel = 12)

    for x in range(1,5):
        val_x = F.values(x/10, channel = None)

        print( 'x = %s' % (x/10) )
        print( 's-ch: %s' % val_x[0] )
        print( 't-ch: %s' % val_x[1] )


# In[ ]:


if __name__ == '__main__':

    """Generic charge and dimensions"""

    c = 4.72+0.12*1j
    eps = 0
    C = Charge('c', c+eps)
    beta0 = C.get('beta')

    Champs = four_point_args(beta0, 0.231+0.1432*1j, (2,3/2,'n') ).channel('s')

    F = ND4(Champs, L = 8, blocklevel = 12) 

    for x in range(1,5):
        s_val = F.values(x/10, channel = True)
        t_val = F.values(x/10, channel = False, norm = True)

        print( 'x = %s' % (x/10) )
        print( 's-ch: %s' % s_val.real )
        print( 't-ch: %s' % t_val.real )


# ##  Case of four different fields

# In[ ]:


if __name__ == '__main__':

    """Functions with four different fields, arbitrary dimensions for diagonal fields."""

    c = -0.5432
    eps = 0
    C = Charge('c', c+eps)
    beta0 = C.get('beta')

    Champs = four_point_args(beta0, 0.2314, (4,-7/2 ,'n') , 1.265, (2,1/2,'n')).channel('s')

    F = ND4(Champs, L = 12, blocklevel = 20)

    for x in range(1,10):
        s_val = F.values(x/10, channel = True)
        t_val = F.values(x/10, channel = False, norm = True) # It's possible to turn off t-ch norm.

        print( 'x = %s' % (x/10) )
        print( 's-ch: %s' % s_val.real )
        print( 't-ch: %s' % t_val.real )

