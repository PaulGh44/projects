#!/usr/bin/env python
# coding: utf-8

# # Conformal field theory
# 
# We define basic data of conformal field theory: the central charge, the conformal dimension, and fields.

# In[ ]:


from __future__ import division, print_function

from sympy import Rational, I, sqrt, latex, symbols
from IPython.display import display, Math
import numpy

from Auxiliary_classes import LatexTable, numerical


# ## Central charge
# 
# The central charge $c$ can alternatively be parametrized by the variables $Q, b, B, \beta, S$ such that
# $$
# c = 1 + 6Q^2,\ Q = b+\frac{1}{b},\ B=b^2,\ \beta = ib,\ S=\frac{c-1}{24}=\frac{Q^2}{4}=\frac12+\frac14(B+1/B)
# $$
# In the case of the $M_{p,q}$ minimal model we have
# $$
# B = -\frac{q}{p}
# $$

# In[ ]:


class Charge:
    """ The central charge with all possible notations. """

    def __init__(self, variable = 'b', value = None):
        """ Needs a string, corresponding to a variable, and a value, which by default
        is a symbolic variable.  We then compute the values of all variables. 
        """

        if value is None:
            if variable == 'minimal':
                value = symbols(['p', 'q'])
            else:
                value = symbols(variable)     # Giving a symbolic value by default.                           

        self.names = [['c', 'c'], ['Q', 'Q'], ['b', 'b'], ['beta', r'\beta'], ['B', 'B'], ['S', 'S']]
        """ Class variables tend to get modified when used, so let's not use them! """

        for name in self.names:
            name += [Charge.convert(value, variable, name[0])]

        if variable == 'minimal': 
            self.names = [['minimal', '(p,q)', tuple(value)]] + self.names


    @staticmethod        
    def convert(x, var1, var2):
        """ Given a value x and two names of variables, returns the value of var2 if var1 = x. 
        Beware that relations such as B = b**2 are not guaranteed to hold when both are computed
        from c. 
        """
        if var1 == var2:
            result = x
        else:
            result = {
                ('c', 'Q'): lambda x: sqrt((x - 1) / Rational(6)),
                ('c', 'b'): lambda x: - sqrt((x - 1) / Rational(24)) + sqrt((x - 25) / Rational(24)),
                ('c', 'beta'): lambda x: - sqrt((1 - x) / Rational(24)) + sqrt((25 - x) / Rational(24)),
                ('c', 'B'): lambda x: (x - 13 - sqrt(x - 1) * sqrt(x - 25)) / Rational(12),
                ('c', 'S'): lambda x: (x - 1) / Rational(24),
                ('Q', 'c'): lambda x: 1 + 6 * x**2,
                ('Q', 'b'): lambda x: (x + sqrt(x**2 - 4)) / Rational(2),
                ('Q', 'beta'): lambda x: I * (x + sqrt(x**2 - 4)) / Rational(2),
                ('Q', 'B'): lambda x: (x**2 - 2 + x * sqrt(x**2 - 4)) / Rational(2),
                ('Q', 'S'): lambda x: x**2 / Rational(4),
                ('b', 'c'): lambda x: 13 + 6 * x**2 + Rational(6) / x**2,
                ('b', 'Q'): lambda x: x + Rational(1) / x,
                ('b', 'beta'): lambda x: I * x,
                ('b', 'B'): lambda x: x**2,
                ('b', 'S'): lambda x: (x + Rational(1) / x)**2 / Rational(4),
                ('beta', 'c'): lambda x: 13 - 6 * x**2 - Rational(6) / x**2,
                ('beta', 'Q'): lambda x: - I * (x - Rational(1) / x),
                ('beta', 'b'): lambda x: - I * x,
                ('beta', 'B'): lambda x: - x**2,
                ('beta', 'S'): lambda x: - (x - Rational(1) / x)**2 / Rational(4),
                ('B', 'c'): lambda x: 13 + 6 * x + Rational(6) / x,
                ('B', 'Q'): lambda x: sqrt(x) + Rational(1) / sqrt(x),
                ('B', 'b'): lambda x: sqrt(x),
                ('B', 'beta'): lambda x: sqrt(-x),
                ('B', 'S'): lambda x: (x + 2 + Rational(1) / x) / Rational(4), 
                ('S', 'c'): lambda x: 24 * x + 1, 
                ('S', 'Q'): lambda x: 2 * sqrt(x),
                ('S', 'b'): lambda x: sqrt(x) + sqrt(x - 1),
                ('S', 'beta'): lambda x: sqrt(-x) + sqrt(1 - x),
                ('S', 'B'): lambda x: 2 * x - 1 + 2 * sqrt(x * (x - 1)),
                ('minimal', 'c'): lambda x: 13 - Rational(6) / x[0] * x[1] - Rational(6) / x[1] * x[0],
                ('minimal', 'Q'): lambda x: sqrt(-Rational(1) / x[0] * x[1]) - sqrt(-Rational(1) / x[1] * x[0]),
                ('minimal', 'b'): lambda x: sqrt(-Rational(1) / x[0] * x[1]),
                ('minimal', 'beta'): lambda x: sqrt(Rational(1) / x[0] * x[1]),
                ('minimal', 'B'): lambda x: -Rational(1) / x[0] * x[1],
                ('minimal', 'S'): lambda x: (2 - Rational(1) / x[0] * x[1] - Rational(1) / x[1] * x[0]) / Rational(4)
                    }[(var1, var2)](x)
        return result

    def get(self, variable, num = True):
        """ Returns the value of a given variable. By default this value is a float or complex. """
        result = None
        for name in self.names:
            if variable == name[0]:
                result = name[2]
                if num and name[0] != 'minimal':
                    result = numerical(result)
        return result  

    def display(self):
        """ Prints the relevant data. """
        LatexTable(self.names, [1, 2])

    def KacTable(self, show = True):
        """ Displays the Kac table, if the model is minimal. """
        result = None
        if self.names[0][0] == 'minimal':
            (p,q) = self.names[0][2]
            if isinstance(p,int) and isinstance(q, int) and p>1 and q>1:           
                string = r"\begin{array}{c|"
                for r in range(1,p):
                    string += "c"
                string += r"} "
                string2 = ""
                for s in range(1,q):
                    substring = latex(s)
                    for r in range(1,p):
                        substring += " & " + latex(Dimension('degenerate', [r, s], self).get('Delta', False))
                    string2 = substring + r" \\ " + string2
                string += string2 + r" \hline"         
                for r in range(1,p):
                    string += " & " + latex(r)
                string += r" \end{array}"   
                if show:
                    display(Math(string))
                result = string
        return result    

if __name__ == '__main__':

    Charge().display()
    Charge('minimal').display()
    Charge('minimal', [3,4]).display()
    Charge('c').display()
    Charge('Q').display()
    Charge('b').display()
    Charge('S').display()
    Charge('beta').display()
    Charge('B').display()
    Charge('b', .3).display()
    Charge('b', .3 - .4*1j).display()


# ## Conformal dimension
# 
# The conformal dimension $\Delta$ of a primary field can alternatively be parametrized by the variables $\delta, \alpha, A, P$ such that
# $$
# \Delta = \delta + \frac{Q^2}{4} = \alpha(Q-\alpha), \ \delta = P^2 = -\frac{A^2}{b^2}
# $$
# We call degenerate a dimension of the type
# $$
# \alpha = \frac{Q - r b - sb^{-1}}{2}
# $$
# where $r$ and $s$ are however arbitrary. We moreover say that we have a null vector if $r,s\in \mathbb{Z}$ and $rs>0$. 

# In[ ]:


class Dimension:
    """ The conformal dimension with all possible notations. """

    def __init__(self, variable = 'alpha', value = None, charge = Charge()):
        """ Needs a charge, a string, corresponding to a variable, and a value, which by default
        is a symbolic variable. We then compute the values of all variables. 
        """

        self.charge = charge
        self.S = charge.get('S', False)
        self.beta = charge.get('beta', False)
        self.Q = charge.get('Q', False)
        self.b = charge.get('b', False)
        self.B = charge.get('B', False)

        self.names = [['Delta', r'\Delta'], ['delta', r'\delta'], ['alpha', r'\alpha'], ['A', 'A'], ['P', 'P']]
        """ Class variables tend to get modified when used, so let's not use them! """  

        if value is None:
            if variable == 'degenerate':
                value = symbols(['r', 's'])
            else:
                for i in range(len(self.names)):
                    if variable == self.names[i][0]:
                        value = symbols(self.names[i][1]) # Giving a symbolic value by default.   

        for name in self.names:
            name += [self.convert(value, variable, name[0])]        

        self.isDegenerate = False   
        self.hasNullVector = False
        if variable == 'degenerate':
            self.isDegenerate = True
            self.names = [['degenerate', '(r,s)', tuple(value)]] + self.names  
            (r, s) = value
            self.hasNullVector = isinstance(r, int) and isinstance(s, int) and r*s > 0

    def convert(self, x, var1, var2):
        """ Given a value x and two names of variables, returns the value of var2 if var1 = x. """
        if var1 == var2:
            result = x
        else:
            result = {
                ('Delta', 'delta'): lambda x: x - self.S,
                ('Delta', 'alpha'): lambda x: self.Q / Rational(2) + sqrt(self.S - x),
                ('Delta', 'A'): lambda x: sqrt(self.B * (self.S - x)),
                ('Delta', 'P'): lambda x: sqrt(x - self.S),
                ('delta', 'Delta'): lambda x: x + self.S,
                ('delta', 'alpha'): lambda x: self.Q / Rational(2) + sqrt(-x),
                ('delta', 'A'): lambda x: sqrt(-self.B * x),
                ('delta', 'P'): lambda x: sqrt(x),
                ('alpha', 'Delta'): lambda x: x * (self.Q - x),
                ('alpha', 'delta'): lambda x: -(x - self.Q / Rational(2))**2,
                ('alpha', 'A'): lambda x: (1 + self.B)/Rational(2) - self.b * x,
                ('alpha', 'P'): lambda x: I * (self.Q / Rational(2) - x),
                ('A', 'Delta'): lambda x: self.S - x**2 * Rational(1) / self.B,
                ('A', 'delta'): lambda x: -x**2 * Rational(1) / self.B,
                ('A', 'alpha'): lambda x: self.Q / Rational(2) - x * Rational(1) / self.b,
                ('A', 'P'): lambda x: x * I * Rational(1) / self.b,
                ('P', 'Delta'): lambda x: self.S + x**2,
                ('P', 'delta'): lambda x: x**2,
                ('P', 'alpha'): lambda x: self.Q / Rational(2) + I * x, 
                ('P', 'A'): lambda x: -I * self.b * x,
                ('degenerate', 'Delta'): lambda x: self.S - (x[0]**2 * self.B + 2 * x[0] * x[1] 
                                                    + x[1]**2 * Rational(1) / self.B) / Rational(4),
                ('degenerate', 'delta'): lambda x: -(x[0]**2 * self.B + 2 * x[0] * x[1] 
                                                    + x[1]**2 * Rational(1) / self.B) / Rational(4),
                ('degenerate', 'alpha'): lambda x: -((x[0] - 1) * self.b + 
                                                     (x[1] - 1) * Rational(1) / self.b) / Rational(2),
                ('degenerate', 'A'): lambda x: (x[0] * self.B + x[1]) / Rational(2),
                ('degenerate', 'P'): lambda x: (x[0] * self.b + x[1] * Rational(1) / self.b) * I / Rational(2),
                    }[(var1, var2)](x)
        return result  

    def get(self, variable, num = True):
        """ Returns the value of a given variable. By default this value is a float or complex. """
        result = None
        for name in self.names:
            if variable == name[0]:
                result = name[2]
                if num and name[0] != 'degenerate':
                    result = numerical(result)
        return result  

    def fuse(self, otherField):
        """ Returns the list of dimensions of fused fields, assuming the model is not minimal. """
        if self.isDegenerate and otherField.isDegenerate:
            (r1, s1) = self.get('degenerate')
            (r2, s2) = otherField.get('degenerate')
            dims = []
            for r in range(abs(r1 - r2) + 1, r1 + r2 + 1, 2):
                for s in range(abs(s1 - s2) + 1, s1 + s2 + 1, 2):
                    dims.append(Dimension('degenerate', [r, s], self.charge))
            return dims
        if self.isDegenerate:
            (r, s) = self.get('degenerate')
            alpha = otherField.get('alpha', False)
        if otherField.isDegenerate:
            (r, s) = otherField.get('degenerate')
            alpha = self.get('alpha', False)
        if self.isDegenerate or otherField.isDegenerate:    
            ars = Dimension('degenerate', [r, s], self.charge).get('alpha', False)
            dims = []
            b = self.charge.get('b', False)
            for i in range(r):
                for j in range(s):
                    dims.append(Dimension('alpha', alpha + ars + i*b + j/b))
            return dims 
        return None  

    def hasForNullVector(self, otherField):
        """ Says whether the field is degenerate, and the other field is its null vector. """
        answer = False
        if self.hasNullVector and isinstance(otherField, Dimension) and otherField.isDegenerate:
            (r1, s1) = self.get('degenerate')
            (r2, s2) = otherField.get('degenerate')
            if r1 == r2 and s1 == -s2 or r1 == -r2 and s1 == s2:
                answer = True
        return answer

    def display(self):
        """ Prints the relevant data. """
        LatexTable(self.names, [1, 2])


if __name__ == '__main__':

    Charge('minimal', [9, 14]).KacTable()
    Dimension().display()
    Dimension('degenerate').display()
    Dimension('degenerate', [2, 1]).display()
    Dimension('Delta').display()
    Dimension('delta').display()
    Dimension('A').display()
    Dimension('P').display()
    field1 = Dimension('degenerate', [2, 3])
    field2 = Dimension('degenerate', [3, 5])
    field3 = Dimension('alpha')
    display(Math(latex([dim.get('degenerate') for dim in field1.fuse(field2)])))
    display(Math(latex([dim.get('alpha', False) for dim in field1.fuse(field3)])))
    display(Math(latex([dim.get('alpha', False) for dim in field3.fuse(field2)])))
    Dimension('delta', .3*1j + .1, Charge('minimal', [3, 2])).display()
    dim = Dimension('degenerate', (1, 1/2), Charge('beta', .3))
    print( dim.isDegenerate )
    print( dim.hasNullVector )


# In[ ]:


if __name__ == '__main__':

    field1 = Dimension('degenerate', [1, -2])
    field2 = Dimension('degenerate', [-1, -2])
    print( field1.hasForNullVector(field2) )
    print( field2.hasForNullVector(field1) )


# ## Fields
# 
# There are two types of fields: diagonal and non-diagonal. Fields have a pair of dimensions, which are equal if the field is diagonal. The spin is the difference of the two dimensions. Bosonic fields have integer spins, fermionic fields have half-integer spins. 
# 
# Particularly useful non-diagonal fields may be built from pairs of degenerate Dimensions of the type $((r,s) , (r, -s))$ with $r,s,rs\in\frac12 \mathbb{Z}$. The spin of such a field is $rs$. We call them non-diagonal degenerate fields. For such fields we define the signs $((-1)^{2r},(-1)^{2s}) \in \{1,-1\}^2$. For any other field, the signs are $(1,1)$.

# In[ ]:


class Field:
    """ A conformal field. """

    def __init__(self, dim, diagonal = False, non_diagonal = False, spin = 0):
        """ Give one Dimension object, and declare whether the field is diagonal, whether it is 
        a non-diagonal degenerate field, or what its spin is. """

        self.dims = [dim]
        self.charge = dim.charge
        self.signs = (1, 1)
        self.non_diagonal = non_diagonal  # This boolean differs from (self.spin == 0) !
        if diagonal:
            self.dims.append(dim)
            self.spin = 0
        elif non_diagonal:
            (r, s) = dim.get('degenerate')
            self.dims.append(Dimension('degenerate', (r, -s), self.charge))
            self.spin = Rational(r*s)
            self.signs = (Field.sign(r), Field.sign(s))
        else:
            self.spin = Rational(spin)
            self.dims.append(Dimension('Delta', dim.get('Delta') + spin, self.charge))
        self.dimensions = [self.dims[0].get('Delta'), self.dims[1].get('Delta')]
        self.total_dimension = self.dimensions[0] + self.dimensions[1]

    @staticmethod
    def sign(r):
        """ Computes (-1)^2r . """
        return -1 if 2*r %2 == 1 else 1

    def display(self, show_deg = True, show_spin = True, show_dims = True, show_signs = True): 
        """ Can display the degenerate indices, the spin, the dimensions. """
        string = ""
        if show_deg:
            string += (str(self.dims[0].get('degenerate')) + " " 
                       + str(self.dims[1].get('degenerate')))
        if show_spin:
            string += ' spin ' + str(self.spin)
        if show_dims:
            string += ' ' + str(self.dimensions)
        if show_signs:
            string += ' ' + str(self.signs)
        print( string )

    @staticmethod
    def sort(fields_list, erase_dims = True):
        """ Sort the list by increasing (real parts of) total dimensions. """

        fields = sorted(fields_list, key = lambda field: field.total_dimension.real)
        if not erase_dims:
            fields = [[field, field.total_dimension.real] for field in fields]
        return fields


# In[ ]:


if __name__ == '__main__':

    charge = Charge('c', 0)
    fields = []
    for m in range(3):
        for n in range(5):
            dim = Dimension('degenerate', (m, Rational(n/2)), charge)
            field = Field(dim, non_diagonal = True)
            fields.append(field)
    fields = Field.sort(fields)
    for field in fields:
        field.display()


# In[ ]:




