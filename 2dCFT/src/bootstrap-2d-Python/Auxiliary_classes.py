#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division
import scipy
from scipy.integrate import quad
from sympy import latex, symbols, Matrix, Rational, exp
from IPython.display import display, Math
import copy, re
import cmath


# # Auxiliary classes

# ## Complex integration

# In[ ]:


def complex_quadrature(func, a, b, **kwargs):
    """ This method is copied from StackOverflow. It is a version of 'quad' which works with complex numbers. 
    We add the option full_output = True in order to eliminate the warnings about maximum numbers of 
    subdivisions.
    """
    def real_func(x):
        return scipy.real(func(x))
    def imag_func(x):
        return scipy.imag(func(x))
    real_integral = quad(real_func, a, b, full_output = True, **kwargs)
    imag_integral = quad(imag_func, a, b, full_output = True, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])


# ## Conversion to complex numbers or floats

# In[ ]:


def numerical(x):
    x = complex(x)
    if x.imag == 0:
        x = x.real
    return x


# ## Displaying tables 

# This class is for building nice Latex tables in order to display results. 

# In[ ]:


class LatexTable: 
    """ A table built from an array, whose columns will become the lines of the table. """

    def __init__(self, array, columns = None, show = True):
        """ Needs the array itself, and optionally the list of columns to be displayed. """

        n = len(array)
        if columns == None:
            columns = range(len(array[0]))

        """ Beginning the table. """
        string = r"\begin{array}{|"  
        for i in range(n):
            string += "c|"
        string += "} \hline " 

        """ Adding lines. """
        for j in columns:
            line = ""
            for i in range(n):
                line += latex(array[i][j])
                if i < n - 1:
                    line += " & "             
            string += line
            string += r" \\ \hline "

        """ Ending the table. """    
        string += r" \end{array} "

        if show:
            display(Math(string))              
        self.string = string    


if __name__ == '__main__':

    array = [[Rational(i,j) for i in range(4)] for j in range(1, 7)]
    LatexTable(array)


# ## Formal variables 

# This class is for creating formal variables, vectors and matrices.

# In[ ]:


class Formal:
    """ A collection of static methods, with no constructor. """

    @staticmethod
    def var(string, index1 = None, index2 = None):
        """ Builds formal variables with at most two indices. """
        var = symbols(string)
        if index1 != None:
            var = symbols(string + "_{" + latex(index1) + "}")
            if index2 != None:
                var = symbols(string + "_{" + latex(index1) + "\," + latex(index2) + "}")    
        return var

    @staticmethod
    def vector(string, N):
        """ Builds a vector of size N, whose elements are formal variables. """
        return [Formal.var(string, i) for i in range(1, N+1)]

    @staticmethod
    def matrix(string, N):
        """ Builds a matrix of size N, whose elements are formal variables. """
        return [[Formal.var(string, i, j) for j in range(1, N+1)] for i in range(1, N+1)]  

    @staticmethod
    def display(matrix):
        """ Displays formal vectors and matrices. """
        display(Math(latex(Matrix(matrix))))


if __name__ == '__main__':

    Formal.display(Formal.matrix('M', 5))
    Formal.display(Formal.vector('V', 7))


# ## Formal series

# This class is for dealing with formal series, including polynomials.
# 
# In particular we compute the following series, which is related to elliptic functions:
# $$
# q = \frac{x}{16} \exp  \frac{\sum_{n=0}^\infty \left(\frac{(\frac12)_n}{n!}\right)^2 \sum_{k=1}^n \frac{2}{k(2k-1)} x^n}
# { \sum_{n=0}^\infty \left(\frac{(\frac12)_n}{n!}\right)^2 x^n }
# $$
# We also recall
# $$
# \theta_3(q) = \sum_{n\in\mathbb{Z}} q^{n^2} 
# $$

# In[ ]:


class Series:
    """A power series, viewed as a finite collection of numbers. Beware that coefficients
    are treated as formal expressions: only the sum of the series, which is given by the 
    method 'compute', is numerical. """

    formal = False    
    """ A formal series has formal coefficients i.e. Sympy expressions. A series that is 
    not formal has numerical coefficients. """

    def __init__(self, collection = None, exact = False):
        """ Needs an existing collection. An exact series is a polynomial. """
        self.exact = exact
        if collection is None:
            self.value = []
        else:
            self.value = collection

    @staticmethod
    def rational(x, y = 1):
        """ A method to deal with fractions as numerical or not, depending on whether 
        Series are formal. """
        if Series.formal:
            return Rational(x, y)
        else:
            return x/y

    @staticmethod
    def scalar_exp(x):
        if Series.formal:
            return exp(x)
        else:
            return cmath.exp(x)

    @staticmethod
    def cleanSigns(string):
        """ This method removes plus signs at the end, and contracts double signs. """
        eliminate = ["-", "+"]
        replace = [[r"\+\-", "-"], [r"\+\+", "+"], [r"\-\-", "+"], [r"\-\+", "-"]]
        for char in eliminate:
            if string[-1] == char:
                string = string[:-1]
        for pair in replace:
            string = re.sub(pair[0], pair[1], string)
        return string

    @staticmethod
    def varpower(var, n):
        """ Writes x^n. """
        if n == 0:
            return "1"
        if n == 1:
            return var
        return var + r"^{" + str(n) + r"}"

    def change_terms(self, f):
        """ Applies a function to all terms. For example, take the real part. """
        self.value = [f(val) for val in self.value]

    def display(self, brackets = False, var = "x"):
        """ Displays the series. """
        string = ""
        n = self.length()
        for i in range(n):
            val = self.value[i]
            if val != 0:
                if brackets:
                    string += r"\left(" + latex(val) + r"\right)"
                else:
                    string += latex(val)
                    if i != 0:
                        if abs(val) == 1:
                            string = string[:-1]
                        else:    
                            string += r"\ "
                if i != 0:
                    string += Series.varpower(var, i)                        
                string += "+"
        if self.exact:
            if string == "":
                string = "0"
        else:
            string += r"O(" + Series.varpower(var, n) + ")"
        string = Series.cleanSigns(string)    
        display(Math(string))    

    def compute(self, var, num = True):
        """ Sums the series at a value of x. """
        total = 0 
        power = 1
        for i in range(self.length()):
            val = self.value[i]
            total += val * power
            power *= var
        if num:
            total = numerical(total)
        return total    

    def get(self, i):
        """ Returns an element if it exists, None otherwise. """
        result = None
        if self.exact and i >= self.length():
            result = 0 
        if i < self.length():
            result = self.value[i]
        return result

    def append(self, x):
        """ Appends one element. """
        self.value.append(x)

    def length(self):
        """ Measures length. """
        return len(self.value)

    def truncate(self, length):
        """ Truncates the series at a given length, producing a non-exact series. """  
        diff = length - self.length()
        if diff < 0:
            del self.value[length:]
        elif self.exact:    
            for i in range(diff):
                self.append(0)
        self.exact = False 
        return self

    def product(self, other, truncate = False):
        """ Multiplies two Series and produces a Series of the expected length, whether we have exact series
        or not, and whether there are leading zeros or not. Switching on the truncate option allows to ignore
        leading zeros, and therefore to produce shorter series.
        """
        product = Series()
        length = self.length() + other.length()
        if self.exact and other.exact:
            product.exact = True
        i = 0 
        for i in range(length):
            term = 0
            for j in range(i+1):
                x = self.get(j)
                y = other.get(i-j)
                if (x == 0 or y == 0) and not truncate:
                    continue
                if x is None or y is None:                    
                    return product
                term += x * y
            product.append(term)
        return product

    def add(self, other):
        """ Adds two series. """
        total = Series()
        length = max(self.length() , other.length())
        if self.exact and other.exact:
            total.exact = True
        for i in range(length):
            x = self.get(i)
            y = other.get(i)
            if x is None or y is None:
                return total
            total.append(x + y)
        return total    

    def multiply(self, scalar):
        """ Multiplies with a scalar. """
        result = Series([], self.exact)
        for i in range(self.length()):
            result.append(scalar * self.get(i))
        return result

    @staticmethod
    def monomial(exponent):
        """ Produces an exact monomial series. """
        result = Series([], True)
        for i in range(exponent):
            result.append(0)
        result.append(1)
        return result

    @staticmethod
    def binomial(length, exponent):
        """ Produces the series (1-x)**exponent. """
        power = Series([1])
        term = 1
        for i in range(length - 1):
            term *= (i - exponent)*Series.rational(1, i+1)
            power.append(term)
        return power

    @staticmethod
    def exponential(length):
        """ Produces the exponential series. """
        exp = Series([1])
        term = 1 
        for i in range(length - 1):
            term *= Series.rational(1, i+1)
            exp.append(term)
        return exp    

    @staticmethod
    def hypergeometric(length, a, b, c):
        """ Produces the hypergeometric series. """
        hyp = Series([1])
        term = 1
        for i in range(length - 1):
            term *= Series.rational(1, i+1) / (c + i) * (a + i) * (b + i) 
            hyp.append(term)
        return hyp

    @staticmethod
    def qExp(length):
        """ Computing the expression in the exponent of q as a series in x. """
        num = Series([])
        denom = Series([])
        prefactor = 1
        term = 0
        for i in range(length):
            num.append(prefactor * term)
            denom.append(prefactor)
            prefactor *= ((Series.rational(1, 2) + i)/(1 + i))**2
            term += Series.rational(2, i+1)/(2*i + 1)
        return num.product(denom.power(-1))

    @staticmethod
    def q(length):
        """ Computing q itself. """
        qExp = Series.qExp(length - 1)
        return qExp.exp().product(Series.monomial(1)).multiply(Series.rational(1, 16))   

    @staticmethod
    def theta(series):
        """ Computing theta3 of a series, assuming that series has a vanishing first term. """
        result = Series([1], True)
        n = 1
        aux = Series([1], True)
        while n**2 < series.length():
            for i in range(2*n - 1):
                aux = aux.product(series, True)
            result = result.add(aux.multiply(2))
            n += 1
        result.exact = False
        return result        

    def derivative(self):
        """ Computes the derivative. """
        der = Series([], self.exact)
        for i in range(1, self.length()):
            der.append(i * self.get(i))
        return der

    def integral(self):
        """ Computes the integral. """
        integral = Series([0], self.exact)
        for i in range(self.length()):
            integral.append(self.get(i) * Series.rational(1, i+1))
        return integral

    def of(self, other):
        """ Composes two series. """
        total = Series([], True)
        aux = copy.deepcopy(other) 
        if not self.exact:
            length = 1
            for i in range(aux.length()):
                if other.value[i] != 0:
                    length = i * self.length()
                    total = Series([0 for i in range(length)])
                    aux.truncate(length)
                    break   
        prod = Series([1], True)  
        for i in range(self.length()):  
            total = total.add(prod.multiply(self.get(i)))
            prod = prod.product(aux, True)
        return total    

    def exp(self, length = 1):
        """ Computes the exponential of the series. If the series is exact,
        we need to specify the desired length of the resulting non-exact series. 
        """
        if not self.exact:
            length = self.length()
        prefactor = Series.scalar_exp(self.get(0))
        series = copy.deepcopy(self)
        series.value[0] = 0
        return Series.exponential(length).of(series).multiply(prefactor)

    def power(self, exponent, length = 1):
        """ Computes powers of the series, assuming the first term is nonzero. If the series is exact,
        we need to specify the desired length of the resulting non-exact series. 
        """        
        if not self.exact:
            length = self.length()
        a = self.get(0)
        if isinstance(exponent, int) and exponent < 0:
            prefactor = Series.rational(1) / a**(-exponent)
        else:    
            prefactor = a**exponent    
        series = copy.deepcopy(self)
        series.value[0] = 0
        series = series.multiply(-Series.rational(1) / a)
        return Series.binomial(length, exponent).of(series).multiply(prefactor)

if __name__ == '__main__':

    Series.formal = True
    s1 = Series([0,1,2,3,4,5])
    s2 = Series([4,5], True)
    s2.power(3, 8).display()
    s3 = Series([0,1,1])
    s3.product(s3).display()
    s3.product(s3, True).display()
    s5 = Series([1, -1], True)
    s5.power(-1, 7).display()
    s6 = Series([0, 1, Rational(1,2), Rational(1,3), Rational(1,4), Rational(1,5), Rational(1,6)])
    s6.exp().power(-1).display()
    s1.of(Series.monomial(2)).display()
    s8 = Series([1, 1, 1, 1, 1, 1])
    s8.power(-1).display()
    s9 = Series.exponential(7)
    s9.display()
    Series.monomial(2).exp(5).display()
    Series.hypergeometric(5, symbols('a'), symbols('b'), symbols('c')).display()
    Series.qExp(3).display()
    Series.q(3).display()


# In[ ]:


if __name__ == '__main__':

    V = Formal.vector('V', 7)
    s = Series(V)
    s.display()


# In[ ]:




