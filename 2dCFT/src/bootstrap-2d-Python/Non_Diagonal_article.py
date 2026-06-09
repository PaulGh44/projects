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
from Non_Diagonal_Shifts import *
import matplotlib.pyplot as plt


# # Numerical tests of crossing symmetry
# 
# In this notebook we produce the tables of Section 4.2 of the article "The analytic bootstrap equations of non-diagonal two-dimensional CFT", by S. Migliaccio and S. Ribault.
# 
# ## Comparison with numerical bootstrap
# We generate a table comparing the four-point structure constants obtained analytically with those of the numerical bootstrap, in the case of the four-point function $Z_0 = \left\langle V^D_{P_{(0,\frac{1}{2})}} V^N_{(0,\frac{1}{2})} V^D_{P_{(0,\frac{1}{2})}} V^N_{(0,\frac{1}{2})} \right\rangle$ .

# In[ ]:


""" Select central charge and truncation parameters. """

c = 0 # Values of the central charge.
eps = 10**(-9) # A small shift to allow the computation of conformal blocks.

charge = Charge('c', c + eps)
beta0 = charge.get('beta')

N, L, bl = (8, 5.5, 30) #Truncation parameters. 

# N: Number of sample sets for the numerical bootstrap.
# L: Truncation parameter for the spectrum.
# bl: Truncation paramenter of the conformal blocks. 

"""Generating numerical bootstrap data"""
t0 = time.clock()
spectrum = Another_Spectrum(beta = beta0) # spectrum 2Z,Z+1/2 for num bootstrap.

num_bootstrap = Struct_Csts_Stats(spectrum = spectrum, Nsamples = N, L = L, blocklevel = bl)

dur = time.clock() - t0

print('Numerical results in %s seconds.' % Table.format_time(dur))

rbound = max([indices[0] for indices in num_bootstrap.indices]) # maximal r index
sbound = max([indices[1] for indices in num_bootstrap.indices]) # maximal s index

""" Generating analytic results"""
t0 = time.clock()

Ext_fields = four_point_args(beta0, (0,1/2,'d'), (0,1/2,'n')).fields # Building fields.
Z = ND_csts(Ext_fields) # Building constants object.

analytic_csts = Z.make_table((rbound, sbound))
dur = time.clock() - t0

print('Analytic results in %s seconds.' % Table.format_time(dur))

"""Computing relative differences"""
compared_data = []
for i in range(len(num_bootstrap.indices)):
    if num_bootstrap.indices[i][1] > 0: #keepin only positive s
        k = analytic_csts[1].index(num_bootstrap.indices[i][0])
        j = analytic_csts[2].index(num_bootstrap.indices[i][1])

        an_c = analytic_csts[0][j][k]

        num_c = num_bootstrap.mean_cst[i]
        num_var = num_bootstrap.cvar[i]

        d = 2 * abs( (an_c - num_c)/(an_c + num_c) )

        row_values = [num_bootstrap.indices[i], num_c, num_var, an_c, d ]

        compared_data.append(row_values)

"""Truncating table"""
j = None
for row in compared_data:
    if row[0] == (0,7/2): # We choose to keep constants only up to (0,7/2).
        j = compared_data.index(row)
        break
if j is not None:
    compared_data = compared_data[:j+1]

"""Creating list with formatted results for the table"""
tbl_contents = []
for row in compared_data:
    pretty_indices = (Rational(row[0][0]), Rational(row[0][1])) #formattin indices

    if abs(row[1]) < 10**(-4):
        pretty_num_csts = Table.format_exp('{:.5g}'.format(row[1].real) )
    else:
        pretty_num_csts = Table.format_result(row[1], digits = 10, real = True)  # formatting numerical csts.

    pretty_cvar = Table.format_precision(row[2]) # formatting numeric variations

    if abs(row[3]) < 10**(-4):
        pretty_a_csts = Table.format_exp('{:.5g}'.format(row[3]) )
    else:
        pretty_a_csts = Table.format_result(row[3], digits = 10, real = True)  # formatting analytic csts.

    pretty_diffs = Table.format_precision(row[4])

    tbl_contents.append([ pretty_indices, pretty_num_csts, pretty_cvar, 
                         pretty_a_csts, pretty_diffs  ])


"""Make Tex table."""

header = (r'\begin{array}{|c|cr|c|c|} \hline ' + '\n' + r' & \text{Numerical bootstrap} ' 
          + r'& & \text{Analytic bootstrap} & \text{Relative} \\' + '\n'
         + r'(r,s) & D_{(r,s)} & c_{(r,s)} & D_{(r,s)} & \text{difference} \\ \hline '+ '\n')

contents = ''
for row in tbl_contents:
    contents += latex(row[0])
    for i in range(1,len(row)):
        contents += r'& ' + row[i]

    contents += r' \\ ' + '\n'

ending = r'\hline' + '\n' + r'\end{array}'

tbl_str = header + contents + ending
display(Math(tbl_str))

parameters = (r'c = %s, \epsilon = %s \\'%(c, eps) + '\n ' 
              + 'Nsamples = %s, L = %s, bl = %s . \\'%(N, L, bl) + '\n')
print(parameters + tbl_str)


# ## Crossing symmetry tests
# 
# We compute four-point functions of the form 
# 
# \begin{align}
# \left\langle V^D_{P_1} \, V^N_{(r_2,s_2)} \, V^D_{P_1} \, V^N_{(r_2,s_2)} \right\rangle\ ,
# \end{align}
# 
# and display tables of the $s$- and $t$-channel values. 
# 
# The code below builds one table at a time. For use:
# 
# * Run a cell to generate the central charge and the corresponding fields. There are three different cells, corresponding to the different values used in the article. 
# 
# * Run the cell to generate the four-point function and compute its values.
# 
# * Run the table builder. 
# 
# 
# ###  Generate charge and fields

# In[ ]:


## Select central charge and truncation parameters. 
c = 0
eps = 10**(-9)
charge = Charge('c', c + eps)
beta0 = charge.get('beta')

L_s = 8  #spectrum truncation order
bl = 25 # blocks truncation order

Ext_fields = four_point_args(beta0, (0,1/2,'d'), (0,1/2,'n')).fields


# In[ ]:


c = 0.7513
eps = 0
charge = Charge('c', c + eps)
beta0 = charge.get('beta')

L_s = 8  #spectrum truncation order
bl = 25 # blocks truncation order

"""Generating fields. Change commented line to get both cases."""
Ext_fields = four_point_args(beta0, 0.0731, (0,3/2,'n')).fields
#Ext_fields = four_point_args(beta0, 0.0731, (2,1/2,'n')).fields


# In[ ]:


c = 4.72+0.12*1j
eps = 0
charge = Charge('c', c + eps)
beta0 = charge.get('beta')

L_s = 8  #spectrum truncation order
bl = 25 # blocks truncation order

"""Generating fields. Change commented line to get both cases."""
Ext_fields = four_point_args(beta0, 0.231+0.1432*1j, (0,1/2,'n')).fields
#Ext_fields = four_point_args(beta0, 0.231+0.1432*1j, (2,3/2,'n')).fields


# ### Computing function values

# In[ ]:


"""Generating functions and computing their values."""

"""Generating four-point function"""
t0 = time.clock()
Z = ND4(Ext_fields, L = L_s, blocklevel = bl)
dur = time.clock()-t0
print('Function created in %s seconds.' %Table.format_time(dur))

"""Computing s-and-t-channel values."""

points = [0.01, 0.03, 0.1, 0.2, 0.4] # points to compute the functions.

st_vals = [Z.values(z = p) for p in points] # s-ch, t-ch for each point.

"""Computing relative differences"""
diffs = []

for pair in st_vals:
    rel_dif = abs(2*(pair[0]-pair[1])/(pair[0]+pair[1]))
    diffs.append(rel_dif)


# ###  Building table

# In[ ]:


"""Building the Tex table"""

pretty_vals = []
for pair in st_vals:
    point_vals = []
    for i in range(2):
        if abs(pair[i]) < 10**(-2):
            point_vals.append(Table.format_exp('{:.9g}'.format(pair[i].real)))
        else:
            point_vals.append(Table.format_result(pair[i], digits = 15))
    pretty_vals.append(point_vals)

header = r'\begin{array}{|c|c|c|} \hline ' + '\n'

header += r' z & Z(z) & \text{Difference} \\ \hline'

data = ''
for i in range(len(points)):
    data += '\n ' + '%s & ' % points[i]

    data += (r'\begin{array}{l}  ' + '\n' + r's: %s \\' %pretty_vals[i][0]
             + '\n ' + r't: %s ' %pretty_vals[i][1] + '\n'+ r'\end{array}')

    data += r' & ' + Table.format_precision(diffs[i]) + r' \\ \hline'

ending = '\n' + r'\end{array}'

tbl_st = header + data + ending # string for the table.

"""Creating references for the table"""

ref_str = (r'\left[ \begin{array}{l}' + '\n' 
            + r'c = %s \\ ' %c  + '\n' 
            + r'\Delta_1 = ')

D1 = Ext_fields[0].dims[0].get('degenerate') 
if D1 == None:
    D1 = Ext_fields[0].dims[0].get('Delta')
    ref_str += str(D1) + r'\\' + '\n'
else:
    ref_str += r'\Delta_{%s} \\' %latex((Rational(D1[0]), Rational(D1[1])) )+ '\n'

D2 = Ext_fields[1].dims[0].get('degenerate') 
ref_str += r'(r_2, s_2) = %s \\ ' %latex((Rational(D2[0]), Rational(D2[1]))) + '\n'

ref_str += r'\end{array} \right] \rightarrow' + '\n' #string for references.


""" Displaying Table and Tex commands"""

display(Math(ref_str + tbl_st))
print(r'\epsilon = %s, L= %s, blocklevel = %s' %(eps, L_s, bl))
print(ref_str + tbl_st)


# ##  Functions with four different fields

# Now we compute a function of the type
# \begin{align}
# \left\langle V^D_{P_1} V^N_{(r_2,s_2)} V^D_{P_3} V^N_{(r_4,s_4)} \right \rangle
# \end{align}

# In[ ]:


"""Generate charge and fields"""

c = -0.5432
eps = 0
C = Charge('c', c+eps)
beta0 = C.get('beta')

L_s = 8  #spectrum truncation order
bl = 20 # blocks truncation order

Ext_fields = four_point_args(beta0, 0.2314, (4,-7/2 ,'n') , 1.265, (2,1/2,'n')).fields


# In[ ]:


"""Generating functions and computing their values."""

"""Generating four-point function"""
t0 = time.clock()
Z = ND4(Ext_fields, L = L_s, blocklevel = bl)
dur = time.clock()-t0
print('Function created in %s seconds.' %Table.format_time(dur))

"""Computing s-and-t-channel values."""

points = [0.01, 0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9] # points to compute the functions.

st_vals = [Z.values(z = p) for p in points] # s-ch, t-ch for each point.

"""Computing relative differences"""
diffs = []

for pair in st_vals:
    rel_dif = abs(2*(pair[0]-pair[1])/(pair[0]+pair[1]))
    diffs.append(rel_dif)


# In[ ]:


"""Building the Tex table"""

pretty_vals = []
for pair in st_vals:
    point_vals = []
    for i in range(2):
        if abs(pair[i]) < 10**(-2):
            point_vals.append(Table.format_exp('{:.9g}'.format(pair[i].real)))
        else:
            point_vals.append(Table.format_result(pair[i], digits = 15))
    pretty_vals.append(point_vals)

header = r'\begin{array}{|c|c|c|} \hline ' + '\n'

header += r' z & Z(z) & \text{Difference} \\ \hline'

data = ''
for i in range(len(points)):
    data += '\n ' + '%s & ' % points[i]

    data += (r'\begin{array}{l}  ' + '\n' + r's: %s \\' %pretty_vals[i][0]
             + '\n ' + r't: %s ' %pretty_vals[i][1] + '\n'+ r'\end{array}')

    data += r' & ' + Table.format_precision(diffs[i]) + r' \\ \hline'

ending = '\n' + r'\end{array}'

tbl_st = header + data + ending # string for the table.

"""Creating references for the table"""

ref_str = (r'\left[ \begin{array}{l}' + '\n' 
            + r'c = %s \\ ' %c  + '\n' )


D1 = Ext_fields[0].dims[0].get('degenerate') 
ref_str += r'\Delta_1 = '
if D1 == None:
    D1 = Ext_fields[0].dims[0].get('Delta')
    ref_str += str(D1) + r'\\' + '\n'
else:
    ref_str += r' \Delta_{%s} \\' %latex((Rational(D1[0]), Rational(D1[1])) )+ '\n'

D2 = Ext_fields[1].dims[0].get('degenerate') 
ref_str += r'(r_2, s_2) = %s \\ ' %latex((Rational(D2[0]), Rational(D2[1]))) + '\n'

D3 = Ext_fields[2].dims[0].get('degenerate') 
ref_str += r'\Delta_3 = '
if D3 == None:
    D3 = Ext_fields[2].dims[0].get('Delta')
    ref_str += str(D3) + r'\\' + '\n'
else:
    ref_str += r'\Delta_{%s} \\' %latex((Rational(D3[0]), Rational(D3[1])) )+ '\n'

D4 = Ext_fields[3].dims[0].get('degenerate') 
ref_str += r'(r_4, s_4) = %s \\ ' %latex((Rational(D4[0]), Rational(D4[1]))) + '\n'

ref_str += r'\end{array} \right] \rightarrow' + '\n' #string for references.


""" Displaying Table and Tex commands"""

display(Math(ref_str + tbl_st))
print(r'\epsilon = %s, L= %s, blocklevel = %s' %(eps, L_s, bl))
print(ref_str + tbl_st)


# In[ ]:




