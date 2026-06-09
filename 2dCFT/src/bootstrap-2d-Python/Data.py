#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
import socket
from Correlators import *
import time
import mpmath
from IPython.display import display, Math
import matplotlib.pyplot as plt
plt.rc("figure", facecolor="white")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
try:
    import builtins
except ImportError:
    import __builtin__
def map(*args):
    return list(builtins.map(*args))


# # Generating and saving data

# There are several types of data we want to consider, including theory-dependent cutoffs:
# 
# 1. Global data:
#  1. Computer
#  2. Date 
#  
# 3. Physical:
#  1. Central charge
#  2. Position
#  3. Four dimensions
#  4. Channel
#  5. Reduced 
#  
# 4. Cutoffs:
#  1. Level of truncation of blocks
#  2. Cutoff of $s$- or $t$-channel momentum integration (in Liouville, Analytic and Rational)
#  3. Cutoff of integration for structure constants (in Liouville)
#  4. Cutoff of product for structure constants (in Analytic and Rational)
#  5. Spline (in Liouville and Analytic)
#  6. Deviation of the central charge from the rational value in conformal blocks (in Rational)
# 
# 5. Results:
#  1. Value of four-point function
#  2. Duration of creation of object
#  3. Duration of calculation of value

# In[ ]:


class Data:

    theories = {'Liouville': 'FourPoint', 'GMM': 'GMM4', 'Analytic': 'Z4', 'Rational': 'R4'}

    dictionary = {'theory': 'theory',
                  'computer': 'computer',
                  'date': 'date',                 
                  't_channel': 'channel',
                  'Ps': 'dimensions',
                  'fields': 'dimensions', 
                  'beta': 'charge',
                  'b': 'charge',
                  'reduced': 'reduced',
                  'xs': 'position',
                  'levels': 'level',
                  'Pcutoff': 'integration',
                  'Kcutoff': 'structure integration',
                  'Ucutoff': 'structure product',
                  'spline' : 'spline',
                  'eta' : 'deviation'}

    def __init__(self, theory = 'Liouville', crossing = True, q = 0.02, **kwargs):
        """ Computes results in the form of a nested array. The first two nestings
        correspond to the level and the position. The last nesting 
        is [s_channel, t_channel, precision, computation time].
        Arguments can include positions either as q (item or list), or as x (item or list). 
        """       

        self.theory = theory
        self.computer = socket.gethostname()
        self.date = time.strftime("%Y/%m/%d -- %H:%M:%S")
        self.crossing = crossing

        if 'x' in kwargs.keys():
            xs = kwargs.pop('x')
            q = map(lambda x: mpmath.qfrom(m = x), xs) if isinstance(xs, list) else mpmath.qfrom(m = xs)
        self.positions = q if isinstance(q, list) else [q]
        self.xs = xs if 'xs' in locals() else map(lambda q: complex(mpmath.mfrom(q = q)), self.positions) 

        if 'Nmax' in kwargs.keys():
            Ns = kwargs['Nmax']
            Ns = Ns if isinstance(Ns, list) else [Ns]
            kwargs['Nmax'] = 0

        self.creator = eval(Data.theories[theory])
        time0 = time.clock()
        self.created = [self.creator(**kwargs)]
        if crossing: 
            self.created.append(self.creator(t_channel = True, **kwargs))
        self.creation_duration = time.clock() - time0

        self.results = []
        self.levels = []
        if 'Nmax' in kwargs.keys():
            for Nmax in Ns:
                self.add_Nmax(Nmax)
        else:
            self.results = [self.get_all_values()]
            self.levels = [self.created[0].Nmax]
        self.total_time = time.clock() - time0

        self.parameters = self.full_parameters()

    def full_parameters(self):
        """ Retrieves values of all parameters and builds a string. """
        values = {}
        for key in Data.dictionary.keys():
            if hasattr(self, key):
                values[Data.dictionary[key]] = getattr(self, key)
            elif hasattr(self.created[0], key):
                values[Data.dictionary[key]] = getattr(self.created[0], key)
        return values  

    def add_Nmax(self, Nmax):
        """ Adds one more value of Nmax. """
        for created in self.created:
            created.computeBlock(Nmax)
        self.results.append(self.get_all_values())  
        self.levels.append(Nmax)

    def get_values(self, q):
        """ Computes a four-point function. """
        time0 = time.clock()
        values = []
        for created in self.created:
            values.append(created.value(q))
        if len(values) == 2:
            values.append(abs(1 - values[0]/values[1]))
        values.append(time.clock() - time0)    
        return values

    def get_all_values(self):
        """ Computes the four-point function for all needed values of q. """
        values = []
        for q in self.positions:
            values.append(self.get_values(q))
        return values    


# In[ ]:


class Table:
    """ A table for displaying results of computations of four-point functions. We want to display
    how one or both channels depend on a given parameter.
    """

    s_channel_name = r"s\text{-channel}"
    t_channel_name = r"t\text{-channel}"
    precision_name = r"\text{precision}"
    time_name = r"\text{time}"
    header = [s_channel_name, t_channel_name, precision_name, time_name]    

    Nmax_name = r"N_\mathrm{max}"
    x_name = "x"    

    def __init__(self, datas, parameter_values = [0], parameter = "", qdep = False, time = True, 
                 digits = 0, real = True, precision = True):
        """ 
        datas: one or more Data objects. If more than one, we also need the name and values 
                of the relevant parameter. 
        parameter: name of the parameter, if applicable.
        parameter_values: values of the parameter, if applicable.
        q_dep: if only one Data object involving dependences on both q and Nmax, indicate which 
                dependence we want to display.
        time: whether we display computation times.
        digits: number of digits in results. If zero we use an exponential formatting.
        real: whether we display only real parts of results.
        precision: whether we display the precision.
        """

        self.datas = datas
        self.islist = isinstance(datas, list)
        data = datas[0] if self.islist else datas
        self.data = data
        self.qdep = qdep or len(data.levels) == 1
        self.crossing = data.crossing
        self.digits = digits
        self.real = real
        self.time = time
        self.precision = precision

        self.parameter = parameter if self.islist else (Table.x_name if self.qdep else Table.Nmax_name)
        self.parameter_values = parameter_values if self.islist else (data.xs if self.qdep
                                                                      else data.levels)

        self.parameters = [self.parameter, self.parameter_values, data.parameters]
        self.build()

    def build(self):
        """ Writes the tables. """
        datas = self.datas
        data = self.data
        tables = []
        table_start = self.table_start()
        first_line = self.build_line(self.parameter, Table.header, formatting = False)
        if self.islist:
            for i in range(len(data.levels)):
                tables.append([])
                for j in range(len(data.positions)):
                    table = table_start + first_line + r" \hline "
                    for k in range(len(datas)):
                        table += self.build_line(self.parameter_values[k], datas[k].results[i][j])
                    table += r" \end{array} "                             
                    tables[-1].append(table)     
        elif self.qdep:
            for i in range(len(data.levels)):
                table = table_start + first_line + r" \hline "
                for j in range(len(data.positions)):
                    table += self.build_line(self.parameter_values[j], data.results[i][j])
                table += r" \end{array} "                             
                tables.append(table)   
        else:
            for j in range(len(data.positions)): 
                table = table_start + first_line + r" \hline "
                for i in range(len(data.levels)):
                    table += self.build_line(self.parameter_values[i], data.results[i][j])
                table += r" \end{array} "                             
                tables.append(table)               
        self.tables = tables            

    def table_start(self):
        """ Writes the preamble. """
        string = r"\begin{array}{|r|" 
        if self.crossing:
            string += "|r|r||"
            if self.precision: 
                string += "r|"
        else:
            string += "r|"
        if self.time:
            string += "r|"
        string += r"} \hline " 
        return string

    def build_line(self, parameter, values, formatting = True):
        """ Writes a line. """ 
        string = Table.format_result(parameter, real = False) + " & " + (Table.format_result(values[0], self.digits, self.real) if formatting else values[0])
        if self.crossing:
            string += " & " + (Table.format_result(values[1], self.digits, self.real) if formatting else values[1])
            if self.precision:
                string += " & " + (Table.format_precision(values[2]) if formatting else values[2])
        if self.time:
            string += " & " + (Table.format_time(values[-1]) if formatting else values[-1])
        string += r" \\ \hline "
        return string

    @staticmethod
    def format_precision(precision):
        """ Formats the precision. """
        return Table.format_exp('{:.2g}'.format(precision))

    @staticmethod
    def format_exp(string):
        """ Writes the exponents in scientific notation. """
        if "e" in string:
            base, exponent = string.split("e")
            return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
        else:
            return string        

    @staticmethod
    def format_time(time):
        """ Formats computation times. """
        return '{:.1f}'.format(time)

    @staticmethod
    def format_result(result, digits = 2, real = True):
        """ Formats results by keeping a given number of digits. """
        if isinstance(result, str) or isinstance(result, int):
            return str(result)
        else:
            number = result.real if real else result
            string = Table.format_exp('{:.5g}'.format(number)) if digits == 0 else '{1:.{0}f}'.format(digits, number)
            return string.replace("j", "i")

    def display(self, all_parameters = False):
        """ Displays all results as Latex tables. """
        if all_parameters:
            print( self.data.parameters )
        levels = self.data.levels
        xs = self.data.xs        
        if self.islist:
            for j in range(len(xs)):
                for i in range(len(levels)):                  
                    string = Table.Nmax_name + "=" + str(levels[i]) + r",\ "
                    string += Table.x_name + "=" + Table.format_result(xs[j])
                    display(Math(string))
                    display(Math(self.tables[i][j]))
        elif self.qdep:
            for i in range(len(levels)):
                string = Table.Nmax_name + "=" + str(levels[i])
                display(Math(string))
                display(Math(self.tables[i]))
        else:
            for j in range(len(xs)):
                string = Table.x_name + "=" + Table.format_result(xs[j])
                display(Math(string))
                display(Math(self.tables[j]))



# In[ ]:


class Graph:
    """ A graph showing the dependence on the cross-ratio x in some interval such as (0, 1). 
    We can show one or both channels. There can be several curves, depending on Nmax or some 
    other parameter.
    """

    xlabel = 'position'
    ylabel = 'four-point function'
    window_title = 'Four-point correlation functions'
    slabel = '$s$-channel'
    tlabel = '$t$-channel'

    def __init__(self, datas, parameter_values = ["test"], parameter = "", Nmax = False, crossing = True):
        """
        datas : one or more Data objects.
        parameter: name of the parameter, if applicable.
        parameter_values: values of the parameter, if applicable.
        Nmax: one or more values of Nmax, False for the largest available level, True for all available levels.
        crossing: whether we display both channels.
        """

        self.islist = isinstance(datas, list)
        self.datas = datas if self.islist else [datas]
        data = datas[0] if self.islist else datas
        self.data = data

        if Nmax is False or len(data.levels) == 1:
            Nindices = [-1]
        else:
            Nmax = [Nmax] if isinstance(Nmax, int) and not isinstance(Nmax, bool) else Nmax
            Nindices = [i for i in range(len(data.levels)) if (Nmax is True or data.levels[i] in Nmax)]
        self.Nindices = Nindices   

        self.crossing = crossing and data.crossing

        self.parameter = parameter
        self.parameter_values = parameter_values 
        self.parameters = [self.parameter, self.parameter_values, data.parameters]

    def show(self, n = 1, loc = 'best', save = None):
        """ Builds and show the plot. If we want multiple figures we should give them numbers,
        ending with the number n = 1.
        """
        xs = self.data.xs
        fig = plt.figure(n)
        fig.canvas.set_window_title(Graph.window_title) 
        showNmax = len(self.Nindices) > 1
        for i in self.Nindices:
            for data in self.datas:
                ys = [data.results[i][j][0].real for j in range(len(xs))]    
                label = "$" + Table.Nmax_name + "=" + str(self.data.levels[i]) + "$" if showNmax else ""
                label = (label + ", " + self.parameter + "$ =" 
                         + Table.format_result(self.parameter_values[self.datas.index(data)]) 
                         + "$") if self.islist else label
                if not showNmax and not self.islist:
                    label = Graph.slabel
                plt.plot(xs, ys, label = label)
                if self.crossing:
                    yt = [data.results[i][j][1].real for j in range(len(xs))] 
                    label = label + " ($t$)"
                    if not showNmax and not self.islist:
                        label = Graph.tlabel
                    plt.plot(xs, yt, label = label)
        if showNmax or self.islist or self.crossing:
            plt.legend(loc = loc)
        plt.xlabel(Graph.xlabel, fontsize = 14.4)
        plt.ylabel(Graph.ylabel, fontsize = 14.4)
        plt.gca().tick_params(axis = 'x', pad = 8)
        plt.gca().tick_params(axis = 'y', pad = 8)      
        if save is not None:
            plt.savefig(save, bbox_inches = 'tight', pad_inches = 0)
        if n == 1:
            plt.show()


# In[ ]:


import os, re

class Save:
    """ Pickling and unpickling objects such as Data, Table and Graph. We use their 'parameters'
    variable. """

    file_ext = '.pkl'
    number_search = re.compile('(\d+)' + file_ext)
    digits = 4

    def __init__(self, to_save, save_msg = None, file_name = 'save', folder_name = 'Pickles/', 
             list_file = 'listsave.txt', save_number = None, verbose = True):
        """ Will pickle the object to_save. """
        self.folder_name = folder_name
        file_name = (folder_name + file_name 
                     + str(Save.next_number(folder_name)).zfill(Save.digits) + Save.file_ext)

        full_msg = file_name + '\n' 
        if save_msg is not None:
            full_msg += save_msg + '\n'
        full_msg += str(to_save.parameters) + '\n'        
        with open(file_name, "wb") as f:
            pickle.dump(to_save, f)
        with open(folder_name + list_file, "ab") as f:
            f.write(full_msg + '\n')
        if verbose:
            print( full_msg )

    @staticmethod 
    def next_number(folder_name):
        """ Looks for numbers in file names and returns a higher number. """
        file_names = os.listdir(folder_name)
        numbers = []
        for name in file_names:
            match = re.search(Save.number_search, name)
            if match:
                numbers.append(int(match.group(1)))   
        return max(numbers) + 1 if len(numbers) > 0 else 1



# Now comes a demonstration of saving results for later use. For this to work we need a 'Pickles' directory to exist. (Other names are possible, using the variable 'folder_name' of the Save class.) First, we should run the following cell.

# In[ ]:


if __name__ == '__main__':

    Ps = [1.3, 1.01, .45, .22]
    b0 = 1.9476
    data0 = Data(theory = 'Liouville', b = b0, Ps = Ps, Nmax = [0, 2, 4], Pcutoff = 3)

    Save(data0, 'These are only test data whose only interest is that they are quick to generate.')
    Table(data0).display()


# Now we may want to have a look at the '.pkl' file whose name was printed above. We may also want to have
# a look at the 'listsave.txt' file, which is supposed to be a list of all saved files. Anyway, we can now switch off the computer, go for a break, come back later -- or simply restart the kernel, so that we are sure that Python forgot all about the computations we just did. Then we recover and display the data by running the following cell.

# In[ ]:


if __name__ == '__main__':

    with open('Pickles/save0003.pkl', 'rb') as f:
        resurrected_data = pickle.load(f)
    Table(resurrected_data).display(True)  


# Let's pickle a Table object. The interest of this object, as opposed to a simple array of Data objects, is that we have 'parameter' and 'parameter_values', which tell us which parameter was varied when generating the Datas. So we will consider an example where the central charge varies. We could similarly pickle Graph objects. Of course, after unpickling a Table object we can use its Datas for other purposes, such as building a Graph -- provided these Datas do have an $x$-dependence.

# In[ ]:


if __name__ == '__main__':

    cs = [1 + 4**(-i) for i in range(6)]
    bs = [Charge('c', c).get('b') for c in cs]
    datas = [Data(theory = 'Liouville', b = b, Nmax = [4, 6, 8]) for b in bs]


# In[ ]:


if __name__ == '__main__':

    table = Table(datas, parameter = r'\text{central charge}', parameter_values = cs)
    table.display()
    Save(table, "Let's see what happens if c -> 1. ")


# Another coffee break...

# In[ ]:


if __name__ == '__main__':

    with open('Pickles/save0004.pkl', 'rb') as f:
        table = pickle.load(f)
    table.display(True)


# The problem with this approach of saving results is that we can safely use the recovered data only with the
# particular program which originally created them. So, if we start modifying the classes whose instances we pickled, we may run into trouble.

# In[ ]:




