# coding: utf-8
"""
Common function for :epkg:`SIR` models.
"""
import numpy
from sympy import symbols, Symbol
import sympy.printing as printing
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_application)


class BaseSIR:
    """
    Base model for :epkg:`SIR` models.

    :param p: list of `[(name, initial value or None, comment)]` (parameters)
    :param q: list of `[(name, initial value or None, comment)]` (quantities)
    :param c: list of `[(name, initial value or None, comment)]` (constants)
    :param eq: equations
    """

    def __init__(self, p, q, c=None, eq=None):
        if not isinstance(p, list):
            raise TypeError("p must be a list of tuple.")
        if not isinstance(q, list):
            raise TypeError("q must be a list of tuple.")
        if not isinstance(c, list):
            raise TypeError("c must be a list of tuple.")
        if eq is not None and not isinstance(eq, dict):
            raise TypeError("eq must be a dictionary.")
        self._p = p
        self._q = q
        self._c = c
        if eq is not None:
            locs = {'t': symbols('t', cls=Symbol)}
            for v in self._p:
                locs[v[0]] = symbols(v[0], cls=Symbol)
            for v in self._c:
                locs[v[0]] = symbols(v[0], cls=Symbol)
            for v in self._q:
                locs[v[0]] = symbols(v[0], cls=Symbol)
            self._syms = locs
            tr = standard_transformations + (implicit_application, )
            self._eq = {}
            for k, v in eq.items():
                try:
                    self._eq[k] = parse_expr(v, locs, transformations=tr)
                except (TypeError, ValueError) as e:
                    raise RuntimeError(
                        "Unable to parse '{}'.".format(v)) from e
        else:
            self._eq = None
        self._init()

    def _init(self):
        """
        Starts from the initial values.
        """
        def _def_(name, v):
            if v is not None:
                return v
            if name == 'N':
                return 10000.
            return 0.

        self._val_p = numpy.array([_def_(v[0], v[1]) for v in self._p])
        self._val_q = numpy.array([_def_(v[0], v[1]) for v in self._q])
        self._val_c = numpy.array([_def_(v[0], v[1]) for v in self._c])

    def get_index(self, name):
        '''
        Returns the index of a name (True or False, position).
        '''
        for i, v in enumerate(self._p):
            if v[0] == name:
                return 'p', i
        for i, v in enumerate(self._q):
            if v[0] == name:
                return 'q', i
        for i, v in enumerate(self._c):
            if v[0] == name:
                return 'c', i
        raise ValueError("Unable to find name '{}'.".format(name))

    def __setitem__(self, name, value):
        """
        Updates a value whether it is a parameter or a quantity.

        :param name: name
        :param value: new value
        """
        p, pos = self.get_index(name)
        if p == 'p':
            self._val_p[pos] = value
        elif p == 'q':
            self._val_q[pos] = value
        elif p == 'c':
            self._val_c[pos] = value

    def __getitem__(self, name):
        """
        Retrieves a value whether it is a parameter or a quantity.

        :param name: name
        :return: value
        """
        p, pos = self.get_index(name)
        if p == 'p':
            return self._val_p[pos]
        if p == 'q':
            return self._val_q[pos]
        if p == 'c':
            return self._val_c[pos]

    @property
    def names(self):
        'Returns the list of names.'
        return list(sorted(
            [v[0] for v in self._p] + [v[0] for v in self._q] +
            [v[0] for v in self._c]))

    @property
    def P(self):
        '''
        Returns the parameters
        '''
        return self._p

    @property
    def Q(self):
        '''
        Returns the quantities
        '''
        return self._q

    @property
    def C(self):
        '''
        Returns the quantities
        '''
        return self._c

    def update(self, **values):
        """Updates values."""
        for k, v in values.items():
            self[k] = v

    def get(self):
        """Retrieves all values."""
        return {n: self[n] for n in self.names}

    def to_rst(self):
        '''
        Returns a string formatted in RST.
        '''
        rows = [
            '*{}*'.format(self.__class__.__name__),
            '',
            '*Q*',
            ''
        ]
        for name, _, doc in self._q:
            rows.append('* *{}*: {}'.format(name, doc))
        rows.extend(['', '*C*', ''])
        for name, _, doc in self._c:
            rows.append('* *{}*: {}'.format(name, doc))
        rows.extend(['', '*P*', ''])
        for name, _, doc in self._p:
            rows.append('* *{}*: {}'.format(name, doc))
        if self._eq is not None:
            rows.extend(['', '*E*', '', '.. math::',
                         '', '    \\begin{array}{l}'])
            for i, (k, v) in enumerate(sorted(self._eq.items())):
                line = "".join(
                    ["    ", "\\frac{d%s}{dt} = " % k, printing.latex(v)])
                if i < len(self._eq) - 1:
                    line += " \\\\"
                rows.append(line)
            rows.append("    \\end{array}")

        return '\n'.join(rows)

    def evalf(self, name, t):
        """
        Evaluate quantity *name* at time *t*.
        *t* is unused.
        """
        return self[name]

    @property
    def cst_param(self):
        '''
        Returns a dictionary with the constant and the parameters.
        '''
        res = {}
        for k, v in zip(self._c, self._val_c):
            res[k[0]] = v
        for k, v in zip(self._p, self._val_p):
            res[k[0]] = v
        return res

    def _eval_cache(self):
        values = self.cst_param
        svalues = {self._syms[k]: v for k, v in values.items()}
        return svalues

    def eval_diff(self, t=0):
        """
        Evaluates derivatives.
        Returns a directionary.
        """
        svalues = self._eval_cache()
        svalues[self._syms['t']] = t
        for k, v in zip(self._q, self._val_q):
            svalues[self._syms[k[0]]] = v

        res = {}
        for k, v in self._eq.items():
            res[k] = v.evalf(subs=svalues)
        return res

    def iterate(self, n=10, t=0, derivatives=False):
        """
        Evalues the quantities for *n* iterations.
        Returns a list of dictionaries.
        If *derivatives* is True, it returns two dictionaries.
        """
        svalues = self._eval_cache()
        svalues[self._syms['t']] = t
        vals = {k[0]: v for k, v in zip(self._q, self._val_q)}
        for i in range(t, t + n):

            for k, v in zip(self._q, self._val_q):
                svalues[self._syms[k[0]]] = v
            diff = {}
            for k, v in self._eq.items():
                diff[k] = v.evalf(subs=svalues)

            for k, v in diff.items():
                vals[k] += v
            fvals = {k: float(v) for k, v in vals.items()}
            if derivatives:
                yield fvals, diff
            else:
                yield fvals
            self.update(**vals)
