# coding: utf-8
"""
Common functions for :epkg:`SIR` models.
"""
import numpy
from sympy import symbols, Symbol
import sympy.printing as printing
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_application)
from ._sympy_helper import enumerate_traverse
from ._base_sir_sim import BaseSIRSimulation
from ._base_sir_estimation import BaseSIRSklAPI


class BaseSIR(BaseSIRSimulation, BaseSIRSklAPI):
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
                except (TypeError, ValueError) as e:  # pragma: no cover
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
    def quantity_names(self):
        'Returns the list of quantities names (unsorted).'
        return [v[0] for v in self._q]

    @property
    def param_names(self):
        'Returns the list of parameters names (unsorted).'
        return [v[0] for v in self._p]

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

    def enumerate_edges(self):
        """
        Enumerates the list of quantities contributing
        to others. It ignores constants.
        """
        if self._eq is not None:
            params = set(_[0] for _ in self.P)
            quants = set(_[0] for _ in self.Q)
            for k, v in sorted(self._eq.items()):
                n2 = k
                for dobj in enumerate_traverse(v):
                    term = dobj['e']
                    if not hasattr(term, 'name'):
                        continue
                    if term.name not in params:
                        continue
                    parent = dobj['p']
                    others = list(
                        _['e'] for _ in enumerate_traverse(parent))
                    for o in others:
                        if hasattr(o, 'name') and o.name in quants:
                            sign = self.eqsign(n2, o.name)
                            yield (sign, o.name, n2, term.name)

    def to_dot(self, verbose=False, full=False):
        """
        Produces a graph in :epkg:`DOT` format.
        """
        rows = ['digraph{']

        pattern = ('    {name} [label="{name}\\n{doc}" shape=record];'
                   if verbose else
                   '    {name} [label="{name}"];')
        for name, _, doc in self._q:
            rows.append(pattern.format(name=name, doc=doc))
        for name, _, doc in self._c:
            rows.append(pattern.format(name=name, doc=doc))

        if self._eq is not None:
            pattern = (
                '    {n1} -> {n2} [label="{sg}{name}\\nvalue={v:1.2g}"];'
                if verbose else '    {n1} -> {n2} [label="{sg}{name}"];')
            for sg, a, b, name in set(self.enumerate_edges()):
                if not full and (a == b or sg < 0):
                    continue
                value = self[name]
                stsg = '' if sg > 0 else '-'
                rows.append(
                    pattern.format(n1=a, n2=b, name=name, v=value, sg=stsg))

        rows.append('}')
        return '\n'.join(rows)

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
