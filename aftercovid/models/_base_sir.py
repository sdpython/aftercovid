# coding: utf-8
"""
Common functions for :epkg:`SIR` models.
"""
import numpy
from sympy import symbols, Symbol, latex, lambdify
import sympy.printing as printing
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_application)
from ._sympy_helper import enumerate_traverse
from ._base_sir_sim import BaseSIRSimulation
from ._base_sir_estimation import BaseSIREstimation


class BaseSIR(BaseSIRSimulation, BaseSIREstimation):
    """
    Base model for :epkg:`SIR` models.

    :param p: list of `[(name, initial value or None, comment)]` (parameters)
    :param q: list of `[(name, initial value or None, comment)]` (quantities)
    :param c: list of `[(name, initial value or None, comment)]` (constants)
    :param eq: equations
    """
    _pickled_atts = [
        '_p', '_q', '_c', '_eq', '_val_p', '_val_q', '_val_c',
        '_val_ind', '_val_len', '_syms']

    def __init__(self, p, q, c=None, eq=None, **kwargs):
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
        if len(kwargs) != 0:
            raise NotImplementedError(
                "Not implemented.")
        self._init()

    def copy(self):
        inst = self.__class__.__new__(self.__class__)
        for k in BaseSIR._pickled_atts:
            setattr(inst, k, getattr(self, k))
        if hasattr(inst, '_eq') and inst._eq is not None:
            inst._init_lambda_()
        return inst

    def __getstate__(self):
        '''
        Returns the pickled data.
        '''
        return {k: getattr(self, k) for k in BaseSIR._pickled_atts}

    def __setstate__(self, state):
        '''
        Sets the pickled data.
        '''
        for k, v in state.items():
            setattr(self, k, v)
        if hasattr(self, '_eq') and self._eq is not None:
            self._init_lambda_()

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

        self._val_p = numpy.array(
            [_def_(v[0], v[1]) for v in self._p], dtype=numpy.float64)
        self._val_q = numpy.array(
            [_def_(v[0], v[1]) for v in self._q], dtype=numpy.float64)
        self._val_c = numpy.array(
            [_def_(v[0], v[1]) for v in self._c], dtype=numpy.float64)
        self._val_len = (len(self._val_p) + len(self._val_q) +
                         len(self._val_c))
        self._val_ind = numpy.array([
            0, len(self._val_q), len(self._val_q) + len(self._val_p),
            len(self._val_q) + len(self._val_p) + len(self._val_c)])

        if hasattr(self, '_eq') and self._eq is not None:
            self._init_lambda_()

    def _init_lambda_(self):
        self._leq = {}
        for k, v in self._eq.items():
            fct = self._lambdify_(k, v)
            eval1 = float(self.evalf_eq(v))
            eval2 = self.evalf_leq(k)
            err = (eval2 - eval1) / max(abs(eval1), abs(eval2))
            if err > 1e-8:
                raise ValueError(  # pragma: no cover
                    "Lambdification failed for function '{}': {} "
                    "({} ({}) != {} ({}), error={})".format(
                        k, v, eval1, type(eval1), eval2, type(eval2), err))
            self._leq[k] = fct
        self._leqa = [self._leq[_[0]] for _ in self._q]

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
    def params_dict(self):
        'Returns the list of parameters names in a dictionary.'
        return {k: self[k] for k in self.param_names}

    @property
    def cst_names(self):
        'Returns the list of constants names (unsorted).'
        return [v[0] for v in self._c]

    @property
    def vect_names(self):
        'Returns the list of names.'
        return ([v[0] for v in self._q] + [v[0] for v in self._p] +
                [v[0] for v in self._c] + ['t'])

    def vect(self, t=0, out=None, derivative=False):
        """
        Returns all values as a vector.

        :param
        """
        if derivative:
            if out is None:
                out = numpy.empty((self._val_len + 1 + self._val_ind[1], ),
                                  dtype=numpy.float64)
            self.vect(t=t, out=out)
            for i, v in enumerate(self._leqa):
                out[i - self._val_ind[1]] = v(*out[:self._val_len + 1])
        else:
            if out is None:
                out = numpy.empty((self._val_len + 1, ), dtype=numpy.float64)
        out[:self._val_ind[1]] = self._val_q
        out[self._val_ind[1]:self._val_ind[2]] = self._val_p
        out[self._val_ind[2]:self._val_ind[3]] = self._val_c
        out[self._val_ind[3]] = t
        return out

    @property
    def P(self):
        '''
        Returns the parameters
        '''
        return [(a[0], b, a[2]) for a, b in zip(self._p, self._val_p)]

    @property
    def Q(self):
        '''
        Returns the quantities
        '''
        return [(a[0], b, a[2]) for a, b in zip(self._q, self._val_q)]

    @property
    def C(self):
        '''
        Returns the quantities
        '''
        return [(a[0], b, a[2]) for a, b in zip(self._c, self._val_c)]

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
            '*Quantities*',
            ''
        ]
        for name, _, doc in self._q:
            rows.append('* *{}*: {}'.format(name, doc))
        rows.extend(['', '*Constants*', ''])
        for name, _, doc in self._c:
            rows.append('* *{}*: {}'.format(name, doc))
        rows.extend(['', '*Parameters*', ''])
        for name, _, doc in self._p:
            rows.append('* *{}*: {}'.format(name, doc))
        if self._eq is not None:
            rows.extend(['', '*Equations*', '', '.. math::',
                         '', '    \\begin{array}{l}'])
            for i, (k, v) in enumerate(sorted(self._eq.items())):
                line = "".join(
                    ["    ", "\\frac{d%s}{dt} = " % k, printing.latex(v)])
                if i < len(self._eq) - 1:
                    line += " \\\\"
                rows.append(line)
            rows.append("    \\end{array}")

        return '\n'.join(rows)

    def _repr_html_(self):
        '''
        Returns a string formatted in RST.
        '''
        rows = [
            '<p><b>{}</b></p>'.format(self.__class__.__name__),
            '',
            '<p><i>Quantities</i></p>',
            '',
            '<ul>'
        ]
        for name, _, doc in self._q:
            rows.append('<li><i>{}</i>: {}</li>'.format(name, doc))
        rows.extend(['</ul>', '', '<p><i>Constants</i></p>', '', '<ul>'])
        for name, _, doc in self._c:
            rows.append('<li><i>{}</i>: {}</li>'.format(name, doc))
        rows.extend(['</ul>', '', '<p><i>Parameters</i></p>', '', '<ul>'])
        for name, _, doc in self._p:
            rows.append('<li><i>{}</i>: {}</li>'.format(name, doc))
        if self._eq is not None:
            rows.extend(['</ul>', '', '<p><i>Equations</i></p>', '', '<ul>'])
            for i, (k, v) in enumerate(sorted(self._eq.items())):
                lats = "\\frac{d%s}{dt} = %s" % (k, printing.latex(v))
                lat = latex(lats, mode='equation')
                line = "".join(["<li>", str(lat), '</li>'])
                rows.append(line)
            rows.append("</ul>")

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
                n = []
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
                            if o.name != n2:
                                n.append((sign, o.name, n2, term.name))
                if len(n) == 0:
                    yield (0, '?', n2, '?')

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
                if name == '?':
                    rows.append(
                        pattern.format(n1=a, n2=b, name=name,
                                       v=numpy.nan, sg='0'))
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

    def evalf_eq(self, eq, t=0):
        """
        Evaluates an :epkg:`sympy` expression.
        """
        svalues = self._eval_cache()
        svalues[self._syms['t']] = t
        for k, v in zip(self._q, self._val_q):
            svalues[self._syms[k[0]]] = v
        return eq.evalf(subs=svalues)

    def evalf_leq(self, name, t=0):
        """
        Evaluates a lambdified expression.

        :param name: name of the lambdified expresion
        :param t: t values
        :return: evaluation
        """
        leq = self._lambdified_(name)
        if leq is None:
            raise RuntimeError(  # pragma: no cover
                "Equation '{}' was not lambdified.".format(name))
        return leq(*self.vect(t))

    def _eval_cache(self):
        values = self.cst_param
        svalues = {self._syms[k]: v for k, v in values.items()}
        return svalues

    def _lambdify_(self, name, eq, derivative=False):
        'Lambdifies an expression and caches in member `_lambda_`.'
        if not hasattr(self, '_lambda_'):
            self._lambda_ = {}
        if name not in self._lambda_:
            names = (self.quantity_names + self.param_names +
                     self.cst_names + ['t'])
            sym = [Symbol(n) for n in names]
            if derivative:
                sym += [Symbol('d' + n) for n in self.quantity_names]
            self._lambda_[name] = {
                'names': names,
                'symbols': sym,
                'eq': eq,
                'pos': {n: i for i, n in enumerate(names)},
            }
            ll = lambdify(sym, eq, 'numpy')
            self._lambda_[name]['la'] = ll
        return self._lambda_[name]['la']

    def _lambdified_(self, name):
        """
        Returns the lambdified expression of name *name*.
        """
        if hasattr(self, '_lambda_'):
            r = self._lambda_.get(name, None)
            if r is not None:
                return r['la']
        return None

    def _eval_diff_sympy(self, t=0):
        """
        Evaluates derivatives.
        Returns a dictionary.
        """
        svalues = self._eval_cache()
        svalues[self._syms['t']] = t
        for k, v in zip(self._q, self._val_q):
            svalues[self._syms[k[0]]] = v

        x = self.vect(t=t)
        res = {}
        for k, v in self._eq.items():
            res[k] = v.evalf(subs=svalues)
        for k, v in self._leq.items():
            res[k] = v(*x)
        return res

    def eval_diff(self, t=0):
        """
        Evaluates derivatives.
        Returns a dictionary.
        """
        x = self.vect(t=t)
        res = {}
        for k, v in self._leq.items():
            res[k] = v(*x)
        return res
