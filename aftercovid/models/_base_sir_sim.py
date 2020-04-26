# coding: utf-8
"""
Common methods about simulation for :epkg:`SIR` models.
"""
import numpy
from sympy import Symbol, diff as sympy_diff


class BaseSIRSimulation:
    """
    Common methods about simulation for :epkg:`SIR` models.
    """

    def eqsign(self, eqname, name):
        """
        Returns the sign of the second derivative for equation
        *eqname* against *name*.

        :param eqname: equation name
        :param name: symbol name
        :return: boolean
        """
        eq = self._eq[eqname]
        df = sympy_diff(eq, Symbol(name))
        ev = self.evalf_eq(df)
        return 1 if ev >= 0 else -1

    def evalf_eq(self, eq, t=0):
        """
        Evaluates an :epkg:`sympy` expression.
        """
        svalues = self._eval_cache()
        svalues[self._syms['t']] = t
        for k, v in zip(self._q, self._val_q):
            svalues[self._syms[k[0]]] = v
        return eq.evalf(subs=svalues)

    def _eval_cache(self):
        values = self.cst_param
        svalues = {self._syms[k]: v for k, v in values.items()}
        return svalues

    def eval_diff(self, t=0):
        """
        Evaluates derivatives.
        Returns a dictionary.
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

        :param n: number of iterations
        :param t: first *t*
        :param derivatives: returns the derivative as well
        :return: iterator on dictionaries
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

            fvals = {k: float(v) for k, v in vals.items()}
            if derivatives:
                yield fvals, diff
            else:
                yield fvals

            for k, v in diff.items():
                vals[k] += v
            self.update(**vals)

    def iterate2array(self, n=10, t=0, derivatives=False):
        """
        Evalues the quantities for *n* iterations.
        Returns matrices.

        :param n: number of iterations
        :param t: first *t*
        :param derivatives: returns the derivative as well
        :return: iterator on dictionaries
        """
        clq = self.quantity_names
        pos = {n: i for i, n in enumerate(clq)}
        res = list(self.iterate(n=n, t=t, derivatives=derivatives))
        qu = numpy.zeros((len(res), len(clq)), dtype=numpy.float32)
        if derivatives:
            de = numpy.zeros((len(res), len(clq)), dtype=numpy.float32)
            for i, (r, d) in enumerate(res):
                for j, n in enumerate(pos):
                    qu[i, j] = r.get(n, numpy.nan)
                for j, n in enumerate(pos):
                    de[i, j] = d.get(n, numpy.nan)
            return qu, de
        else:
            for i, r in enumerate(res):
                for j, n in enumerate(pos):
                    qu[i, j] = r.get(n, numpy.nan)
            return qu

    def R0(self, t=0):
        '''Returns R0 coefficient.'''
        raise NotImplementedError()
