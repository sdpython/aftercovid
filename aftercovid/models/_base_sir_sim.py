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
        leqname = 'd' + eqname + '/d' + name
        eql = self._lambdified_(leqname)
        if eql is None:
            eq = self._eq[eqname]
            df = sympy_diff(eq, Symbol(name))
            self._lambdify_(leqname, df)
            eval1 = self.evalf_eq(df)
            eval2 = self.evalf_leq(leqname)
            if abs(eval1 - eval2) > 1e-5:
                raise ValueError(
                    "Lambdification failed for derivative '{}' by '{}' "
                    "({} != {})".format(eqname, name, eval1, eval2))
        ev = self.evalf_leq(leqname)
        return 1 if ev >= 0 else -1

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
        for i in range(t, t + n):
            x = self.vect(t=i)
            diff = {k: v(*x) for k, v in self._leq.items()}
            vals = {k[0]: v for k, v in zip(self._q, x)}

            if derivatives:
                yield vals.copy(), diff
            else:
                yield vals.copy()

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
