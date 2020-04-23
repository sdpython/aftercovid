# coding: utf-8
"""
Common function for :epkg:`SIR` models.
"""


class BaseSIRSimulation:
    """
    Base model for :epkg:`SIR` models simulation.
    """

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

    def R0(self, t=0):
        '''Returns R0 coefficient.'''
        raise NotImplementedError()
