# coding: utf-8
"""
Helpers for :epkg:`sympy`.
"""


def enumerate_traverse(e, parent=None):
    """
    Enumerates all nodes in *e*.

    :param e: :epkg:`sympy` expression or node
    :param parent: parent of *e*
    :return: iterator on itself and the children
    """
    yield dict(e=e, p=parent)
    for arg in e.args:
        for r in enumerate_traverse(arg, e):
            yield r


class SympyNode:
    """
    Model a :epkg:`sympy` expression.
    It probably exist in :epkg:`sympy`.

    :param e: :epkg:`sympy` expression
    :param parent: *SympyNode*
    """

    def __init__(self, e, parent=None):
        if parent is not None and not isinstance(parent, SympyNode):
            raise TypeError(  # pragma: no cover
                "parent must be None or SympyNode")
        self._e = e
        self._parent = parent
        self._children = []
        for a in self._e.args:
            self._children.append(SympyNode(a, self))

    def __repr__(self):
        return f'SympyNode("{self.element!r}")'

    @property
    def element(self):
        return self._e

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return self._children

    def __iter__(self):
        """
        Enumerate all nodes.
        """
        yield self.element
        for a in self.children:
            for n in a:
                yield a

    def enumerate_parents(self):
        """
        Enumerates all parents including itself.
        """
        n = self
        while n.parent is not None:
            yield n
            n = n.parent
