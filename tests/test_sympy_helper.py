"""
Unit tests for ``CovidSir``.
"""
import unittest
from aftercovid.models._sympy_helper import enumerate_traverse, SympyNode
from aftercovid.models.covid_sir import CovidSIR


class TestSympyHelper(unittest.TestCase):

    def test_traverse(self):
        model = CovidSIR()
        exp = model._eq['S']
        nodes = list(enumerate_traverse(exp))
        self.assertEqual(len(nodes), 8)
        self.assertIsInstance(nodes[0], dict)

    def test_node(self):
        model = CovidSIR()
        exp = model._eq['S']
        node = SympyNode(exp)
        nodes = list(node)
        self.assertEqual(len(nodes), 8)
        st = list(map(repr, nodes))
        self.assertEqual(len(st), 8)

    def test_parent(self):
        model = CovidSIR()
        exp = model._eq['S']
        node = SympyNode(exp)
        n = node.children[0]
        ps = list(n.enumerate_parents())
        self.assertEqual(len(ps), 1)


if __name__ == '__main__':
    unittest.main()
