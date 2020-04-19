"""
Unit tests for ``CovidSir``.
"""
import unittest
from aftercovid.models._base_sir import BaseSIR
from aftercovid.models.covid_sir import CovidSIR


class TestModelsCovidSir(unittest.TestCase):

    def test_base_sir(self):
        with self.assertRaises(TypeError):
            BaseSIR(('p', 0.5, 'PP'), [('q', 0.6, 'QQ')])
        with self.assertRaises(TypeError):
            BaseSIR([('p', 0.5, 'PP')], ('q', 0.6, 'QQ'))
        with self.assertRaises(TypeError):
            BaseSIR([('p', 0.5, 'PP')], [('q', 0.6, 'QQ')],
                    ('N', 0.6, 'NN'))
        models = BaseSIR([('p', 0.5, 'PP')], [('q', 0.6, 'QQ')],
                         [('N', 0.6, 'NN')])
        names = models.names
        self.assertEqual(names, ['N', 'p', 'q'])
        self.assertEqual(models['p'], 0.5)
        self.assertEqual(models['q'], 0.6)
        self.assertEqual(models['N'], 0.6)
        self.assertEqual(models.P, [('p', 0.5, 'PP')])
        self.assertEqual(models.Q, [('q', 0.6, 'QQ')])
        self.assertEqual(models.C, [('N', 0.6, 'NN')])
        with self.assertRaises(ValueError):
            models['qq']
        models['q'] = 6.1
        self.assertEqual(models['q'], 6.1)
        rst = models.to_rst()
        self.assertIn('*q*: QQ', rst)

    def test_covid_sir(self):
        model = CovidSIR()
        rst = model.to_rst()
        self.assertIn('\\frac', rst)
        self.assertIn('I{', rst)


if __name__ == '__main__':
    unittest.main()
