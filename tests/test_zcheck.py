"""
Unit tests for ``random_strategy``.
"""
import unittest
from contextlib import redirect_stdout
import io
from aftercovid import check


class TestCheck(unittest.TestCase):

    def test_check(self):
        f = io.StringIO()
        with redirect_stdout(f):
            res = check(verbose=1)
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 0)

    def test__main__(self):
        import aftercovid.__main__  # noqa


if __name__ == '__main__':
    unittest.main()
