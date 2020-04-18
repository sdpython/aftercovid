"""
Implements command line ``python -m aftercovid <command> <args>``.
"""
import sys


def main(args, fLOG=print):
    """
    Implements ``python -m aftercovid check <command> <args>``.
    """
    from pyquickhelper.cli import cli_main_helper
    try:
        from . import check
    except ImportError:  # pragma: no cover
        from aftercovid import check

    fcts = dict(check=check)
    return cli_main_helper(fcts, args=args, fLOG=fLOG)


if __name__ == "__main__":
    main(sys.argv[1:])
