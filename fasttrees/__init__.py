'''
The :mod:`fasttrees` module includes the fast-and-frugal tree classifier.
'''
from pbr.version import VersionInfo

__version__ = VersionInfo('fasttrees').release_string()




from .fasttrees import FastFrugalTreeClassifier

__all__ = ['FastFrugalTreeClassifier']
