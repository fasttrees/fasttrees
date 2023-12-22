'''
The :mod:`fasttrees` module includes the fast-and-frugal tree classifier.
'''


__description__ = "A fast and frugal tree classifier for sklearn"
__author__ = "Dominic Zijlstra, Stefan Bachhofner"

__license__ = "MIT"
__version__ = "1.3.0"
__author_email__ = "dominiczijlstra@gmail.com, bachhofner.dev@gmail.com"



from .fasttrees import FastFrugalTreeClassifier

__all__ = ['FastFrugalTreeClassifier']
