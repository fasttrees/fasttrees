'''
The :mod:`fasttrees` module includes the fast-and-frugal tree classifier.
'''
from importlib.metadata import distribution


fasttrees_metadata = dict(distribution('fasttrees').metadata)

__version__ = fasttrees_metadata['Version']
__description__ = fasttrees_metadata['Summary']
__license__ = fasttrees_metadata['License']

__author__ = fasttrees_metadata['Author']
__author_email__ = fasttrees_metadata['Author-email']




from .fasttrees import FastFrugalTreeClassifier

__all__ = ['FastFrugalTreeClassifier']
