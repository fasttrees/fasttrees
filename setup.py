from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='fasttrees',
    version='1.2.1',
    packages=['fasttrees'],
    url='https://github.com/dominiczy/fasttrees',
    license='MIT License',
    author='dominiczijlstra',
    author_email='dominiczijlstra@gmail.com',
    description='A fast and frugal tree classifier for sklearn',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
                'numpy',
                'pandas',
                'sklearn'
          ]
)
