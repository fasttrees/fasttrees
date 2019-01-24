from setuptools import setup

setup(
    name='fasttrees',
    version='1.1.0',
    packages=['fasttrees'],
    url='https://github.com/dominiczy/fasttrees',
    license='MIT License',
    author='dominiczijlstra',
    author_email='dominiczijlstra@gmail.com',
    description='A fast and frugal tree classifier for sklearn',
    install_requires=[
                'numpy',
                'pandas',
                'sklearn'
          ]
)
