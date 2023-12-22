# pylint: disable=missing-module-docstring

from setuptools import setup


import fasttrees


def get_long_description():
    ''' returns the content of the README.md file as a string.
    '''
    from os import path # pylint: disable=import-outside-toplevel

    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md'), mode='r', encoding='utf-8') as f:
        long_description = f.read()

    return long_description



setup(
    name='fasttrees',
    version=fasttrees.__version__,
    packages=['fasttrees'],
    url='https://github.com/fasttrees/fasttrees',
    license='MIT License',
    author=fasttrees.__author__,
    author_email=fasttrees.__author_email__,
    description=fasttrees.__description__,
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    classifiers=[
        'License :: OSI Approved :: MIT License',

        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',

        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',

        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8'
    ],
    project_urls={
        'Documentation': 'https://fasttrees.github.io/fasttrees',
        'Source': 'https://github.com/fasttrees/fasttrees',
        'Tracker': 'https://github.com/fasttrees/fasttrees/issues',
    },
    python_requires='>=3.8',
    install_requires=[
                'numpy<=1.19.5',
                'pandas<=0.25.3',
                'scikit-learn<=0.23.2',
                'logging'
          ]
)
