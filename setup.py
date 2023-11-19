from setuptools import setup


import fasttrees


def get_long_description():
    ''' returns the content of the README.md file as a string.
    '''
    from os import path

    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md'), mode='r', encoding='utf-8') as f:
        long_description = f.read()

    return long_description



setup(
    name='fasttrees',
    version=fasttrees.__version__,
    packages=['fasttrees'],
    url='https://github.com/dominiczy/fasttrees',
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
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12'
    ],
    project_urls={
        'Documentation': 'https://github.com/dominiczy/fasttrees/blob/master/README.md',
        'Source': 'https://github.com/dominiczy/fasttrees',
        'Tracker': 'https://github.com/dominiczy/fasttrees/issues',
    },
    install_requires=[
                'numpy',
                'pandas<=0.25.3',
                'sklearn<=0.23.2',
                'logging'
          ]
)
