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
    install_requires=[
                'numpy',
                'pandas<=0.25.3',
                'sklearn',
                'logging'
          ]
)
