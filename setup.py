from setuptools import setup

setup(
    name='cpartition',
    version='0.1dev',
    description='',
    author='Arthur Nishikawa',
    author_email='nishikawa.poli@gmail.com',
    url='https://github.com/arthursn/cpartition',
    packages=['cpartition'],
    install_requires=['numpy', 'matplotlib',
                      'scipy', 'pandas', 'periodictable'],
    long_description=open('README.md').read(),
)
