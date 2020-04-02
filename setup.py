from distutils.core import setup
#from setuptools import setup
 
setup(
    name='spherical_stats',    # This is the name of your PyPI-package.
    version='0.2', 
    description='Numba based spherical statistics',
    author='Daniel Schmitz',
    license='MIT',                        # Update the version number for new releases
    packages=['spherical_stats'],
    #url='https://jugit.fz-juelich.de/da.schmitz/pyesag',
    install_requires=[
        'numpy',
        'numba>0.44',
        'scipy>0.11',
	'matplotlib>2.0'
    ]              # The name of your scipt, and also the command you'll be using for calling it
)
