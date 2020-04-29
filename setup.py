from distutils.core import setup
#from setuptools import setup
 
setup(
    name='spherical_stats',   
    version='0.2', 
    description='Spherical statistics in Python',
    author='Daniel Schmitz',
    license='MIT',
    packages=['spherical_stats'],
    url='https://github.com/dschmitz89/spherical_stats',
    install_requires=[
        'numpy',
        'numba>0.44',
        'scipy>0.11'
    ]             
)
