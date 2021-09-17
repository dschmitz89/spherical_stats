from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
 
setup(
    name='spherical_stats',   
    version='1.1', 
    description='Spherical statistics in Python',
    author='Daniel Schmitz',
    author_email='danielschmitzsiegen@gmail.com',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['spherical_stats'],
    url='https://github.com/dschmitz89/spherical_stats',
    install_requires=[
        'numpy',
        'numba>0.44',
        'scipy>0.11'
    ],
    include_package_data=True,
    package_data={'': ['tasmanianData.csv']}           
)
