from distutils.core import setup
#from setuptools import setup
 
setup(
    name='spherical_stats',   
    version='1.0', 
    description='Spherical statistics in Python',
    author='Daniel Schmitz',
    author_email='danielschmitzsiegen@gmail.com',
    license='MIT',
    packages=['spherical_stats'],
    url='https://github.com/dschmitz89/spherical_stats',
    download_url = 'https://github.com/dschmitz89/spherical_stats/archive/refs/tags/0.2.tar.gz',
    install_requires=[
        'numpy',
        'numba>0.44',
        'scipy>0.11'
    ],
    include_package_data=True,
    package_data={'': ['tasmanianData.csv']}           
)
