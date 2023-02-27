from setuptools import setup, find_packages

setup(
    name='hds',
    version="0.0.1",
    install_requires=[
        "numpy",
        "scipy"
    ],
    description="Analysis tools for hybrid dynamical systems",
    author='Yuu Miino',
    packages=find_packages(),
    license='MIT'
)