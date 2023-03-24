from setuptools import setup, find_packages

setup(
    name='MapPy',
    version="0.0.2",
    install_requires=[
        "numpy",
        "scipy",
        "simpy"
    ],
    description="Analysis tools for hybrid dynamical systems",
    author='Yuu Miino',
    packages=find_packages(),
    license='MIT'
)