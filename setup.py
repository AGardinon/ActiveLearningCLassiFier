from setuptools import setup, find_packages

setup(
    name='activeclf',
    version='1.0.0',
    packages=find_packages(['activeclf', 'activeclf.*']),
    url='',
    license='MIT',
    author='Andrea Gardin',
    author_email='',
    description='Active classifier tool for supervised learing problems.',
    install_requires=['numpy', 'pandas', 'scikit-learn']
)