from setuptools import setup, find_packages

setup(name='build_vessel', version='1.0.0', packages=find_packages(where="design_automation", exclude=("local", "tests")))