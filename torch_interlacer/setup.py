from setuptools import setup, find_packages

with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="torch-interlacer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
)
