# ---------------------------------------------------------------------------------
# ROM-COMMA
# Language - Python
# https://github.com/C-O-M-M-A/rom-comma
# Licensed under BSD 3-clause license
# Copyright 2021 ROM-COMMA authors
# ---------------------------------------------------------------------------------

from setuptools import setup
import re, toml

pe_prjfile = toml.load("pyproject.toml")

package_requirements = []
with open("requirements.txt") as file:
    for line in file:
        package_requirements.append(str(re.sub("\n","",line)))

setup(name=pe_prjfile['project']['name'],
      version=pe_prjfile['project']['version'][:3] + '.7.21.1',  # "0".<month digit>.<year digit>.<serial no.> | An unofficial versioning for C4U project.
      description=pe_prjfile['project']['description'],
      url=pe_prjfile['project']['homepage'],
      author='Robert A. Milton, Aaron Yeardley, Solomon F. Brown',
      author_email='s.f.brown@sheffield.ac.uk',
      license='BSD 3-clause',
      python_requires='>=3.5',
      install_requires=package_requirements,
      keywords=pe_prjfile['project']['keywords'],
      classifiers=pe_prjfile['project']['classifiers'],
      zip_safe=False)