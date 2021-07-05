# ---------------------------------------------------------------------------------
# ROM-COMMA
# Language - Python
# https://github.com/C-O-M-M-A/rom-comma
# Licensed under BSD license
# Copyright 2021 ROM-COMMA authors
# ---------------------------------------------------------------------------------

from setuptools import setup
import re

package_requirements = []
with open("requirements.txt") as file:
    for line in file:
        package_requirements.append(str(re.sub("\n","",line)))

setup(name='ROM-COMMA',
      version='0.7.21.1',  # "0".<month digit>.<year digit>.<serial no.> | An unofficial versioning for C4U project.
      description="Reduced Order Modelling software produced by Solomon F. Brown's COMMA "
                  "Research Group at The University of Sheffield",
      url='https://github.com/C-O-M-M-A/rom-comma',
      author='Robert A. Milton, Aaron Yeardley, Solomon F. Brown',
      author_email='s.f.brown@sheffield.ac.uk',
      license='BSD',
      python_requires='>=3.5',
      install_requires=package_requirements,
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        # 'Programming Language :: Python :: Implementation :: Jython',
        # 'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules'
        ],
      zip_safe=False)