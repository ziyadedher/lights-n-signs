#!/usr/bin/env python
"""Setup script for the haar part of the project.

Reads metadata and binds from `setup.cfg` to build and generate entry points.
"""
from setuptools import setup, find_packages


if __name__ == "__main__":
    setup(
        # This is the name of your project. The first time you publish this
        # package, this name will be registered for you. It will determine how
        # users can install this project, e.g.:
        #
        # $ pip install sampleproject
        #
        # And where it will live on PyPI: https://pypi.org/project/sampleproject/
        #
        # There are some restrictions on what makes a valid project name
        # specification here:
        # https://packaging.python.org/specifications/core-metadata/#name
        name='haar',  # Required

        # Versions should comply with PEP 440:
        # https://www.python.org/dev/peps/pep-0440/
        #
        # For a discussion on single-sourcing the version across setup.py and the
        # project code, see
        # https://packaging.python.org/en/latest/single_source_version.html
        version='1.0.0',  # Required

        # This is a one-line description or tagline of what your project does. This
        # corresponds to the "Summary" metadata field:
        # https://packaging.python.org/specifications/core-metadata/#summary
        description='haar stuff',  # Optional

        # This should be your name or the name of the organization which owns the
        # project.
        author='Autodrive Lights-n-Signs Team',  # Optional

        # You can just specify package directories manually here if your project is
        # simple. Or you can use find_packages().
        #
        # Alternatively, if you just want to distribute a single Python file, use
        # the `py_modules` argument instead as follows, which will expect a file
        # called `my_module.py` to exist:
        #
        #   py_modules=["my_module"],
        #
        py_modules=["preprocessing", "lns_haar", "preprocessing.mergevec"],

        # This field lists other packages that your project depends on to run.
        # Any package you put here will be installed by pip when your project is
        # installed, so they must be valid existing projects.
        #
        # For an analysis of "install_requires" vs pip's requirements files see:
        # https://packaging.python.org/en/latest/requirements.html
        install_requires=['opencv-python>=2.4.9', 'Augmentor']

    )
