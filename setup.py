import os
import sys
from glob import glob

from setuptools import setup, Command, Extension

try:
    import numpy as np
except ImportError:
    print("Need numpy for installation")
    sys.exit(1)

try:
    from Cython.Build import cythonize
except ImportError:
    print("Need cython for installation")
    sys.exit(1)

try:
    with open("README.rst", "r") as handle:
        long_description = handle.read()
except:
    long_description = "Minimal Cython wrapper of the TNG trajectory library"

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    # https://stackoverflow.com/questions/3779915/why-does-python-setup-py-sdist-create-unwanted-project-egg-info-in-project-r
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./*.so')
        os.system('rm -vrf build')
        os.system('rm -vrf dist')
        os.system('rm -vrf pytng.egg-info')
        os.system("find pytng -name '*.pyc' -delete -print")
        os.system("find pytng -name '*.so' -delete -print")


def extensions():
    """ setup extensions for this module
    """
    exts = []
    exts.append(
        Extension(
            'pytng.pytng',
            sources=glob('pytng/src/compression/*.c') + glob(
                'pytng/src/lib/*.c') + glob('pytng/src/external/*.c') + glob('pytng/src/external/zlib/*.c') + ['pytng/pytng.pyx'],
            include_dirs=[
                "pytng/include/", "pytng/include/external/", "{}/include".format(sys.prefix),
                np.get_include()
            ],
            library_dirs=["{}/lib".format(sys.prefix)]))
    return cythonize(exts, gdb_debug=False)


setup(
    name="pytng",
    python_requires=">=3.9",
    install_requires=['numpy>=1.22.3'],
    description='Minimal Cython wrapper of the TNG trajectory library',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    author='Max Linke, Richard J Gowers, Hugo MacDermott-Opeskin',
    author_email='max_linke@gmx.de',
    packages=['pytng'],
    ext_modules=extensions(),
    zip_safe=False)
