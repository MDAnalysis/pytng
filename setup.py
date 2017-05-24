import os
import sys

from setuptools import setup, Command, Extension

try:
    from Cython.Build import cythonize
except ImportError:
    print("Need cython for installation")
    sys.exit(1)


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
    exts = []
    exts.append(
        Extension(
            'pytng.pytng', ['pytng/pytng.pyx'],
            libraries=['tng_io'],
            include_dirs=[]))

    return cythonize(exts)

setup(
    name="pytng",
    version='0.1',
    description='minimal Cython wrapper of tng',
    author='Max Linke, Richard J Gowers',
    author_email='max_linke@gmx.de',
    packages=['pytng'],
    cmdclass={'clean': CleanCommand},
    ext_modules=extensions(),
    zip_safe=False)
