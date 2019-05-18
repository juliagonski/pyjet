#!/usr/bin/env python

import sys
import os

try:
    import builtins
except ImportError:
    import __builtin__ as builtins


from setuptools import setup, Extension
from pkg_resources import parse_version, get_distribution

from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.install import install as _install

if not os.path.exists('pyjet/src/_libpyjet.cpp') or os.path.getmtime('pyjet/src/_libpyjet.pyx') > os.path.getmtime('pyjet/src/_libpyjet.cpp'):
    from Cython.Build import cythonize
    ext = 'pyx'
else:
    cythonize = None
    ext = 'cpp'

import os
import platform
import subprocess
from glob import glob
from distutils.sysconfig import customize_compiler

# Prevent setup from trying to create hard links
# which are not allowed on AFS between directories.
# This is a hack to force copying.
try:
    del os.link
except AttributeError:
    pass

local_path = os.path.dirname(os.path.abspath(__file__))
# setup.py can be called from outside the source directory
os.chdir(local_path)
sys.path.insert(0, local_path)


def fastjet_prefix(fastjet_config='fastjet-config'):
    try:
        prefix = subprocess.Popen(
            [fastjet_config, '--prefix'],
            stdout=subprocess.PIPE).communicate()[0].strip()
    except IOError:
        sys.exit("unable to locate fastjet-config. Is it in your $PATH?")
    if sys.version > '3':
        prefix = prefix.decode('utf-8')
    return prefix


libpyjet = Extension(
    'pyjet._libpyjet',
    sources=['pyjet/src/_libpyjet.' + ext],
    depends=['pyjet/src/fastjet.h'],
    language='c++',
    include_dirs=[
        'pyjet/src',
    ],
    extra_compile_args=[
        '-Wno-unused-function',
        '-Wno-write-strings',
    ])

if cythonize:
    libpyjet, = cythonize(libpyjet)

external_fastjet = False


class build_ext(_build_ext):
    user_options = _build_ext.user_options + [
        ('external-fastjet', None, None),
    ]

    def initialize_options(self):
        _build_ext.initialize_options(self)
        self.external_fastjet = False

    def finalize_options(self):
        global libpyjet
        global external_fastjet
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process
        try:
            del builtins.__NUMPY_SETUP__
        except AttributeError:
            pass
        import numpy
        libpyjet.include_dirs.append(numpy.get_include())
        if external_fastjet or self.external_fastjet:
            prefix = fastjet_prefix()
            libpyjet.include_dirs += [os.path.join(prefix, 'include')]
            libpyjet.library_dirs = [os.path.join(prefix, 'lib')]
            libpyjet.runtime_library_dirs = libpyjet.library_dirs
            libpyjet.libraries = 'fastjettools fastjet CGAL gmp'.split()
            if platform.system() == 'Darwin':
                libpyjet.extra_link_args.append(
                    '-Wl,-rpath,' + os.path.join(prefix, 'lib'))
        else:
            if 'pyjet/src/fjcore.cpp' not in libpyjet.sources:
                libpyjet.sources.append('pyjet/src/fjcore.cpp')
                libpyjet.depends.append('pyjet/src/fjcore.h')
                libpyjet.define_macros = [('PYJET_STANDALONE', None)]

    def build_extensions(self):
        # Remove the "-Wstrict-prototypes" compiler option, which isn't valid
        # for C++.
        customize_compiler(self.compiler)
        try:
            self.compiler.compiler_so.remove('-Wstrict-prototypes')
        except (AttributeError, ValueError):
            pass
        _build_ext.build_extensions(self)


class install(_install):
    user_options = _install.user_options + [
        ('external-fastjet', None, None),
    ]

    def initialize_options(self):
        _install.initialize_options(self)
        self.external_fastjet = False

    def finalize_options(self):
        global external_fastjet
        if self.external_fastjet:
            external_fastjet = True
        _install.finalize_options(self)


# Only add numpy to *_requires lists if not already installed to prevent
# pip from trying to upgrade an existing numpy and failing.
try:
    import numpy
except ImportError:
    build_requires = ['numpy']
else:
    build_requires = []

setup(
    name='pyjet',
    version='1.5.0',
    description='The interface between FastJet and NumPy',
    long_description=''.join(open('README.rst').readlines()),
    author='Noel Dawe',
    author_email='noel@dawe.me',
    maintainer='the Scikit-HEP admins',
    maintainer_email='scikit-hep-admins@googlegroups.com',
    license='GPLv3',
    url='http://github.com/scikit-hep/pyjet',
    packages=[
        'pyjet',
        'pyjet.tests',
        'pyjet.testdata',
    ],
    package_data={
        'pyjet': [
            'testdata/*.dat',
            'src/*.pxd', 'src/*.h', 'src/*.cpp',
        ],
    },
    ext_modules=[libpyjet],
    cmdclass={
        'build_ext': build_ext,
        'install': install,
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Programming Language :: Cython',
        'Development Status :: 5 - Production/Stable',
    ],
    setup_requires=build_requires,
    install_requires=build_requires,
    extras_require={
        'with-numpy': ('numpy',),
    },
    zip_safe=False,
)
