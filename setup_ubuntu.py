import os
import sys

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

use_openmp = sys.platform != 'darwin'
extra_args = ['-std=c++17', '-march=native', '-O3', '-fopenmp']
extra_link_args = []

is_mac = sys.platform == "darwin"

if is_mac:
    omp_compile_args = ["-Xpreprocessor", "-fopenmp"]
    omp_link_args = ["-lomp"]
    omp_include_dirs = ["/opt/homebrew/opt/libomp/include"]
    omp_library_dirs = ["/opt/homebrew/opt/libomp/lib"]
else:
    omp_compile_args = ["-fopenmp"]
    omp_link_args = ["-fopenmp"]
    omp_include_dirs = []
    omp_library_dirs = []

if use_openmp:
    extra_args += ['-fopenmp']
    extra_link_args += ['-fopenmp']
if sys.platform == 'darwin':
    extra_args += ['-mmacosx-version-min=10.9', '-stdlib=libc++']
    os.environ['LDFLAGS'] = '-mmacosx-version-min=10.9'

ext_modules = [
    Pybind11Extension(
        "cluprop",
        ["python/python_wrapper.cpp",
         "src/cluprop.cpp", "src/utilities.cpp"
         ],
        # Example: passing in the version to the compiled code
        define_macros=[("VERSION_INFO", __version__)],
            extra_compile_args=extra_args,
            extra_link_args=extra_link_args,
        include_dirs=['src', '/usr/include/eigen3','/usr/local/include/eigen3', '/usr/include/pybind11','/usr/local/include/pybind11', 'libs']
    ),
]


setup(
    name='cluprop',
    version='0.0.1',
    license='MIT',
    keywords=
    'density-based clustering, label propagation, kNN graph',
    # include_package_data=True,
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
