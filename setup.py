import os
import sys
import subprocess

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

def brew_prefix(package):
    try:
        return subprocess.check_output(
            ["brew", "--prefix", package],
            text=True
        ).strip()
    except Exception:
        return None


is_mac = sys.platform == "darwin"

extra_args = ["-O3"]
extra_link_args = []
include_dirs = ["src", "libs"]
library_dirs = []

if is_mac:
    libomp = brew_prefix("libomp") or "/opt/homebrew/opt/libomp"
    eigen = brew_prefix("eigen") or "/opt/homebrew"
    boost = brew_prefix("boost") or "/opt/homebrew"

    extra_args += [
        "-Xpreprocessor",
        "-fopenmp",
        "-mmacosx-version-min=10.9",
        "-stdlib=libc++",
    ]

    extra_link_args += [
        "-lomp",
        "-mmacosx-version-min=10.9",
        "-stdlib=libc++",
    ]

    include_dirs += [
        f"{libomp}/include",
        f"{eigen}/include/eigen3",
        f"{boost}/include",
        "/opt/homebrew/include",
        "/opt/homebrew/include/eigen3",
    ]

    library_dirs += [
        f"{libomp}/lib",
    ]

else:
    extra_args += ["-fopenmp", "-march=native"]
    extra_link_args += ["-fopenmp"]

    include_dirs += [
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
        "/usr/include/pybind11",
        "/usr/local/include/pybind11",
    ]


ext_modules = [
    Pybind11Extension(
        "cluprop",
        [
            "python/python_wrapper.cpp",
            "src/cluprop.cpp",
            "src/csr_graph.cpp",
            "src/utilities.cpp",
        ],
        depends=[
            "src/cluprop.h",
            "src/csr_graph.h",
            "src/header.h",
            "src/utilities.h",
        ],
        define_macros=[("VERSION_INFO", __version__)],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=extra_args,
        extra_link_args=extra_link_args,
        cxx_std=17,
    ),
]


setup(
    name="cluprop",
    version=__version__,
    license="MIT",
    author='Yingtao Zheng, Hugo Phibbs, Ninh Pham',
    author_email='pham@imada.sdu.dk',
    url='https://github.com/NinhPham/CluProp',
    keywords="density-based clustering, kNN graph, label propagation, DANE",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
