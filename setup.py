from setuptools import setup, find_packages
import os
import re
import codecs


# Read the version from the __init__.py file
def read_version():
    version_file = os.path.join(os.path.dirname(__file__), "unravel", "__init__.py")
    with codecs.open(version_file, "r", "utf-8") as f:
        version_match = re.search(r'^__version__ = ["\']([^"\']*)["\']', f.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


# Define test dependencies based on Python version
test_deps_common = [
    "pytest==8.2.2",
    "black[jupyter]==24.4.2",
    "matplotlib>=3.9",
    "mplsoccer>=1.4",
    "ffmpeg-python==0.2.0",
]

test_deps_py311_spektral = [
    "spektral==1.2.0",
    "keras==2.14.0",
    "tensorflow>=2.14.0;platform_machine != 'arm64' or platform_system != 'Darwin'",
    "tensorflow-macos>=2.14.0;platform_machine == 'arm64' and platform_system == 'Darwin'",
]

test_deps_torch = [
    "torch>=2.5.0",
    "torch-geometric>=2.6.0",
    "torchmetrics>=1.0.0",
    "pytorch-lightning>=2.0.0",
]

setup(
    name="unravelsports",
    version=read_version(),
    author="Joris Bekkers",
    author_email="joris@unravelsports.com",
    description="A project to analyze sports event and tracking data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/unravelsports/unravelsports",
    packages=["unravel"] + ["unravel." + pkg for pkg in find_packages("unravel")],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=["kloppy>=3.18.0", "polars[numpy]>=1.35.0", "scipy>=1.0.0"],
    extras_require={
        # Full test suite with all dependencies (for Python 3.11)
        "test": test_deps_common + test_deps_py311_spektral + test_deps_torch,
        # Python 3.11 only - Spektral + common test deps
        "test-py311": test_deps_common + test_deps_py311_spektral + test_deps_torch,
        # Python 3.12+ - PyTorch only + common test deps
        "test-torch": test_deps_common + test_deps_torch,
    },
)
