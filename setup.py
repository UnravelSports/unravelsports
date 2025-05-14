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
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Operating System :: OS Independent",
    ],
    python_requires="~=3.11",
    install_requires=[
        "spektral==1.2.0",
        "kloppy==3.16.0",
        "tensorflow>=2.14.0;platform_machine != 'arm64' or platform_system != 'Darwin'",
        "tensorflow-macos>=2.14.0;platform_machine == 'arm64' and platform_system == 'Darwin'",
        "keras==2.14.0",
        "polars==1.2.1",
    ],
    extras_require={
        "test": [
            "pytest==8.2.2",
            "black[jupyter]==24.4.2",
            "matplotlib>=3.9",
            "mplsoccer>=1.4",
            "ffmpeg-python==0.2.0",
        ]
    },
)
