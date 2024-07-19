from setuptools import setup, find_packages

setup(
    name="unravelsports",
    version="0.1.0",
    author="Joris Bekkers",
    author_email="joris@unravelsports.com",
    description="A project to analyze sports event and tracking data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/unravelsports/unravelsports",
    packages=["unravel"] + ["unravel." + pkg for pkg in find_packages("unravel")],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: GNU Affero General Public License version 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "spektral==1.3.1",
        "kloppy @ git+https://github.com/PySport/kloppy.git@9ccbc77c57c2caafaaf04f5aaf090deea8f03b7d",
    ],
    extras_require={
        "test": [
            "pytest==8.2.2",
            "black[jupyter]==24.4.2",
        ]
    },
)
