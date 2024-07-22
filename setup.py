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
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: GNU Affero General Public License version 3",
        "Operating System :: OS Independent",
    ],
    python_requires="~=3.11",
    install_requires=[
        "spektral==1.2.0",
        "kloppy @ git+https://github.com/PySport/kloppy.git@9ccbc77c57c2caafaaf04f5aaf090deea8f03b7d",
        "tensorflow>=2.14.0;platform_machine != 'arm64' or platform_system != 'Darwin'",
        "tensorflow-macos>=2.14.0;platform_machine == 'arm64' and platform_system == 'Darwin'",
        "keras==2.14.0",
    ],
    extras_require={
        "test": [
            "pytest==8.2.2",
            "black[jupyter]==24.4.2",
        ]
    },
)
