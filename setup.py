# pylint:disable=missing-docstring
from setuptools import setup, find_packages


def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


setup(
    name="gym-cartpole-swingup",
    version="0.0.6",
    author="Ângelo G. Lovatto",
    author_email="angelolovatto@gmail.com",
    description="A simple, continuous-control environment for OpenAI Gym",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    license="GNU General Public License v3.0",
    url="https://github.com/angelolovatto/gym-cartpole-swingup",
    packages=find_packages(),
    scripts=[],
    install_requires=["gym"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
