from io import open

from setuptools import find_packages
from setuptools import setup


setup(
    name="long-range-arena",
    version="1.0.0",
    url="https://github.com/lucaslingle/long-range-arena/",
    author="Lucas Dax Lingle",
    author_email="lucasdaxlingle@gmail.com",
    description="Working fork of Long Range Arena.",
    long_description=open("README.md", encoding="utf-8").read(),
    packages=find_packages(where="."),
    package_dir={"": "."},
    platforms="any",
    python_requires=">=3.8",
    install_requires=[
        "jax==0.2.4",
        "jaxlib==0.1.56",
        "flax>=0.2.8,<=0.3.6",
        "ml-collections>=0.1.0",
        "tensorboard>=2.3.0",
        "tensorflow>=2.3.1",
        "tensorflow-datasets==4.8.3",
        "tensorflow-hub>=0.15.0",
        "tensorflow-text>=2.7.3",
        "gin-config==0.5.0",
        "attrs==23.1.0",
    ],
)
