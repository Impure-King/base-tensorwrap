"""setup.py for TensorWrap"""

import os
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
try:
  README = open(os.path.join(here, "README.md"), encoding="utf-8").read()
except OSError:
  README = ""

install_requires = [
    "numpy>=1.12",
    "jax>=0.3.16",
    "matplotlib",  # only needed for tensorboard export
    "msgpack",
    "optax",
    "orbax",
    "tensorstore",
    "rich>=11.1",
    "typing_extensions>=4.1.1",
    "PyYAML>=5.4.1",
]

tests_require = [
    "atari-py==0.2.5",  # Last version does not have the ROMs we test on pre-packaged
    "clu",  # All examples.
    "gym==0.18.3",
    "jaxlib",
    "jraph>=0.0.6dev0",
    "ml-collections",
    "mypy",
    "opencv-python",
    "pytest",
    "pytest-cov",
    "pytest-custom_exit_code",
    "pytest-xdist==1.34.0",  # upgrading to 2.0 broke tests, need to investigate
    "pytype",
    "sentencepiece",  # WMT example.
    "tensorflow_text>=2.4.0",  # WMT example.
    "tensorflow_datasets",
    "tensorflow",
    "torch",
]

__version__ = None

with open("tensorwrap/version.py") as f:
  exec(f.read(), globals())

setup(
    name="tensorwrap",
    version=__version__,
    description="TensorWrap: A high level TensorFlow wrapper for JAX.",
    long_description="\n\n".join([README]),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: MIT License",
        "Programming Language :: Python :: 3.10.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    keywords="",
    author="Lelouch",
    author_email="ImpureK@gmail.com",
    url="https://github.com/google/flax",
    packages=find_packages(),
    package_data={"flax": ["py.typed"]},
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
        "testing": tests_require,
        },
    )
