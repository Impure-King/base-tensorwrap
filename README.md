# TensorWrap
<div align="center">
<img src="https://raw.githubusercontent.com/google/jax/main/images/jax_logo_250px.png" alt="logo"></img>
</div>

# TensorWrap - Integration of the Big Three

![PyPI version](https://img.shields.io/pypi/v/tensorwrap)

| [**Install guide**](#installation)


## What is TensorWrap?

TensorWrap is high performance neural network library that acts as a wrapper around [JAX](https://github.com/google/jax) (another high performance machine learning library), bringing the familiar feel of the [TensorFlow](https://tensorflow.org) (2.x.x) and [PyTorch](https://pytorch.org) (1.x.x) for users.

TensorWrap works by creating a layer of abstraction over JAX's low level api and introducing TensorFlow + PyTorch component's while supporting Autograd.

This is a personal project, not professionally affliated with Google in anyway. Expect bugs and several incompatibilities/difference from the original libraries.
Please help by trying it out, [reporting
bugs](https://github.com/Impure-King/base-tensorwrap/issues), and letting us know what you
think!

```python
import tensorwrap as tf
from tensorwrap import jnp
from tensorwrap import torch

def predict(params, inputs):
  for W, b in params:
    outputs = jnp.dot(inputs, W) + b
    inputs = jnp.tanh(outputs)  # inputs to the next layer
  return outputs                # no activation on last layer

def loss(params, inputs, targets):
  preds = predict(params, inputs)
  return jnp.sum((preds - targets)**2)

grad_loss = jit(grad(loss))  # compiled gradient evaluation function
perex_grads = jit(vmap(grad_loss, in_axes=(None, 0, 0)))  # fast per-example grads
```

### Contents
* [Current gimmicks](#current-gimmicks)
* [Installation](#installation)
* [Neural net libraries](#neural-network-libraries)
* [Citations](#citations)
* [Reference documentation](#reference-documentation)

## Current Gimmicks

Current models are all compiled by JAX's internal jit, so it won't be possible to view the actual internals of models, especially if it is a Sequential or Functional equations.

### Installation

The device installation of TensorWrap depends on it's backend, being JAX. Thus, our normal install will be covering both the GPU and CPU installation.

```bash
pip install --upgrade pip
pip install --upgrade tensorwrap
```

On Linux, it is often necessary to first update `pip` to a version that supports
`manylinux2014` wheels. Also note that for Linux, we currently release wheels for `x86_64` architectures only, other architectures require building from source. Trying to pip install with other Linux architectures may lead to `jaxlib` not being installed alongside `jax`, although `jax` may successfully install (but fail at runtime). 
**These `pip` installations do not work with Windows, and may fail silently; see
[above](#installation).**

**Note**

The next section is from the JAX installation page, which further details the installation of the GPU and TPU aspects. Therefore, it won't be needed most of the time.

### GPU Cuda Installation Info

If you want to install JAX with both CPU and NVidia GPU support, you must first
install [CUDA](https://developer.nvidia.com/cuda-downloads) and
[CuDNN](https://developer.nvidia.com/CUDNN),
if they have not already been installed. Unlike some other popular deep
learning systems, JAX does not bundle CUDA or CuDNN as part of the `pip`
package.

JAX provides pre-built CUDA-compatible wheels for **Linux only**,
with CUDA 11.4 or newer, and CuDNN 8.2 or newer. Note these existing wheels are currently for `x86_64` architectures only. Other combinations of
operating system, CUDA, and CuDNN are possible, but require [building from
source](https://jax.readthedocs.io/en/latest/developer.html#building-from-source).

* CUDA 11.4 or newer is *required*.
  * Your CUDA installation must be new enough to support your GPU. If you have
    an Ada Lovelace (e.g., RTX 4080) or Hopper (e.g., H100) GPU,
    you must use CUDA 11.8 or newer.
* The supported cuDNN versions for the prebuilt wheels are:
  * cuDNN 8.6 or newer. We recommend using the cuDNN 8.6 wheel if your cuDNN
    installation is new enough, since it supports additional functionality.
  * cuDNN 8.2 or newer.
* You *must* use an NVidia driver version that is at least as new as your
  [CUDA toolkit's corresponding driver version](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions).
  For example, if you have CUDA 11.4 update 4 installed, you must use NVidia
  driver 470.82.01 or newer if on Linux. This is a strict requirement that
  exists because JAX relies on JIT-compiling code; older drivers may lead to
  failures.
  * If you need to use an newer CUDA toolkit with an older driver, for example
    on a cluster where you cannot update the NVidia driver easily, you may be
    able to use the
    [CUDA forward compatibility packages](https://docs.nvidia.com/deploy/cuda-compatibility/)
    that NVidia provides for this purpose.


Next, run

```bash
pip install --upgrade pip
# Installs the wheel compatible with CUDA 11 and cuDNN 8.6 or newer.
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --upgrade tensorwrap
```


The jaxlib version must correspond to the version of the existing CUDA
installation you want to use. You can specify a particular CUDA and CuDNN
version for jaxlib explicitly:

```bash
pip install --upgrade pip

# Installs the wheel compatible with Cuda >= 11.8 and cudnn >= 8.6
pip install "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install tensorwrap
```
```
pip install --upgrade pip

# Installs the wheel compatible with Cuda >= 11.4 and cudnn >= 8.2
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install tensorwrap
```

You can find your CUDA version with the command:

```bash
nvcc --version
```

Some GPU functionality expects the CUDA installation to be at
`/usr/local/cuda-X.X`, where X.X should be replaced with the CUDA version number
(e.g. `cuda-11.8`). If CUDA is installed elsewhere on your system, you can either
create a symlink:

```bash
sudo ln -s /path/to/cuda /usr/local/cuda-X.X
```

Please let us know on [the issue tracker](https://github.com/google/tensorwrap/issues)
if you run into any errors or problems with the prebuilt wheels.

### pip installation: Google Cloud TPU
JAX also provides pre-built wheels for
[Google Cloud TPU](https://cloud.google.com/tpu/docs/users-guide-tpu-vm).
To install JAX along with appropriate versions of `jaxlib` and `libtpu`, you can run
the following in your cloud TPU VM:
```bash
pip install --upgrade pip
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### pip installation: Colab TPU
Colab TPU runtimes come with JAX pre-installed, but before importing JAX you must run the following code to initialize the TPU:
```python
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()
```
Colab TPU runtimes use an older TPU architecture than Cloud TPU VMs, so installing `jax[tpu]` should be avoided on Colab.
If for any reason you would like to update the jax & jaxlib libraries on a Colab TPU runtime, follow the CPU instructions above (i.e. install `jax[cpu]`).

## Neural network libraries

Multiple Google research groups develop and share libraries for training neural
networks in JAX. If you want a fully featured library for neural network
training with examples and how-to guides, try
[Flax](https://github.com/google/flax).

In addition, DeepMind has open-sourced an [ecosystem of libraries around
JAX](https://deepmind.com/blog/article/using-jax-to-accelerate-our-research)
including [Haiku](https://github.com/deepmind/dm-haiku) for neural network
modules, [Optax](https://github.com/deepmind/optax) for gradient processing and
optimization, [RLax](https://github.com/deepmind/rlax) for RL algorithms, and
[chex](https://github.com/deepmind/chex) for reliable code and testing. (Watch
the NeurIPS 2020 JAX Ecosystem at DeepMind talk
[here](https://www.youtube.com/watch?v=iDxJxIyzSiM))

## Citations

This project have been heavily inspired by __TensorFlow__ and once again, is built on the open-source machine learning XLA framework __JAX__. Therefore, I recongnize the authors of JAX and TensorFlow for the exceptional work they have done and understand that my library doesn't profit in any sort of way, since it is merely an add-on to the already existing community.

```
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/google/jax},
  version = {0.3.13},
  year = {2018},
}
```
## Reference documentation

For details about the TensorWrap API, see the
[main documentation] (coming soon!)

For details about JAX, see the
[reference documentation](https://jax.readthedocs.io/).

For documentation on TensorFlow API, see the
[API documentation](https://www.tensorflow.org/api_docs/python/tf)