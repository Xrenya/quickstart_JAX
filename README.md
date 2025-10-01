# Quickstart: How to think in JAX

The goal of this repo is to make it easier to get started with JAX, Flax, and Haiku!

`JAX` ecosystem is becoming an increasingly popular alternative to PyTorch and TensorFlow. 😎

## JAX Tutorials
### Tutorial #1: From Zero to Hero
The basics and then gradually dig into the nitty-gritty details of jit, grad, vmap, and various other idiosyncrasies of JAX.

[Jupyter Notebook #1](https://github.com/Xrenya/quickstart_JAX/blob/main/Tutorial_1_JAX_Zero2Hero.ipynb)

### Tutorial #2: From Zero to Hero
Train a simple MLP model and we'll even train an ML model on 8 TPU cores (if you have it, just one is enough to test it in google Colab)!

[Jupyter Notebook #2](https://github.com/Xrenya/quickstart_JAX/blob/main/Tutorial_2_JAX_Zero2Hero.ipynb)

### Tutorial #3: Building a Neural Network from Scratch

Build an MLP and train it as a classifier on MNIST using PyTorch's data loader (although it's trivial to use a more complex dataset) - all this in "pure" JAX (no Flax/Haiku/Optax).

[Jupyter Notebook #3](https://github.com/Xrenya/quickstart_JAX/blob/main/Tutorial_3_JAX_Zero2Hero.ipynb)

### Tutorial #4: Machine Learning with Flax - From Zero to Hero
Covering everything you need to know to get started with `Flax`!

We cover init, apply, TrainState, etc. and other idiosyncrasies like the usage of mutable and rngs keywords.

[Jupyter Notebook #4](https://github.com/Xrenya/quickstart_JAX/blob/main/Tutorial_4_JAX_Zero2Hero.ipynb)

### Tutorial #5: Welcome to Flax NNX! - From Zero to Hero
Covering everything you need to know to get started with `Flax NNX API`!

We cover NNX API, train(), eval(), optax, metrics, etc.

[Jupyter Notebook #5](https://github.com/Xrenya/quickstart_JAX/blob/main/Tutorial_5_JAX_Zero2Hero.ipynb)

# References
1. Based on [Get started with JAX! by ](https://github.com/gordicaleksa/get-started-with-JAX) with some minor updates. YouTube videos are available in the repo.
2. [Flax NNX Docs](https://flax.readthedocs.io/en/latest/why.html)
