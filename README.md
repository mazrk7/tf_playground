# TensorFlow Playground

Playground for designing and implementing various learning models on TensorFlow.

The directory layout it as follows:
- `basic`: Very simple regression models
- `nn`: Deep neural networks applied to the MNIST dataset
- `rnn`: Recurrent models applied to a real-world dataset collected from a teleoperated navigation experiment on a robotic wheelchair
- `vae`: Variational autoencoders applied to the MNIST dataset

Many of the scripts included in this repository are adapted from the [TensorFlow tutorials](https://www.tensorflow.org/tutorials/) page and other useful online sources, notably [this blog](https://jmetzen.github.io/2015-11-27/vae.html).

The `dataset.py` script is an attempt at creating a general template for learning over different datasets.