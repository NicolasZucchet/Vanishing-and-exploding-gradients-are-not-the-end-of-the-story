# Vanishing and exploding gradients are not the end of the story

This is the code base for the paper:
> **Recurrent neural networks: exploding and vanishing gradients are not the end of the story.** \
Nicolas Zucchet, Antonio Orvieto \
NeurIPS 2024

## Requirements & Installation

To run the code on your own machine, run `pip install -r requirements.txt`. The GPU installation of
JAX can be tricky; further instructions are available on how to install it
[here](https://github.com/google/jax#installation). PyTorch also needs to be installed separately
because of interference issues with jax: install the CPU version of pytorch from
[this page](https://pytorch.org/get-started/locally/).

## Repository Structure

We extend our [minimal LRU](https://github.com/NicolasZucchet/minimal-LRU) to include other RNN architectures as well as to make it easier to ablate parts of the architecure. It thus
has the same structure and we recommend anyone interested in the LRU architecture to have a
look at this repo. Note that we keep the data processing pipeline available there, although we only
experiment on a synthetic teacher student task in our paper.

Directories and files:

```
source/                Source code for models, datasets, etc.
    dataloaders/       Code mainly derived from S4 processing each dataset.
    dataloading.py     Dataloading functions.
    model.py           Defines the different RNN modules, individual layers and entire models.
    train.py           Training loop code.
    train_helpers.py   Functions for optimization, training and evaluation steps.
    utils/             A range of utility functions.
figures/               Notebooks used to generate figures.
    README.md          More detail on the notebooks.
scans/                 Scans used to generate the data for some of the figures.
    README.md          More detail on the scans.
bin/                   Shell scripts for downloading data.
requirements.txt       Requirements for running the code.
run_train.py           Training loop entrypoint.
```
