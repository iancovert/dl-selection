# Deep learning feature selection

The **dl-selection** repository contains tools for performing feature selection with deep learning models. It currently has three mechanisms for selecting features, each of which relies on a stochastic relaxation of the feature selection problem. Each mechanism is a learnable input layer that determines which features to select throughout the course of training.

**Concrete Mask:** selects a user-specified number of features `k` by learning a `k`-hot vector `m` for element-wise multiplication with the input `x`. The layer is composed with a separate network that learns to make predictions using the masked input `x * m`.

**Concrete Selection:** selects a user-specified number of features `k` by learning a binary matrix `M` that selects features from `x`. The layer is composed with a separate network that learns to make predictions using the selected features `Mx`.

**Concrete Gates:** selects features subject to a L<sub>0</sub> penalty by learning binary gates `m1, m2, ...` for each feature. The layer is composed with a separate network that learns to make predictions using the masked input `x * m`.

## Usage

The module `selection.models` implements a class `SelectorMLP` for automatically creating a model that composes the user-specified input layer with a prediction network. The model has a built-in `train` method, so it can be used like this:

```python
import torch.nn as nn
from selection import models

# Load data
train_dataset, val_dataset = ...
input_size, output_size = ...

# Set up model
model = models.SelectorMLP(
    input_layer='concrete_mask',
    k=20,
    input_size=input_size,
    output_size=output_size,
    hidden=[256, 256],
    activation='elu')

# Train model
model.learn(
    train_dataset,
    val_dataset,
    lr=1e-3,
    mbsize=64,
    max_nepochs=100,
    start_temperature=10.0,
    end_temperature=0.01,
    loss_fn=nn.CrossEntropyLoss())

# Extract selected indices
inds = model.get_inds()
```

For examples of how to do this with each of the three layers, check out the [mnist selection.ipynb](https://github.com/icc2115/dl-selection/blob/master/mnist%20selection.ipynb) notebook.

## Installation

The easiest way to install this package is with pip:

```
pip install dl-selection
```

Or, you can clone the repository to get the most recent version of the code.

## Authors

- Ian Covert (<icovert@cs.washington.edu>)
- Uygar S&uuml;mb&uuml;l
- Su-In Lee
