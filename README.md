# Deep learning feature selection

The **dl-selection** repository contains tools for performing feature selection with deep learning models. It currently has three mechanisms for selecting features, each of which relies on a stochastic relaxation of the feature selection problem.

Each mechanism is a learnable input layer that determines which features to select throughout the course of training.

<!--**Concrete Mask:** selects a user-specified number of features <img src="https://render.githubusercontent.com/render/math?math=k"> by learning a <img src="https://render.githubusercontent.com/render/math?math=k">-hot vector <img src="https://render.githubusercontent.com/render/math?math=m"> for element-wise multiplication with the input <img src="https://render.githubusercontent.com/render/math?math=x">. The layer is composed with a separate network that learns to make predictions using the masked input <img src="https://render.githubusercontent.com/render/math?math=x \odot m">.

**Concrete Selection:** selects a user-specified number of features <img src="https://render.githubusercontent.com/render/math?math=k"> by learning a binary matrix <img src="https://render.githubusercontent.com/render/math?math=M"> that selects features from <img src="https://render.githubusercontent.com/render/math?math=x">. The layer is composed with a separate network that learns to make predictions using the selected features <img src="https://render.githubusercontent.com/render/math?math=Mx">.

**Concrete Gates:** selects features subject to a L<sub>0</sub> penalty by learning binary gates <img src="https://render.githubusercontent.com/render/math?math=m_1, m_2, \ldots&mode=inline"> for each feature. The layer is composed with a separate network that learns to make predictions using the masked input <img src="https://render.githubusercontent.com/render/math?math=x \odot m">.-->

**Concrete Mask:** selects a user-specified number of features `k` by learning a `k`-hot vector `m` for element-wise multiplication with the input `x`. The layer is composed with a separate network that learns to make predictions using the masked input `x * m`.

**Concrete Selection:** selects a user-specified number of features `k` by learning a binary matrix `M` that selects features from `x`. The layer is composed with a separate network that learns to make predictions using the selected features `Mx`.

**Concrete Gates:** selects features subject to a L<sub>0</sub> penalty by learning binary gates `m1, m2, ...` for each feature. The layer is composed with a separate network that learns to make predictions using the masked input `x * m`.

## Usage

The module `selection.models` implements a class `SelectorMLP` for automatically creating a model that composes the user-specified input layer with a prediction network. The model has a built-in `train` method, so it can be used like this:

```python
import torch.nn as nn
from selection import models

train_dataset, val_dataset = ...

model = models.SelectorMLP(
    input_layer='concrete_mask',
    k=20,
    input_size=input_size,
    output_size=output_size,
    hidden=[256, 256],
    activation='elu')
    
model.train(
    train_dataset,
    val_dataset,
    lr=1e-3,
    mbsize=64,
    max_nepochs=100,
    start_temperature=10.0,
    end_temperature=0.01,
    loss_fn=nn.CrossEntropyLoss())
    
inds = model.get_inds()
```

For examples of how to do this with each of the three layers, check out the [mnist selection.ipynb](https://github.com/icc2115/dl-selection/blob/master/mnist%20selection.ipynb) notebook.

## Installation

The easiest way to install this package is with pip:

```
pip install dl-selection
```

Or, you can just clone the repository.

## Authors

- Ian Covert (<icovert@cs.washington.edu>)
- Uygar S&uuml;mb&uuml;l
- Su-In Lee
