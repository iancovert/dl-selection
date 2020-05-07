# Deep learning feature selection

The **dl-selection** repository contains tools for performing feature selection with deep learning models. It currently has three mechanisms for selecting features, each of which relies on a stochastic relaxation of the feature selection problem.

**Concrete Mask:** this layer selects a user-specified number of features $k$ by learning a $k$-hot vector $m$ for element-wise multiplication with the input $x$. The layer is composed with a separate network that learns to make predictions using the masked input $x \odot m$.

**Concrete Selection:** this layer selects a user-specified number of features $k$ by learning a binary matrix $M$ that selects features from $x$. The layer is composed with a separate network that learns to make predictions using the selected features $Mx$.

**Concrete Gates:** this layer selects features subject to a $\ell_0$ penalty by learning binary per-feature gates $m_1, m_2, ...,$ etc. The layer is composed with a separate network that learns to make predictions using the masked input $x \odot m$.

## Install

Please clone the Github repository to use this code. The only packages required are `numpy`, `torch` and `torchvision`. The tools are currently implemented in PyTorch, but we may add support for other frameworks in the future.

## Usage

The `mnist selection.ipnyb` notebook shows how each of the three mechanisms are used.

The `models` module implements a class `SelectionMLP` that automatically creates a network that composes the user-specified input layer with a prediction network. The network can be trained using the `train()` method (which is implemented in `models/train.py`).

## Authors

- Ian Covert (<icovert@cs.washington.edu>)
- Uygar S&uuml;mb&uuml;l
- Su-In Lee
