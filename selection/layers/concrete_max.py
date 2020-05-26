import torch
import torch.nn as nn
from selection.layers import utils


# Implicit temperature for the link function to accelerate optimization.
implicit_temp = 1 / 5.0


class ConcreteMax(nn.Module):
    '''
    Input layer that selects features by learning probabilities for independent
    sampling from a Concrete variable, based on [3].

    [3] Learning to Explain: An Information Theoretic Perspective on Model
    Interpretation (Chen et al., 2018)

    Args:
      input_size: number of inputs.
      k: number of features to be selected.
      temperature: temperature for Concrete samples.
      append: whether to append the mask to the input on forward pass.
    '''
    def __init__(self, input_size, k, temperature=10.0, append=False):
        super().__init__()
        self.logits = nn.Parameter(
            torch.randn(1, input_size, dtype=torch.float32, requires_grad=True))
        self.input_size = input_size
        self.k = k
        self.output_size = 2 * input_size if append else input_size
        self.temperature = temperature
        self.append = append

    @property
    def probs(self):
        return (self.logits / implicit_temp).softmax(dim=1)[0]

    def forward(self, x, n_samples=None, return_mask=False):
        # Sample mask.
        n = n_samples if n_samples else 1
        m = self.sample(sample_shape=(n, len(x)))

        # Apply mask.
        x = x * m

        # Post processing.
        if self.append:
            x = torch.cat((x, m), dim=-1)

        if not n_samples:
            x = x.squeeze(0)
            m = m.squeeze(0)

        if return_mask:
            return x, m
        else:
            return x

    def sample(self, n_samples=None, sample_shape=None):
        '''Sample approximate k-hot vectors.'''
        if n_samples:
            sample_shape = torch.Size([n_samples])
        elif not sample_shape:
            raise ValueError('n_samples or sample_shape must be specified')
        samples = utils.concrete_sample(self.logits.repeat(self.k, 1) / implicit_temp,
                                        self.temperature, sample_shape)
        return torch.max(samples, dim=-2).values

    def get_inds(self, **kwargs):
        return torch.argsort(self.logits[0])[-self.k:].cpu().data.numpy()

    def extra_repr(self):
        return 'input_size={}, temperature={}, k={}, append={}'.format(
            self.input_size, self.temperature, self.k, self.append)
