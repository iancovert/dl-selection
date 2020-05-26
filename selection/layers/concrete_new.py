import torch
import torch.nn as nn
from selection.layers import utils


class ConcreteNew(nn.Module):
    '''
    Input layer that selects features by learning a k-hot vector.

    Args:
      input_size: number of inputs.
      k: number of features to be selected.
      temperature: temperature for Concrete samples.
    '''
    def __init__(self, input_size, k, temperature=10.0, append=False):
        super().__init__()
        self.logits = nn.Parameter(
            torch.randn(k, input_size, dtype=torch.float32, requires_grad=True))
        self.input_size = input_size
        self.k = k
        self.output_size = k
        self.temperature = temperature
        self.append = append

    @property
    def probs(self):
        probs = torch.softmax(self.logits / self.temperature, dim=1)
        return torch.clamp(torch.sum(probs, dim=0), max=1.0)

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
        '''Sample approximate binary masks.'''
        if n_samples:
            sample_shape = torch.Size([n_samples])
        return utils.concrete_bernoulli_sample(self.probs, self.temperature,
                                               sample_shape)

    def get_inds(self, **kwargs):
        return torch.argsort(self.probs)[-self.k:].cpu().data.numpy()

    def extra_repr(self):
        return 'input_size={}, temperature={}, k={}'.format(
            self.input_size, self.temperature, self.k)
