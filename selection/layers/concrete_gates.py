import torch
import torch.nn as nn
from selection.layers import utils


# Implicit temperature for the link function to accelerate optimization.
implicit_temp = 1 / 2.0


class ConcreteGates(nn.Module):
    '''
    Input layer that selects features by learning binary gates for each feature,
    based on [1].

    [1] Dropout Feature Ranking for Deep Learning Models (Chang et al., 2017)

    Args:
      input_size: number of inputs.
      k: number of features to be selected.
      temperature: temperature for Concrete samples.
      init: initial value for each gate's probability of being 1.
      append: whether to append the mask to the input on forward pass.
    '''
    def __init__(self, input_size, temperature=1.0, init=0.01, append=False):
        super().__init__()
        init_logit = - torch.log(1 / torch.tensor(init) - 1) * implicit_temp
        self.logits = nn.Parameter(torch.full(
            (input_size,), init_logit, dtype=torch.float32, requires_grad=True))
        self.input_size = input_size
        self.output_size = 2 * input_size if append else input_size
        self.temperature = temperature
        self.append = append

    @property
    def probs(self):
        return torch.sigmoid(self.logits / implicit_temp)

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
        return utils.bernoulli_concrete_sample(- self.logits / implicit_temp,
                                               self.temperature, sample_shape)

    def get_inds(self, num_features=None, threshold=None, **kwargs):
        if num_features:
            return torch.argsort(self.probs)[-num_features:].cpu().data.numpy()
        elif threshold:
            return (self.probs < threshold).nonzero()[:, 0].cpu().data.numpy()
        else:
            raise ValueError('num_features or p_threshold must be specified')

    def extra_repr(self):
        return 'input_size={}, temperature={}, append={}'.format(
            self.input_size, self.temperature, self.append)
