import torch
import torch.nn as nn
from selection.layers import utils


class ConcreteGates(nn.Module):
    def __init__(self, input_size, temperature=1.0, init=0.01, append=False):
        super().__init__()
        # init_logit = - temperature * torch.log(1 / torch.tensor(init) - 1)
        init_logit = - torch.log(1 / torch.tensor(init) - 1)
        self.logits = nn.Parameter(torch.full(
            (input_size,), init_logit, dtype=torch.float32, requires_grad=True))
        self.input_size = input_size
        self.output_size = 2 * input_size if append else input_size
        self.temperature = temperature
        self.append = append

    @property
    def probs(self):
        # return torch.sigmoid(self.logits / self.temperature)
        return torch.sigmoid(self.logits)

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
        if n_samples:
            sample_shape = torch.Size([n_samples])
        return utils.concrete_bernoulli_sample(1 - self.probs, self.temperature,
                                               sample_shape)

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
