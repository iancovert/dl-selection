import torch
import torch.nn as nn
from selection import utils


class ConcreteMask(nn.Module):
    def __init__(self, input_size, k, temperature=10.0, append=False):
        super().__init__()
        self.logits = nn.Parameter(
            torch.randn(k, input_size, dtype=torch.float32, requires_grad=True))
        self.input_size = input_size
        self.k = k
        self.output_size = 2 * input_size if append else input_size
        self.temperature = temperature
        self.append = append

    @property
    def u_probs(self):
        return torch._C._nn.softplus(self.logits) ** 2

    @property
    def probs(self):
        u_probs = self.u_probs
        return u_probs / torch.sum(u_probs, dim=1, keepdim=True)

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
        elif not sample_shape:
            raise ValueError('n_samples or sample_shape must be specified')
        samples = utils.concrete_sample(self.u_probs, self.temperature,
                                        sample_shape)
        return torch.clamp(torch.sum(samples, dim=-2), max=1.0)

    def get_inds(self, **kwargs):
        return torch.argmax(self.probs, dim=1).cpu().data.numpy()

    def extra_repr(self):
        return 'input_size={}, temperature={}, k={}, append={}'.format(
            self.input_size, self.temperature, self.k, self.append)


class ConcreteSelector(nn.Module):
    def __init__(self, input_size, k, temperature=10.0):
        super().__init__()
        self.logits = nn.Parameter(
            torch.randn(k, input_size, dtype=torch.float32, requires_grad=True))
        self.input_size = input_size
        self.k = k
        self.output_size = k
        self.temperature = temperature

    @property
    def u_probs(self):
        return torch._C._nn.softplus(self.logits) ** 2

    @property
    def probs(self):
        u_probs = self.u_probs
        return u_probs / torch.sum(u_probs, dim=1, keepdim=True)

    def forward(self, x, n_samples=None, **kwargs):
        # Sample selection matrix.
        n = n_samples if n_samples else 1
        M = self.sample(sample_shape=(n, len(x)))

        # Apply selection matrix.
        x = torch.matmul(x.unsqueeze(1), M.permute(0, 1, 3, 2)).squeeze(2)

        # Post processing.
        if not n_samples:
            x = x.squeeze(0)

        return x

    def sample(self, n_samples=None, sample_shape=None):
        if n_samples:
            sample_shape = torch.Size([n_samples])
        return utils.concrete_sample(self.u_probs, self.temperature,
                                     sample_shape)

    def get_inds(self, **kwargs):
        return torch.argmax(self.probs, dim=1).cpu().data.numpy()

    def extra_repr(self):
        return 'input_size={}, temperature={}, k={}'.format(
            self.input_size, self.temperature, self.k)


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

