import torch
import torch.nn as nn
from selection.layers import utils


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
