import torch
import torch.nn.functional as F


def clamp_probs(probs):
    eps = torch.finfo(probs.dtype).eps
    return torch.clamp(probs, min=eps, max=1-eps)

def concrete_sample(logits, temperature, shape=torch.Size([])):
    '''
    Sampling for Concrete distribution.

    See Eq. 10 of Maddison et al., 2017.
    '''
    uniform_shape = torch.Size(shape) + logits.shape
    u = clamp_probs(torch.rand(uniform_shape, dtype=torch.float32,
                               device=logits.device))
    gumbels = - torch.log(- torch.log(u))
    scores = (logits + gumbels) / temperature
    return scores.softmax(dim=-1)

def bernoulli_concrete_sample(logits, temperature, shape=torch.Size([])):
    '''
    Sampling for BinConcrete distribution.

    See PyTorch source code, differs from Eq. 16 of Maddison et al., 2017.
    '''
    uniform_shape = torch.Size(shape) + logits.shape
    u = clamp_probs(torch.rand(uniform_shape, dtype=torch.float32,
                               device=logits.device))
    return torch.sigmoid((F.logsigmoid(logits) - F.logsigmoid(-logits)
                          + torch.log(u) - torch.log(1 - u)) / temperature)
