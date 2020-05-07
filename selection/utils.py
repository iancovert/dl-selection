import torch


def clamp_probs(probs):
    eps = torch.finfo(probs.dtype).eps
    return torch.clamp(probs, min=eps, max=1-eps)


def concrete_sample(u_probs, temperature, shape=torch.Size([])):
    uniform_shape = torch.Size(shape) + u_probs.shape
    u = clamp_probs(torch.rand(uniform_shape, dtype=torch.float32,
                               device=u_probs.device))
    gumbels = - torch.log(- torch.log(u))
    scores = (u_probs + gumbels) / temperature
    return scores.softmax(dim=-1)


def concrete_bernoulli_sample(probs, temperature, shape=torch.Size([])):
    probs = clamp_probs(probs)
    uniform_shape = torch.Size(shape) + probs.shape
    u = clamp_probs(torch.rand(uniform_shape, dtype=torch.float32,
                               device=probs.device))
    return torch.sigmoid(
        (torch.log(probs) - torch.log(1 - probs)
         + torch.log(u) - torch.log(1 - u)) / temperature)

