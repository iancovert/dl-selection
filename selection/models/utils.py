import torch
import torch.nn as nn
import torch.optim as optim
import selection.layers as layers


class MSELoss(nn.Module):
    '''MSE loss that sums over output dimensions and allows weights.'''
    def __init__(self, reduction='mean'):
        super().__init__()
        assert reduction in ('none', 'mean')
        self.reduction = reduction

    def forward(self, pred, target, weights=None):
        if weights is not None:
            loss = torch.sum(weights * ((pred - target) ** 2), dim=-1)
        else:
            loss = torch.sum((pred - target) ** 2, dim=-1)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss


class Accuracy(nn.Module):
    '''0-1 loss.'''
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return (torch.argmax(pred, dim=1) == target).float().mean()


def get_activation(activation):
    '''Get activation function.'''
    if activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation is None:
        return nn.Identity()
    else:
        raise ValueError('unsupported activation: {}'.format(activation))


def get_optimizer(optimizer, params, lr):
    '''Get optimizer.'''
    if optimizer == 'SGD':
        return optim.SGD(params, lr=lr)
    elif optimizer == 'Momentum':
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    elif optimizer == 'Adam':
        return optim.Adam(params, lr=lr)
    elif optimizer == 'Adagrad':
        return optim.Adagrad(params, lr=lr)
    elif optimizer == 'RMSprop':
        return optim.RMSprop(params, lr=lr)
    else:
        raise ValueError('unsupported optimizer: {}'.format(optimizer))


def validate(model, loader, loss_fn):
    '''Calculate average loss.'''
    device = next(model.parameters()).device
    mean_loss = 0
    N = 0
    for x, y in loader:
        # Move to GPU.
        x = x.to(device=device)
        y = y.to(device=device)
        n = len(x)

        # Calculate loss.
        loss = loss_fn(model(x), y)
        mean_loss = (N * mean_loss + n * loss) / (N + n)
        N += n

    return mean_loss


def validate_input_layer(model, loader, loss_fn, n_samples=None,
                         mask_output=False):
    '''Calculate average loss.'''
    device = next(model.parameters()).device
    mean_loss = 0
    N = 0
    for x, y in loader:
        # Move to GPU.
        x = x.to(device=device)
        y = y.to(device=device)
        n = len(x)

        # Forward pass.
        if mask_output:
            pred, m = model(x, n_samples=n_samples, return_mask=True)
        else:
            pred = model(x, n_samples=n_samples)

        # Calculate loss.
        if mask_output:
            loss = loss_fn(pred, y, weights=1-m)
        else:
            loss = loss_fn(pred, y)
        mean_loss = (N * mean_loss + n * loss) / (N + n)
        N += n

    return mean_loss


def input_layer_converged(input_layer, tol=1e-3, n_samples=None):
    '''Determine whether the input layer has converged.'''
    with torch.no_grad():
        if isinstance(input_layer, layers.ConcreteMask):
            m = input_layer.sample(n_samples=n_samples)
            mean = torch.mean(m, dim=0)
            return torch.sort(mean).values[-input_layer.k] > 1 - tol

        elif isinstance(input_layer, layers.ConcreteSelector):
            M = input_layer.sample(n_samples=n_samples)
            mean = torch.mean(M, dim=0)
            return torch.min(torch.max(mean, dim=1).values) > 1 - tol

        elif isinstance(input_layer, layers.ConcreteGates):
            m = input_layer.sample(n_samples=n_samples)
            mean = torch.mean(m, dim=0)
            return torch.max(torch.min(mean, 1 - mean)) < tol

        elif isinstance(input_layer, layers.ConcreteMax):
            return False


def input_layer_fix(input_layer):
    '''Fix collisions in the input layer.'''
    needs_reset = (
        isinstance(input_layer, layers.ConcreteMask) or
        isinstance(input_layer, layers.ConcreteSelector))
    if needs_reset:
        # Extract logits.
        logits = input_layer.logits
        argmax = torch.argmax(logits, dim=1).cpu().data.numpy()

        # Locate collisions and reinitialize.
        for i in range(len(argmax) - 1):
            if argmax[i] in argmax[i+1:]:
                logits.data[i] = torch.randn(
                    logits[i].shape, dtype=logits.dtype, device=logits.device)


def input_layer_penalty(input_layer, m):
    '''Calculate the regularization term for the input layer.'''
    assert isinstance(input_layer, layers.ConcreteGates)
    return torch.mean(torch.sum(m, dim=-1))


def input_layer_summary(input_layer, n_samples=None):
    '''Provide a short summary of the input layer's convergence.'''
    with torch.no_grad():
        if isinstance(input_layer, layers.ConcreteMask):
            m = input_layer.sample(n_samples=n_samples)
            mean = torch.mean(m, dim=0)
            relevant = torch.sort(mean, descending=True).values[:input_layer.k]
            return 'Max = {:.2f}, Mean = {:.2f}, Min = {:.2f}'.format(
                relevant[0].item(), torch.mean(relevant).item(),
                relevant[-1].item())

        elif isinstance(input_layer, layers.ConcreteSelector):
            M = input_layer.sample(n_samples=n_samples)
            mean = torch.mean(M, dim=0)
            relevant = torch.max(mean, dim=1).values
            return 'Max = {:.2f}, Mean = {:.2f}, Min = {:.2f}'.format(
                torch.max(relevant).item(), torch.mean(relevant).item(),
                torch.min(relevant).item())

        elif isinstance(input_layer, layers.ConcreteGates):
            m = input_layer.sample(n_samples=n_samples)
            mean = torch.mean(m, dim=0)
            dist = torch.min(mean, 1 - mean)
            return 'Mean dist = {:.2f}, Max dist = {:.2f}, Num sel = {}'.format(
                torch.mean(dist).item(),
                torch.max(dist).item(),
                int(torch.sum((mean > 0.5).float()).item()))

        elif isinstance(input_layer, layers.ConcreteMax):
            m = input_layer.sample(n_samples=n_samples)
            mean = torch.mean(m, dim=0)
            relevant = torch.sort(mean, descending=True).values[:input_layer.k]
            return 'Max = {:.2f}, Mean = {:.2f}, Min = {:.2f}'.format(
                relevant[0].item(), torch.mean(relevant).item(),
                relevant[-1].item())


def restore_parameters(model, best_model):
    '''Move parameter values from best_model to model.'''
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params
