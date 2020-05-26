import torch.nn as nn
import selection.layers as layers
from selection.models import utils
from selection.models import train
from torch.utils.data import DataLoader


class MLP(nn.Module):
    '''
    Multilayer perceptron (MLP) model.

    Args:
      input_size: number of inputs.
      output_size: number of outputs.
      hidden: list of hidden layer widths.
      activation: nonlinearity between hidden layers.
      output_activation: nonlinearity at output.
    '''
    def __init__(self,
                 input_size,
                 output_size,
                 hidden,
                 activation,
                 output_activation=None,
                 batch_norm=False):
        super().__init__()

        # Fully connected layers.
        self.input_size = input_size
        self.output_size = output_size
        fc_layers = [nn.Linear(d_in, d_out) for d_in, d_out in
                     zip([input_size] + hidden, hidden + [output_size])]
        self.fc = nn.ModuleList(fc_layers)

        # Activation functions.
        self.activation = utils.get_activation(activation)
        self.output_activation = utils.get_activation(output_activation)

        # Set up batch norm.
        if batch_norm:
            layer_normalizers = [nn.BatchNorm1d(d) for d in hidden]
        else:
            layer_normalizers = [nn.Identity() for d in hidden]
        self.layer_normalizers = nn.ModuleList(layer_normalizers)

        # Set up training.
        self.learn = train.Training(self)

    def forward(self, x):
        for fc, norm in zip(self.fc, self.layer_normalizers):
            x = fc(x)
            x = self.activation(x)
            x = norm(x)

        return self.output_activation(self.fc[-1](x))

    def evaluate(self, dataset, loss_fn, mbsize=None):
        training = self.training
        self.eval()
        mbsize = mbsize if mbsize else len(dataset)
        loader = DataLoader(dataset, batch_size=mbsize)
        loss = utils.validate(self, loader, loss_fn)
        if training:
            self.train()
        return loss

    def extra_repr(self):
        return 'hidden={}'.format([fc.in_features for fc in self.fc[1:]])


class SelectorMLP(nn.Module):
    '''MLP with input layer selection.

    Args:
      input_layer: input layer type (e.g., 'concrete_gates').
      input_size: number of inputs.
      output_size: number of outputs.
      hidden: list of hidden layer widths.
      activation: nonlinearity between hidden layers.
      output_activation: nonlinearity at output.
      kwargs: additional arguments (e.g., k, init, append). Some are optional,
        but k is required for ConcreteMask and ConcreteGates.
    '''
    def __init__(self,
                 input_layer,
                 input_size,
                 output_size,
                 hidden,
                 activation,
                 output_activation=None,
                 batch_norm=False,
                 **kwargs):
        super().__init__()

        # Set up input layer.
        if input_layer == 'concrete_mask':
            k = kwargs.get('k')
            append = kwargs.get('append', True)
            kwargs['append'] = append
            mlp_input_size = 2 * input_size if append else input_size
            self.input_layer = layers.ConcreteMask(input_size, **kwargs)
        elif input_layer == 'concrete_selector':
            k = kwargs.get('k')
            mlp_input_size = k
            self.input_layer = layers.ConcreteSelector(input_size, **kwargs)
        elif input_layer == 'concrete_gates':
            append = kwargs.get('append', True)
            kwargs['append'] = append
            mlp_input_size = 2 * input_size if append else input_size
            self.input_layer = layers.ConcreteGates(input_size, **kwargs)
        elif input_layer == 'concrete_max':
            append = kwargs.get('append', True)
            kwargs['append'] = append
            mlp_input_size = 2 * input_size if append else input_size
            self.input_layer = layers.ConcreteMax(input_size, **kwargs)
        else:
            raise ValueError('unsupported input layer: {}'.format(input_layer))

        # Set up MLP.
        self.mlp = MLP(mlp_input_size, output_size, hidden, activation,
                       output_activation, batch_norm)

        # Set up training.
        self.learn = train.AnnealedTemperatureTraining(self)

    def forward(self, x, **kwargs):
        return_mask = kwargs.get('return_mask', False)
        if return_mask:
            assert (
                isinstance(self.input_layer, layers.ConcreteMask) or
                isinstance(self.input_layer, layers.ConcreteGates))
            x, m = self.input_layer(x, **kwargs)
            return self.mlp(x), m
        else:
            return self.mlp(self.input_layer(x, **kwargs))

    def evaluate(self, dataset, loss_fn, mbsize=None, **kwargs):
        training = self.training
        self.eval()
        mbsize = mbsize if mbsize else len(dataset)
        loader = DataLoader(dataset, batch_size=mbsize)
        loss = utils.validate_input_layer(self, loader, loss_fn, **kwargs)
        if training:
            self.train()
        return loss

    def get_inds(self, **kwargs):
        return self.input_layer.get_inds(**kwargs)
