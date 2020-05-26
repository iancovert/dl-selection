import torch
import numpy as np
import selection.layers as layers
from selection.models import utils
from copy import deepcopy
from torch.utils.data import DataLoader


class Training:
    '''
    Class for training PyTorch models.

    Args:
      model: the model to be trained.
    '''
    def __init__(self, model):
        self.model = model
        self.trained = False

    def __call__(self,
                 train_dataset,
                 val_dataset,
                 lr,
                 mbsize,
                 max_nepochs,
                 loss_fn,
                 optimizer='Adam',
                 lookback=5,
                 check_every=1,
                 verbose=True):
        '''
        Train the model.

        Args:
          train_dataset: training dataset.
          val_dataset: validation dataset.
          lr: learning rate.
          mbsize: minibatch size.
          max_nepochs: maximum number of epochs.
          loss_fn: loss function.
          optimizer: optimizer type.
          lookback: number of epochs to wait for improvement before stopping.
          check_every: number of epochs between loss value checks.
          verbose: verbosity.
        '''
        # Ensure model has not yet been trained.
        assert not self.trained
        self.trained = True

        # Set up optimizer.
        optimizer = utils.get_optimizer(optimizer, self.model.parameters(), lr)

        # Set up data loaders.
        train_loader = DataLoader(train_dataset, batch_size=mbsize,
                                  shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=mbsize)

        # Determine device.
        device = next(self.model.parameters()).device

        # For tracking loss.
        self.train_loss = []
        self.val_loss = []
        best_model = None
        best_loss = np.inf
        best_epoch = None

        # Begin training.
        for epoch in range(max_nepochs):
            for x, y in train_loader:
                # Move to device.
                x = x.to(device)
                y = y.to(device)

                # Forward pass.
                pred = self.model(x)

                # Calculate loss.
                loss = loss_fn(pred, y)

                # Gradient step.
                loss.backward()
                optimizer.step()
                self.model.zero_grad()

            # Check progress.
            with torch.no_grad():
                # Calculate loss.
                self.model.eval()
                train_loss = utils.validate(
                    self.model, train_loader, loss_fn).item()
                val_loss = utils.validate(
                    self.model, val_loader, loss_fn).item()
                self.model.train()

                # Record loss.
                self.train_loss.append(train_loss)
                self.val_loss.append(val_loss)

                if verbose and ((epoch + 1) % check_every == 0):
                    print('{}Epoch = {}{}'.format('-' * 8, epoch + 1, '-' * 8))
                    print('Train loss = {:.4f}'.format(train_loss))
                    print('Val loss = {:.4f}'.format(val_loss))

            # Check for early stopping.
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(self.model)
                best_epoch = epoch
            elif (epoch - best_epoch) > lookback:
                if verbose:
                    print('Stopping early')
                break

        # Restore model parameters.
        utils.restore_parameters(self.model, best_model)


class AnnealedTemperatureTraining:
    '''
    Class for training PyTorch models with a temperature parameter.

    Args:
      model: the model to be trained.
    '''
    def __init__(self, model):
        self.model = model
        self.trained = False

    def __call__(self,
                 train_dataset,
                 val_dataset,
                 lr,
                 mbsize,
                 max_nepochs,
                 start_temperature,
                 end_temperature,
                 loss_fn,
                 optimizer='Adam',
                 check_every=1,
                 verbose=True,
                 **kwargs):
        '''
        Train the model.

        Args:
          train_dataset: training dataset.
          val_dataset: validation dataset.
          lr: learning rate.
          mbsize: minibatch size.
          max_nepochs: maximum number of epochs.
          start_temperature:
          end_temperature:
          loss_fn: loss function.
          optimizer: optimizer type.
          lookback: number of epochs to wait for improvement before stopping.
          check_every: number of epochs between loss value checks.
          verbose: verbosity.
          kwargs: additional arguments (e.g. n_samples, mask_output, lam). These
            are optional, except lam is required for ConcreteGates.
        '''
        # Ensure model has not yet been trained.
        assert not self.trained
        self.trained = True

        # Get additional arguments.
        mask_output = kwargs.get('mask_output', False)
        n_samples = kwargs.get('n_samples', None)
        lam = kwargs.get('lam', None)

        # Verify arguments.
        if mask_output:
            # Verify that model is based on mask or gates.
            assert (
                isinstance(self.model.input_layer, layers.ConcreteMask) or
                isinstance(self.model.input_layer, layers.ConcreteMax) or
                isinstance(self.model.input_layer, layers.ConcreteGates))

        if lam is not None:
            # Verify that model is based on gates.
            assert isinstance(self.model.input_layer, layers.ConcreteGates)
        else:
            # Verify that model is not based on gates.
            assert not isinstance(self.model.input_layer, layers.ConcreteGates)

        # Determine whether or not to require mask return.
        return_mask = lam or mask_output

        # Set up optimizer.
        optimizer = utils.get_optimizer(optimizer, self.model.parameters(), lr)

        # Set up data loaders.
        train_loader = DataLoader(train_dataset, batch_size=mbsize,
                                  shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=mbsize)

        # Determine device.
        device = next(self.model.parameters()).device

        # Set temperature and determine rate for decreasing.
        self.model.input_layer.temperature = start_temperature
        r = np.power(end_temperature / start_temperature,
                     1 / ((len(train_dataset) // mbsize) * max_nepochs))

        # For tracking loss.
        self.train_loss = []
        self.val_loss = []

        # Begin training.
        for epoch in range(max_nepochs):
            for x, y in train_loader:
                # Move to device.
                x = x.to(device)
                y = y.to(device)

                # Forward pass.
                if return_mask:
                    pred, m = self.model(x, n_samples=n_samples,
                                         return_mask=True)
                else:
                    pred = self.model(x, n_samples=n_samples)

                # Reshape to handle n_samples if necessary.
                if n_samples:
                    pred = pred.permute(1, 0, 2).reshape(n_samples * len(y), -1)
                    y = y.repeat(n_samples, 0)

                # Calculate loss.
                if mask_output:
                    loss = loss_fn(pred, y, weights=1-m)
                else:
                    loss = loss_fn(pred, y)

                # Calculate penalty if necessary.
                if lam:
                    penalty = lam * utils.input_layer_penalty(
                        self.model.input_layer, m)
                    loss = loss + penalty

                # Gradient step.
                loss.backward()
                optimizer.step()
                self.model.zero_grad()

                # Adjust temperature.
                self.model.input_layer.temperature *= r

            # Check progress.
            with torch.no_grad():
                # Calculate loss.
                self.model.eval()
                train_loss = utils.validate_input_layer(
                    self.model, train_loader, loss_fn).item()
                val_loss = utils.validate_input_layer(
                    self.model, val_loader, loss_fn).item()
                self.model.train()

                # Calculate penalty if necessary.
                if lam:
                    penalty = lam * utils.input_layer_penalty(
                        self.model.input_layer, m)
                    train_loss = train_loss + penalty
                    val_loss = val_loss + penalty

                # Record loss.
                self.train_loss.append(train_loss)
                self.val_loss.append(val_loss)

                if verbose and ((epoch + 1) % check_every == 0):
                    print('{}Epoch = {}{}'.format('-' * 8, epoch + 1, '-' * 8))
                    print('Train loss = {:.4f}'.format(train_loss))
                    print('Val loss = {:.4f}'.format(val_loss))
                    print(utils.input_layer_summary(
                        self.model.input_layer, n_samples=mbsize))

            # Check for early stopping.
            if utils.input_layer_converged(self.model.input_layer,
                                           n_samples=mbsize):
                if verbose:
                    print('Stopping early')
                break

            # Fix input layer if necessary.
            utils.input_layer_fix(self.model.input_layer)
