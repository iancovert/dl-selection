import torch
import numpy as np
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    '''
    Dataset capable of using subset of inputs and outputs.

    Args:
      data: inputs (np.ndarray or torch.Tensor).
      targets: outputs (np.ndarray or torch.Tensor)
    '''
    def __init__(self,
                 data,
                 targets):
        self.input_size = data.shape[1]
        if isinstance(data, np.ndarray):
            # Conversions for numpy.
            self.data = data.astype(np.float32)
            if len(targets.shape) == 1:
                self.output_size = len(np.unique(targets))
                self.targets = targets.astype(np.long)
            else:
                self.output_size = targets.shape[1]
                self.targets = targets.astype(np.float32)
        elif isinstance(data, torch.Tensor):
            # Conversions for PyTorch.
            self.data = data.float()
            if len(targets.shape) == 1:
                self.output_size = len(torch.unique(targets))
                self.targets = targets.long()
            else:
                self.output_size = targets.shape[1]
                self.targets = targets.float()
        self.set_inds(None)
        self.set_output_inds(None)

    def set_inds(self, inds=None):
        '''Set input indices to be returned.'''
        data = self.data
        if inds is not None:
            inds = np.array([i in inds for i in range(self.input_size)])
            data = data[:, inds]
        self.input = data

    def set_output_inds(self, inds=None):
        '''Set output indices to be returned.'''
        output = self.targets
        if inds is not None:
            assert len(output.shape) == 2, 'only for multitask regression tasks'
            inds = np.array([i in inds for i in range(self.output_size)])
            output = output[:, inds]
        self.output = output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.input[index], self.output[index]
