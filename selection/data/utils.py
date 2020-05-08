import torch
import numpy as np
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    '''Dataset capable of using subset of inputs and outputs.'''
    def __init__(self,
                 data,
                 labels):
        self.input_size = data.shape[1]
        if isinstance(data, np.ndarray):
            # Conversions for numpy.
            self.data = data.astype(np.float32)
            if len(labels.shape) == 1:
                self.output_size = len(np.unique(labels))
                self.labels = labels.astype(np.long)
            else:
                self.output_size = labels.shape[1]
                self.labels = labels.astype(np.float32)
        elif isinstance(data, torch.Tensor):
            # Conversions for PyTorch.
            self.data = data.float()
            if len(labels.shape) == 1:
                self.output_size = len(torch.unique(labels))
                self.labels = labels.long()
            else:
                self.output_size = labels.shape[1]
                self.labels = labels.float()
        self.set_inds(None)
        self.set_output_inds(None)

    def set_inds(self, inds=None):
        '''Set indices to be returned.'''
        data = self.data
        if inds is not None:
            inds = np.array([i in inds for i in range(self.input_size)])
            data = data[:, inds]
        self.input = data

    def set_output_inds(self, inds=None):
        '''Set indices to be returned.'''
        output = self.labels
        if inds is not None:
            assert len(output.shape) == 2
            inds = np.array([i in inds for i in range(self.output_size)])
            output = output[:, inds]
        self.output = output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.input[index], self.output[index]
