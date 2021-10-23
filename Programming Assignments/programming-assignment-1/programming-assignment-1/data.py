# data.py

import torch
import pickle


class DataLoader:
    def __init__(self, batch_size, split):
        self.batch_size = batch_size

        with open('data.pk', 'rb') as f:
            data_obj = pickle.load(f, encoding='latin1')

        self.vocab = data_obj['vocab']

        if split == 'Train':
            self.inputs = torch.from_numpy(data_obj['train_inputs'])

        if split == 'Valid':
            self.inputs = torch.from_numpy(data_obj['valid_inputs'])

        if split == 'Test':
            self.inputs = torch.from_numpy(data_obj['test_inputs'])

        self.nsamples = self.inputs.size(0)
        self.context_len = self.inputs.size(1)
        self.indices = torch.randperm(self.nsamples)
        self.count = 0

    def get_size(self):
        return self.nsamples

    def get_batch(self):
        if self.count == self.nsamples:
            self.indices = torch.randperm(self.nsamples)
            self.count = 0

        max_index = min(self.batch_size + self.count, self.nsamples)
        data = self.inputs[self.indices[self.count:max_index]].long()
        self.count = max_index

        return data
    
    def sample_mask(self, batch_size):
        mask_idx = torch.randint(0, self.context_len, size=(batch_size,))
        mask = torch.zeros((batch_size, self.context_len), dtype=torch.int)
        mask[torch.arange(batch_size), mask_idx] = 1
        return mask