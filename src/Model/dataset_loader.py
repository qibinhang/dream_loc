import torch
import torch.utils.data as data
import random


class DatasetLoader(data.Dataset):
    def __init__(self, dataset):
        super(DatasetLoader, self).__init__()
        random.seed(10)
        random.shuffle(dataset)
        self.dataset = torch.LongTensor(dataset)
        self.len = self.dataset.shape[0]

    def __getitem__(self, idx):
        report_idx = self.dataset[idx, 0]
        pos_code_idx = self.dataset[idx, 1]
        neg_code_idx = self.dataset[idx, 2]
        return report_idx, pos_code_idx, neg_code_idx

    def __len__(self):
        return self.len
