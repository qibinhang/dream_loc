import torch
import torch.utils.data as data
import random


class DatasetLoader(data.Dataset):
    def __init__(self, dataset):
        super(DatasetLoader, self).__init__()
        dataset_reformat = []
        for item in dataset:
            dataset_reformat += item
        random.seed(10)
        random.shuffle(dataset_reformat)
        self.dataset = torch.FloatTensor(dataset_reformat)
        self.len = self.dataset.shape[0]

    def __getitem__(self, idx):
        report_idx = self.dataset[idx, 0]
        code_idx = self.dataset[idx, 1]
        recency_value = self.dataset[idx, 2]
        frequency_value = self.dataset[idx, 3]
        label = self.dataset[idx, 4]
        # return report_idx.unsqueeze(-1), code_idx.unsqueeze(-1), recency_value.unsqueeze(-1), frequency_value.unsqueeze(-1), label.unsqueeze(-1)
        return report_idx.long(), code_idx.long(), recency_value, frequency_value, label.long()

    def __len__(self):
        return self.len
