from torch.utils.data import Dataset


class TupleInListDataset(Dataset):
    def __init__(self, base_pickle):
        self.dataset = base_pickle

    def __getitem__(self, item):
        current_item = self.dataset[item]

        return current_item[0], current_item[1]

    def __len__(self):
        return len(self.dataset)
