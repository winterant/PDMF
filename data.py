import pickle
from torch.utils.data import Dataset


class MultiViewDataset(Dataset):
    def __init__(self, data_path='dataset/handwritten_6views_train.pkl'):
        super().__init__()
        self.x, self.y = pickle.load(open(data_path, 'rb'))

    def __getitem__(self, index):
        x = dict()
        for v in self.x.keys():
            x[v] = self.x[v][index]
        return {
            'x': x,
            'y': self.y[index],
            'index': index
        }

    def __len__(self):
        return len(self.y)
