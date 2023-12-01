import torch as th
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels) -> None:
        super().__init__()
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> dict:
        item = {key: th.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = th.tensor(self.labels[idx])
        
        return item