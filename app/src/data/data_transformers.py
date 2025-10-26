import torch
import pandas as pd
from torch.utils.data import Dataset


class TransformerDataset(Dataset):

    def __init__(
        self,
        dataset
        ):
        self.dataset = dataset
        self.max_length = 128
        self.tokenizer = None

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["text"]
        labels = item["labels"]

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )

        enc["labels"] = torch.tensor(labels, dtype=torch.float)
        return enc
