from torch.utils.data import DataLoader, Dataset as TorchDataset
from typing import NamedTuple, Dict, List
from collections import defaultdict
import torch
import numpy as np
import pandas as pd
from torch import nn


class Dataset(TorchDataset):
    def __init__(self, device: str = 'cpu',
                 dtypes_mapping: Dict[str, np.dtype] = {},
                 **datasets):
        self.datasets = datasets
        self.dtypes_mapping = dtypes_mapping
        self.device = device
        self._lens = list(map(len, self.datasets.values()))
        assert len(np.unique(self._lens)) == 1

    def __len__(self):
        return self._lens[0]

    def _to_torch(self, name: str, dataset: np.ndarray):
        if name in self.dtypes_mapping:
            dataset = dataset.astype(self.dtypes_mapping[name])
        return torch.from_numpy(np.asarray(dataset)).to(device=self.device)

    def __getitem__(self, index):
        return {name: self._to_torch(name, data[index])
                for name, data in self.datasets.items()}

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame,
                       device: str = 'cpu',
                       dtypes_mapping: Dict[str, np.dtype] = None,
                       **split_features):
        datasets = {features_name: data.loc[:, features].to_numpy()
                    for features_name, features in split_features.items()
                    if features is not None}
        return cls(device=device, dtypes_mapping=dtypes_mapping,
                   **datasets)