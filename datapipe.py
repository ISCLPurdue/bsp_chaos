import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Time-series style dataset that returns sliding windows over the time axis.

    Expects data shaped like (T, C, H, W, ...) and a window length `seq_len`.
    Each item is a tensor of shape (seq_len, C, H, W, ...) corresponding to
    data[t : t + seq_len].
    """

    def __init__(self, data: np.ndarray | torch.Tensor, seq_len: int):
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")

        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data.astype(np.float32))
        else:
            self.data = data.to(torch.float32)

        self.seq_len = seq_len
        self.num_samples = self.data.shape[0] - seq_len + 1
        if self.num_samples <= 0:
            raise ValueError("Not enough time steps for the requested seq_len")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(idx)
        return self.data[idx : idx + self.seq_len]
