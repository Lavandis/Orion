import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils import get_theta_omega


class ODEDataset(Dataset):
    """Sliding-window trajectory dataset for ODE rollout training."""

    def __init__(self, df: pd.DataFrame, seq_len: int, dt: float):
        theta, omega = get_theta_omega(df, fallback_dt=dt)
        self.data = np.stack([theta, omega], axis=1).astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        init_state = self.data[idx]
        target_traj = self.data[idx + 1 : idx + 1 + self.seq_len]
        return torch.tensor(init_state), torch.tensor(target_traj)
