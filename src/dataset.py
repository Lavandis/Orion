# src/dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ODEDataset(Dataset):
    """
    时序轨迹数据集：将长序列切割为滑动窗口格式，专为连续 ODE 推演设计。
    """
    def __init__(self, df: pd.DataFrame, seq_len: int, dt: float):
        # 兼容不同列名提取角位移 theta
        if 'theta_rad' in df.columns:
            theta = df['theta_rad'].values
        else:
            theta = df.iloc[:, -1].values
            
        # 自动计算离散导数 (角速度 omega)
        omega = np.gradient(theta, dt)
        
        # 组合成 (N, 2) 的特征矩阵 [theta, omega]
        self.data = np.stack([theta, omega], axis=1).astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        # 保证最后一个滑动窗口能取到完整的 seq_len 长度，防止越界
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        # 初始状态：当前帧的 [theta, omega]
        init_state = self.data[idx]
        # 目标轨迹：紧接着的 seq_len 帧的 [theta, omega]
        target_traj = self.data[idx + 1 : idx + 1 + self.seq_len]
        
        return torch.tensor(init_state), torch.tensor(target_traj)