# src/models/physics.py
import torch
import torch.nn as nn

class PhysicsModel(nn.Module):
    """
    白盒算子 (F_p)：系统的基础规则引擎。
    根据已知的公式，输出系统状态的理论变化率。
    """
    def __init__(self, g_L: float, k1: float, k2: float):
        super().__init__()
        # 使用 register_buffer 将常数注册为模型的内部状态
        # 作用：它们不会被优化器当作权重去盲目更新，但能随模型一起保存和加载，并自动转移到 GPU
        self.register_buffer('g_L', torch.tensor([g_L], dtype=torch.float32))
        self.register_buffer('k1', torch.tensor([k1], dtype=torch.float32))
        self.register_buffer('k2', torch.tensor([k2], dtype=torch.float32))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        计算理论上的系统状态变化率。

        Args:
            state (torch.Tensor): 当前状态 [theta, omega]，Shape 必须为 (Batch, 2)

        Returns:
            torch.Tensor: 理论推导出的导数 [d_theta, d_omega]，Shape 为 (Batch, 2)
        """
        # 利用切片提取角位移 (theta) 和角速度 (omega)，保持二维矩阵的形状 (Batch, 1)
        theta = state[:, 0:1]
        omega = state[:, 1:2]
        
        # 规则 1：角位移的变化率，在逻辑上天然等于当前的角速度
        d_theta = omega
        
        # 规则 2：角速度的变化率，由预设的公式决定
        d_omega = -self.g_L * torch.sin(theta) \
                  - self.k1 * omega \
                  - self.k2 * omega * torch.abs(omega)
        
        # 将算出的两个导数拼装回 (Batch, 2) 的格式吐出去
        return torch.cat([d_theta, d_omega], dim=1)