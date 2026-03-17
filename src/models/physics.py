# src/models/physics.py
import torch
import torch.nn as nn

class PhysicsBox(nn.Module):
    """
    纯白盒物理动力学模型：严格遵循单摆非线性阻尼微分方程。
    """
    def __init__(self, g: float, m: float, L: float, k1: float, k2: float):
        super().__init__()
        # 将基础实验参数注册为不可训练的 buffer，存入显存
        self.register_buffer('g', torch.tensor(g, dtype=torch.float32))
        self.register_buffer('m', torch.tensor(m, dtype=torch.float32))
        self.register_buffer('L', torch.tensor(L, dtype=torch.float32))
        self.register_buffer('k1', torch.tensor(k1, dtype=torch.float32))
        self.register_buffer('k2', torch.tensor(k2, dtype=torch.float32))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        计算状态对时间的导数 [d_theta, d_omega]
        """
        theta = state[..., 0:1]
        omega = state[..., 1:2]

        # 严格对齐真实的角加速度公式
        alpha_phys = -(self.g / self.L) * torch.sin(theta) \
                     - (self.k1 / self.m) * omega \
                     - (self.k2 * self.L / self.m) * omega * torch.abs(omega)

        d_theta = omega
        d_omega = alpha_phys
        
        return torch.cat([d_theta, d_omega], dim=-1)