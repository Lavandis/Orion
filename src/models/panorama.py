# src/models/panorama.py
import torch
import torch.nn as nn
from .physics import PhysicsBox
from .augmentation import AugmentationNetwork
from src.utils.integrators import rk4_step

class PANORAMA(nn.Module):
    """
    PANORAMA 核心架构：统领物理白盒与神经黑盒，执行时域积分推演。
    """
    def __init__(self, dt: float, g: float, m: float, L: float, k1: float, k2: float, hidden_dim: int, input_scale: list):
        super().__init__()
        self.dt = dt
        
        # 1. 实例化纯物理白盒 (传入基础实验参量)
        self.physics_model = PhysicsBox(g, m, L, k1, k2)
        
        # 2. 实例化神经网络黑盒 (刚刚截图中丢失的这行代码补回来了！)
        self.augmentation = AugmentationNetwork(hidden_dim, input_scale)

    def dynamics(self, state: torch.Tensor) -> torch.Tensor:
        """
        计算某一瞬间的混合动力学导数: F = F_p + F_a
        """
        # 1. 获取物理公式推导出的基准变化率 (注意这里名字改成了 physics_model)
        f_p = self.physics_model(state)
        
        # 2. 获取神经网络算出的残差修正力
        aug = self.augmentation(state)
        
        # 3. 给角位移的导数强行补 0，拼成 (Batch, 2) 的维度
        zeros = torch.zeros_like(aug)
        f_a_vec = torch.cat([zeros, aug], dim=1)
        
        # 4. 叠加输出
        return f_p + f_a_vec

    def forward(self, initial_state: torch.Tensor, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向推演计算图
        
        Args:
            initial_state: 初始状态矩阵，Shape: (Batch, 2)
            steps: 需要向前推演的时间步数
            
        Returns:
            tuple:
                - 预测的轨迹，Shape: (Batch, steps, 2)
                - fa_penalty: 神经网络干预力度的平均值，用于乘子法惩罚
        """
        states = []
        fa_penalties = []
        
        curr_state = initial_state
        
        for _ in range(steps):
            # 记录当前时刻 AI 的“干预力度” (取绝对值的均值)
            aug_force = self.augmentation(curr_state)
            fa_penalties.append(torch.abs(aug_force**2).mean())
            
            # 调用底层工具库里的 RK4 积分器，推演到下一帧
            # 注意：我们将 self.dynamics 这个黑盒函数整体传给了 rk4_step
            curr_state = rk4_step(self.dynamics, curr_state, self.dt)
            states.append(curr_state)
            
        # 将列表堆叠成 Tensor 返回
        pred_traj = torch.stack(states, dim=1)
        fa_penalty_mean = torch.stack(fa_penalties).mean()
        
        return pred_traj, fa_penalty_mean