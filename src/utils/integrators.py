# src/utils/integrators.py
import torch
from typing import Callable

def rk4_step(dynamics_fn: Callable[[torch.Tensor], torch.Tensor], 
             state: torch.Tensor, 
             dt: float) -> torch.Tensor:
    """
    使用四阶龙格-库塔法 (RK4) 进行一步数值积分。
    
    Args:
        dynamics_fn: 计算状态导数的函数，即接收 state 返回变化率的黑盒
        state: 当前时刻的系统状态 Tensor，Shape: (Batch, 2)
        dt: 离散化的时间步长
        
    Returns:
        torch.Tensor: 经过 dt 时间推演后的下一个系统状态
    """
    # 计算四个方向的探测斜率
    k1 = dynamics_fn(state)
    k2 = dynamics_fn(state + 0.5 * dt * k1)
    k3 = dynamics_fn(state + 0.5 * dt * k2)
    k4 = dynamics_fn(state + dt * k3)
    
    # 按照 RK4 的经典权重 (1:2:2:1) 进行加权平均推演
    next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return next_state