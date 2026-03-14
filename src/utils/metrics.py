# src/utils/metrics.py
import torch
import numpy as np

def calculate_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    计算均方误差 (MSE)。
    主要用于训练阶段，保持梯度追踪 (requires_grad=True)。
    
    Args:
        pred: 预测的轨迹 Tensor
        target: 真实的轨迹 Tensor
        
    Returns:
        torch.Tensor: 标量 Loss 值
    """
    return torch.nn.functional.mse_loss(pred, target)

def calculate_rmse_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    计算均方根误差 (RMSE)。
    主要用于评估阶段，剥离计算图后使用 NumPy 快速运算。
    
    Args:
        pred: 预测轨迹的 NumPy 数组
        target: 真实轨迹的 NumPy 数组
        
    Returns:
        float: RMSE 误差值
    """
    mse = np.mean((pred - target) ** 2)
    return np.sqrt(mse).item()