# src/models/augmentation.py
import torch
import torch.nn as nn

class AugmentationNetwork(nn.Module):
    """
    神经黑盒算子 (F_a)：专门用于拟合未知的环境残差（如非线性流体力学扰动）。
    """
    def __init__(self, hidden_dim: int, input_scale: list = [10.0, 1.0]):
        super().__init__()
        
        # 将输入特征放大器注册为 buffer
        # 作用：它不参与梯度更新，但能随模型自动 .to(device) 转移到 GPU，并且完美兼容 JIT 编译
        self.register_buffer('input_scale', torch.tensor(input_scale, dtype=torch.float32))
        
        # 主干网络：两层 MLP，使用 Tanh 激活函数
        # 因为动力学系统的残差通常是平滑连续的，Tanh 的平滑特性比 ReLU 的折线更适合模拟现实物理
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # 输出层：严格限制只输出 1 维残差
        # 因为在单摆系统中，只有角速度(omega)的导数会受到未知阻力影响，角位移(theta)的导数永远是确定的 omega
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # 🔥 核心防御机制：零初始化
        # 强制让网络在未经训练时输出绝对为 0。保证系统在第一轮迭代时，完全等价于一个纯粹的白盒模型。
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        计算神经残差力
        
        Args:
            state (torch.Tensor): 系统当前状态 [theta, omega]，Shape: (Batch, 2)
            
        Returns:
            torch.Tensor: 对角速度的加速度修正量，Shape: (Batch, 1)
        """
        # 放大输入特征，防止由于输入数值过小，导致在 Tanh 激活函数处于线性区，从而损失非线性表达能力
        scaled_state = state * self.input_scale
        x = self.net(scaled_state)
        return self.output_layer(x)*0.001