# src/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.utils.metrics import calculate_mse
import os

def train_panorama(model: nn.Module, 
                   train_loader: DataLoader, 
                   config: dict, 
                   device: torch.device):
    """
    核心训练调度器：执行前向推演、计算混合 Loss 并应用自适应乘子法更新网络。
    """
    # 1. 从配置字典中提取超参数
    epochs = config['train']['epochs']
    lr = config['train']['lr']
    weight_decay = config['train']['weight_decay']
    seq_len = config['train']['seq_len']
    
    # 乘子法核心调度参数
    lambda_val = config['multiplier_method']['lambda_init']
    tau = config['multiplier_method']['tau']
    warmup_epochs = config['multiplier_method']['warmup_epochs']
    
    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    print(f"🚀 开始训练: 共 {epochs} Epochs, Batch Size: {config['train']['batch_size']}")
    
    model.train()
    for epoch in range(epochs):
        total_mse = 0.0
        
        # 🔥 热身期策略：彻底关闭对 Fa 的惩罚，逼迫主干规则先发力拟合
        fa_weight = 0.0 if epoch < warmup_epochs else 1.0
        
        for init_state, target_traj in train_loader:
            init_state = init_state.to(device)
            target_traj = target_traj.to(device)
            
            # 清空上一轮的旧梯度
            optimizer.zero_grad()
            
            # 1. 前向推演 (呼叫 models 加工厂)
            pred_traj, fa_norm = model(init_state, seq_len)
            
            # 2. 计算误差 (切片提取第一维 theta 进行对比)
            pred_theta = pred_traj[:, :, 0]
            target_theta = target_traj[:, :, 0]
            mse_loss = calculate_mse(pred_theta, target_theta)
            
            # 3. 核心大招：动态拉格朗日 Loss 组合
            # 公式: L = ||Fa||^2 + lambda * MSE
            #loss = fa_weight * fa_norm + lambda_val * mse_loss
            loss = mse_loss
            # 4. 反向传播与参数更新
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪，防止爆炸
            optimizer.step()
            
            total_mse += mse_loss.item()
            
        avg_mse = total_mse / len(train_loader)
        
        # 5. 乘子法自适应升级：热身结束后，根据当前残留的 Bug 率 (MSE) 放大惩罚力度
        if epoch >= warmup_epochs:
            lambda_val += tau * avg_mse
            
        # 终端进度播报
        if (epoch + 1) % 5 == 0 or epoch == 0:
            status = "🔥 预热中(主干拟合)" if epoch < warmup_epochs else "⚙️ 乘子法(残差解耦)"
            print(f"Epoch {epoch+1:02d}/{epochs} [{status}] | MSE: {avg_mse:.6e} | Fa Norm: {fa_norm.item():.4f} | Lambda: {lambda_val:.1f}")

    # 训练结束，妥善保存模型资产
    save_path = config['paths']['model_save']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 剥离 JIT 外壳保存纯参数，防止跨环境加载报错
    torch.save(model.state_dict(), save_path)
    print(f"💾 模型参数已安全归档至: {save_path}")
    
    return model