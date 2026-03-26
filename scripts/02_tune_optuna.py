# scripts/02_tune_optuna.py
import optuna
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 从我们写好的 src 模块中导包
from src.models import PANORAMA
from src.dataset import ODEDataset
from src.utils import calculate_mse

def load_config(config_path="configs/train_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def objective(trial, config):
    # 1. 定义 Optuna 需要搜索的超参数空间
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    seq_len = trial.suggest_int("seq_len", 30, 90, step=10) # 寻找最佳视野
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    
    # 提取固定参数
    device = torch.device(config['system']['device'])
    fps = config['system']['fps']
    dt = 1.0 / fps
    g = config['physics']['g']
    m = config['physics']['m']
    L = config['physics']['L']
    k1 = config['physics']['k1']
    k2 = config['physics']['k2']
    input_scale = config['model']['input_scale']
    
    # 搜索阶段为了节约时间，Epoch 不用跑满，看前 20 轮的收敛趋势即可
    search_epochs = 20 
    
    # 2. 准备数据
    data_path = config['data']['active_dataset']
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到数据文件: {data_path}")
        
    full_df = pd.read_csv(data_path)
    
    # Optuna 内部数据切分 (假设总训练集是 16200 帧)
    # 我们用前 13000 帧训练，用 13000-16200 帧来验证这组超参数的好坏
    train_df = full_df.iloc[:13000]
    val_df = full_df.iloc[13000:16200]
    
    train_loader = DataLoader(ODEDataset(train_df, seq_len, dt), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ODEDataset(val_df, seq_len, dt), batch_size=batch_size, shuffle=False)
    
    # 3. 初始化模型引擎
    model = PANORAMA(
        dt=dt, g=g, m=m, L=L, k1=k1, k2=k2, 
        hidden_dim=hidden_dim, input_scale=input_scale
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 4. 简化的训练循环 (关闭复杂的乘子法，只看基础拟合潜力)
    for epoch in range(search_epochs):
        model.train()
        for init_state, target_traj in train_loader:
            init_state, target_traj = init_state.to(device), target_traj.to(device)
            
            optimizer.zero_grad()
            # Optuna 阶段我们不关心惩罚项，只看轨迹拟合得好不好
            pred_traj, _ = model(init_state, seq_len)
            
            loss = calculate_mse(pred_traj[:, :, 0], target_traj[:, :, 0])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for init_state, target_traj in val_loader:
                init_state, target_traj = init_state.to(device), target_traj.to(device)
                pred_traj, _ = model(init_state, seq_len)
                val_loss += calculate_mse(pred_traj[:, :, 0], target_traj[:, :, 0]).item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        # 报告当前 Epoch 的表现，用于提前剪枝 (Pruning)
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return avg_val_loss

def main():
    config = load_config()
    print(f"🚀 启动 Optuna 寻优...\n📊 使用数据集: {config['data']['active_dataset']}")
    
    # 创建一个 Study，目标是让验证集的 Loss 最小化
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    
    # 运行 30 次尝试 (Trial)
    study.optimize(lambda trial: objective(trial, config), n_trials=30)
    
    print("\n🏆 搜索完成！")
    print(f"最佳 Loss: {study.best_value:.8e}")
    print("最佳参数组合 (请将这些值手动更新到 configs/train_config.yaml 中):")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()