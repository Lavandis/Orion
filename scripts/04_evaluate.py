# scripts/04_evaluate.py
import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 核心修复：将项目根目录加入系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import PANORAMA
from src.utils import calculate_rmse_numpy

def load_config(config_path="configs/train_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    # ================= 1. 初始化与配置读取 =================
    config = load_config()
    device = torch.device(config['system']['device'])
    fps = config['system']['fps']
    dt = 1.0 / fps
    
    print("📈 开始执行 PANORAMA 模型评估与可视化程序...")
    
    # ================= 2. 准备测试数据 =================
    data_path = config['data']['active_dataset']
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ 找不到数据文件: {data_path}")
        
    df = pd.read_csv(data_path)
    data = df['theta_rad'].values.astype(np.float32)
    
    # 从配置中读取测试集的起点和预测长度
    start_idx = config['data']['test_start']
    pred_len = config['data']['pred_len']
    
    # 安全检查：防止预测长度超出真实数据边界
    max_len = len(data) - (start_idx + 1)
    actual_len = min(pred_len, max_len)
    
    print(f"📊 测试集起点: 第 {start_idx} 帧")
    print(f"📊 计划预测步数: {pred_len}, 实际可用 Ground Truth 步数: {actual_len}")

    # 计算测试集起点的初始状态 [theta_0, omega_0]
    omega_array = np.gradient(data, dt)
    
    theta_0 = data[start_idx]
    omega_0 = omega_array[start_idx]
    
    init_state = torch.tensor([[theta_0, omega_0]], dtype=torch.float32).to(device)
    true_future = data[start_idx + 1 : start_idx + 1 + actual_len]
    
    # ================= 3. 实例化模型引擎 =================
    model = PANORAMA(
        dt=dt,
        g=config['physics']['g'], 
        m=config['physics']['m'], 
        L=config['physics']['L'],  
        k1=config['physics']['k1'],
        k2=config['physics']['k2'],
        hidden_dim=config['model']['hidden_dim'],
        input_scale=config['model']['input_scale']
    ).to(device)
    
    # ================= 4. 实验 A: 纯物理基线 (零神经干预) =================
    print("🧪 正在推演基准线：纯物理模型 (不加载神经网络权重)...")
    # 因为我们在 augmentation.py 里做了极其严密的“零初始化”
    # 所以未经训练的模型，天然就是一个纯粹的白盒物理模型！
    pure_physics_model = model 
    pure_physics_model.eval()
    
    with torch.no_grad():
        phys_pred_traj, _ = pure_physics_model(init_state, actual_len)
        # 提取角位移 theta 序列，并转回 CPU 和 NumPy
        phys_pred = phys_pred_traj.cpu().numpy().squeeze()[:, 0]
        
    # ================= 5. 实验 B: PANORAMA 混合推演 =================
    print("🚀 正在推演 PANORAMA 混合动力学...")
    model_path = config['paths']['model_save']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 找不到训练好的模型权重: {model_path}\n请先运行 03_train.py")
        
    # 加载我们训练好的最优权重
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    with torch.no_grad():
        pano_pred_traj, _ = model(init_state, actual_len)
        pano_pred = pano_pred_traj.cpu().numpy().squeeze()[:, 0]
        
    # ================= 6. 误差量化 (Metrics) =================
    rmse_phys = calculate_rmse_numpy(phys_pred, true_future)
    rmse_pano = calculate_rmse_numpy(pano_pred, true_future)
    
    improvement = (rmse_phys - rmse_pano) / rmse_phys * 100
    
    print(f"\n  评估结果 (RMSE 指标):")
    print(f"    纯物理理论预测 | RMSE: {rmse_phys:.6f} rad")
    print(f"    PANORAMA 预测 | RMSE: {rmse_pano:.6f} rad")
    print(f"    算法性能提升率 | +{improvement:.2f}%")
    
    # ================= 7. 数据可视化与出图 =================
    print("🎨 正在生成对比曲线图...")
    time_axis = np.arange(actual_len) * dt
    
    # 使用比较学术和专业的配色方案
    plt.figure(figsize=(14, 6), dpi=150)
    
    plt.plot(time_axis, true_future, color='#2ca02c', linestyle='-', linewidth=2, alpha=0.7, label='Ground Truth (Real Data)')
    plt.plot(time_axis, phys_pred, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Physics Base (RMSE: {rmse_phys:.4f})')
    plt.plot(time_axis, pano_pred, color='#d62728', linestyle='-', linewidth=1.5, label=f'PANORAMA (RMSE: {rmse_pano:.4f})')
    
    plt.title(f"PANORAMA Long-Horizon Prediction vs Ground Truth\nPerformance Improvement: {improvement:.2f}%", fontsize=14, fontweight='bold')
    plt.xlabel("Time (Seconds)", fontsize=12)
    plt.ylabel("Angle (Radian)", fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 确保存放图片的文件夹存在
    plot_save_path = config['paths']['plot_save']
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    
    plt.savefig(plot_save_path, bbox_inches='tight')
    print(f"✅ 高清对比图已成功保存至: {os.path.abspath(plot_save_path)}")
    
    # 如果是在有图形界面的系统里，可以取消注释下面这行直接弹窗看图
    plt.show()

if __name__ == "__main__":
    main()