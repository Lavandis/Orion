# scripts/03_train.py
import os
import sys
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader

# 核心修复：将项目根目录加入系统路径，完美兼容 VSCode 的右上角运行按钮
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入我们一手打造的模块化组件
from src.models import PANORAMA
from src.dataset import ODEDataset
from src.trainer import train_panorama

def load_config(config_path="configs/train_config.yaml"):
    """加载全局 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    # 1. 读取总控配置
    config = load_config()
    device = torch.device(config['system']['device'])
    fps = config['system']['fps']
    dt = 1.0 / fps
    
    # 为了保证实验完全可复现，固定全局随机种子
    torch.manual_seed(config['system']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['system']['seed'])
        
    print(f"⚙️ 系统初始化完成 | 设备: {device} | 采样率: {fps} FPS")

    # 2. 数据流装配 (Data Pipeline)
    data_path = config['data']['active_dataset']
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ 找不到预处理数据文件: {data_path}\n请先运行 01_preprocess.py")
        
    print(f"📂 正在加载数据集: {data_path}")
    full_df = pd.read_csv(data_path)
    
    # 按照配置截断训练集 (例如前 16200 帧)
    train_split = config['data']['train_split']
    train_df = full_df.iloc[:train_split]
    
    # 实例化数据集与 DataLoader
    seq_len = config['train']['seq_len']
    batch_size = config['train']['batch_size']
    train_dataset = ODEDataset(train_df, seq_len=seq_len, dt=dt)
    
    # pin_memory=True 可以加速 CPU 到 GPU 的数据拷贝，是极佳的工程习惯
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 3. 核心引擎实例化 (Model Instantiation)
    print("🧠 正在构建 PANORAMA 混合推理引擎...")
    model = PANORAMA(
        dt=dt,
        g_L=config['physics']['g_L'],
        k1=config['physics']['k1'],
        k2=config['physics']['k2'],
        hidden_dim=config['model']['hidden_dim'],
        input_scale=config['model']['input_scale']
    ).to(device)
    
    # 开启 JIT 编译加速 (如果你的算子支持，这会让运算速度起飞)
    # model = torch.jit.script(model)
    
    # 4. 启动训练循环 (依赖注入，将控制权交给 trainer 模块)
    print("==================================================")
    trained_model = train_panorama(
        model=model,
        train_loader=train_loader,
        config=config,
        device=device
    )
    print("==================================================")
    print("🎉 训练主流程全部执行完毕！")

if __name__ == "__main__":
    main()