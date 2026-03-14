# scripts/01_preprocess.py
import os
import glob
import yaml
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

def load_config(config_path="configs/train_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def process_data():
    config = load_config()
    
    # 1. 获取路径与相机参数
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    pivot_x = config['camera']['pivot_x']
    pivot_y = config['camera']['pivot_y']
    window = config['camera']['smooth_window']
    poly = config['camera']['smooth_poly']
    
    # 2. 批量扫描所有 CSV 文件
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
        print(f"❌ 在 {raw_dir} 中没有找到任何 CSV 文件！")
        return
        
    print(f"🔍 找到 {len(csv_files)} 个原始数据文件，开始批量处理...")
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"  -> 正在处理: {filename}")
        
        # 读取原始数据
        df = pd.read_csv(file_path)
        
        # 确保包含我们需要的列
        if 'Center_X' not in df.columns or 'Center_Y' not in df.columns:
            print(f"     ⚠️ 跳过 {filename}: 缺少 Center_X 或 Center_Y 列")
            continue
            
        # 3. 像素坐标 -> 弧度 (Radian) 转换
        # 计算相对悬挂点的偏移
        dx = df['Center_X'].values - pivot_x
        dy = df['Center_Y'].values - pivot_y
        
        # 使用 arctan2 计算角度 (向下为 0 度，向右为正，向左为负)
        theta_raw = np.arctan2(dx, dy)
        
        # 4. 数据平滑 (极其重要)
        # YOLO-Pose 在捕捉时不可避免会有像素级的抖动，直接求导会导致角速度出现巨大噪音
        # 使用 Savitzky-Golay 滤波器在保留曲线形状的同时消除高频噪音
        # 4. 数据平滑开关
        if config['camera']['use_filter'] and len(theta_raw) > window:
            theta_smoothed = savgol_filter(theta_raw, window_length=window, polyorder=poly)
        else:
            theta_smoothed = theta_raw
            
        # 5. 重组并保存为新的 DataFrame
        # 我们保留 Frame 和 Time_Sec，并加入计算好的 theta_rad
        processed_df = pd.DataFrame({
            'Frame': df['Frame'],
            'Time_Sec': df['Time_Sec'],
            'theta_rad': theta_smoothed
        })
        
        # 导出到 processed 文件夹
        save_path = os.path.join(processed_dir, filename)
        processed_df.to_csv(save_path, index=False)
        
    print(f"✅ 批量预处理完成！所有清洗后的数据已保存至 {processed_dir}")

if __name__ == "__main__":
    process_data()