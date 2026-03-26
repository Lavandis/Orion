# PANORAMA 项目说明

## 项目简介

本项目用于单摆实验的时序建模与长时序预测。核心思路不是完全依赖黑盒神经网络，而是将：

- 物理先验模型
- 神经网络残差修正
- 数值积分求解

组合成一个混合动力学模型 `PANORAMA`，用来拟合真实实验数据中理想物理模型无法完全解释的部分。

当前代码面向以下任务：

- 从原始轨迹 CSV 中提取并预处理角度序列
- 基于角度序列构造状态 `[theta, omega]`
- 训练物理+神经的混合模型
- 对测试区间进行长时序外推预测
- 将纯物理基线与 PANORAMA 模型的效果进行对比

## 模型思路

项目把系统状态定义为：

- `theta`：摆角
- `omega`：角速度

整体动力学由两部分组成：

- `F_p`：物理白盒模型，对应单摆动力学和阻尼项
- `F_a`：神经网络残差项，用来补偿未建模因素

最终系统满足：

```text
F = F_p + F_a
```

其中：

- `F_p` 在 `src/models/physics.py` 中定义
- `F_a` 在 `src/models/augmentation.py` 中定义
- 混合后的前向推演在 `src/models/panorama.py` 中实现
- 时间积分使用 `RK4`，位于 `src/utils/integrators.py`

## 项目结构

```text
panorama_project/
├── assets/
│   ├── models/                  # 训练后保存的模型权重
│   └── plots/                   # 评估输出图像
├── configs/
│   └── train_config.yaml        # 全局配置文件
├── data/
│   ├── raw/                     # 原始 CSV 数据
│   └── processed/               # 预处理后的数据
├── scripts/
│   ├── 01_preprocess.py         # 数据预处理
│   ├── 02_tune_optuna.py        # Optuna 超参数搜索
│   ├── 03_train.py              # 模型训练
│   └── 04_evaluate.py           # 模型评估与可视化
├── src/
│   ├── dataset.py               # 数据集封装
│   ├── trainer.py               # 训练循环
│   ├── models/
│   │   ├── augmentation.py      # 神经残差项 F_a
│   │   ├── panorama.py          # PANORAMA 主模型
│   │   └── physics.py           # 物理项 F_p
│   └── utils/
│       ├── integrators.py       # 数值积分器
│       └── metrics.py           # MSE / RMSE 指标
├── .gitignore
├── README.md
└── requirements.txt
```

## 数据流说明

### 1. 原始数据

预处理脚本默认从 `data/raw/` 读取 CSV。每个 CSV 至少需要包含以下列：

- `Frame`
- `Time_Sec`
- `Center_X`
- `Center_Y`

其中 `Center_X` 和 `Center_Y` 表示摆球中心在图像中的像素坐标。

### 2. 预处理结果

`scripts/01_preprocess.py` 会：

1. 读取原始轨迹点
2. 结合配置中的悬点坐标 `pivot_x`、`pivot_y`
3. 将像素坐标转换为摆角 `theta_rad`
4. 按配置决定是否做 Savitzky-Golay 平滑
5. 输出到 `data/processed/`

输出文件至少包含：

- `Frame`
- `Time_Sec`
- `theta_rad`

### 3. 训练时的状态构造

在 `src/dataset.py` 中：

- `theta` 直接来自 `theta_rad`
- `omega` 通过对 `theta` 按时间步长 `dt` 求梯度得到

因此模型实际学习的是连续状态序列：

```text
[theta, omega]
```

## 配置文件说明

主配置文件为 `configs/train_config.yaml`，主要包含以下部分：

- `system`
  - `fps`：视频采样帧率，用于计算 `dt = 1 / fps`
  - `device`：训练设备，如 `cuda` 或 `cpu`
  - `seed`：随机种子
- `camera`
  - `pivot_x`、`pivot_y`：摆的悬点像素坐标
  - `use_filter`：是否启用平滑
  - `smooth_window`、`smooth_poly`：平滑参数
- `data`
  - `active_dataset`：当前训练/评估使用的数据文件
  - `train_split`：训练集截断位置
  - `test_start`：测试起点
  - `pred_len`：评估时向前预测的长度
- `physics`
  - `g`、`L`、`m`、`k1`、`k2`：物理参数
- `model`
  - `hidden_dim`：神经网络隐藏层宽度
  - `input_scale`：输入缩放系数
- `train`
  - `batch_size`、`epochs`、`lr`、`weight_decay`、`seq_len`
- `multiplier_method`
  - 与动态乘子法相关的超参数
- `paths`
  - `model_save`：模型权重保存路径
  - `plot_save`：评估图像保存路径

## 运行流程

建议按下面顺序执行。

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

如果 `requirements.txt` 尚未补全，当前代码至少依赖：

```bash
pip install torch pandas numpy matplotlib pyyaml scipy optuna
```

### 2. 准备原始数据

把原始 CSV 放入：

```text
data/raw/
```

然后检查 `configs/train_config.yaml` 中的悬点坐标和数据路径是否正确。

### 3. 执行预处理

```bash
python scripts/01_preprocess.py
```

执行后，处理结果会写入：

```text
data/processed/
```

### 4. 训练模型

先确认配置中的 `data.active_dataset` 指向正确的处理后数据文件，然后运行：

```bash
python scripts/03_train.py
```

训练完成后，模型权重默认保存到：

```text
assets/models/panorama_model.pth
```

### 5. 评估模型

```bash
python scripts/04_evaluate.py
```

评估脚本会：

- 从 `test_start` 位置取初始状态
- 分别生成纯物理基线预测和 PANORAMA 预测
- 计算 RMSE
- 输出对比图

图像默认保存到：

```text
assets/plots/panorama_test_result.png
```

### 6. 可选：超参数搜索

```bash
python scripts/02_tune_optuna.py
```

该脚本会搜索：

- `lr`
- `weight_decay`
- `hidden_dim`
- `seq_len`
- `batch_size`

搜索完成后，需要手动把最佳参数写回 `configs/train_config.yaml`。

## 训练与评估逻辑说明

### 训练

`scripts/03_train.py` 负责：

- 读取配置
- 加载处理后数据
- 构建 `DataLoader`
- 初始化 `PANORAMA`
- 调用 `src/trainer.py` 完成训练

当前训练损失以轨迹拟合误差为主，代码中保留了与动态乘子法相关的结构，便于后续继续加入物理约束项。

### 评估

`scripts/04_evaluate.py` 会对比两条曲线：

- 纯物理模型预测
- 训练后的 PANORAMA 预测

最终用 `RMSE` 衡量两者相对真实数据的误差。

## 输入输出约定

### 输入

- 原始输入：摆球中心像素轨迹
- 训练输入：初始状态 `[theta, omega]`
- 训练目标：后续 `seq_len` 步的真实轨迹

### 输出

- 训练输出：模型参数文件 `.pth`
- 评估输出：RMSE 指标与预测对比图

## 常见问题

### 1. 运行训练时报找不到数据文件

先确认：

- 是否已经执行过 `python scripts/01_preprocess.py`
- `configs/train_config.yaml` 中的 `data.active_dataset` 是否存在

### 2. 评估时报找不到模型文件

先确认：

- 是否已经执行过 `python scripts/03_train.py`
- `paths.model_save` 指向的位置是否正确

### 3. 训练使用 CPU 还是 GPU

由 `configs/train_config.yaml` 中的 `system.device` 控制：

```yaml
device: "cuda"
```

如果机器没有可用 GPU，请改成：

```yaml
device: "cpu"
```

## 后续建议

如果你准备继续完善这个项目，优先建议补这几件事：

- 补全 `requirements.txt`
- 在 README 中附一份真实样例 CSV
- 在训练阶段加入验证集评估与最佳模型保存
- 明确记录当前损失函数与动态乘子法的实际启用状态

## 适用场景

这个项目适合用于：

- 物理实验竞赛中的 AI 建模展示
- 单摆实验数据的拟合与预测
- 物理先验与神经网络融合建模的教学或演示
