# PANORAMA 项目说明

## 项目简介

本项目用于单摆实验的时序建模与长时预测。核心思路不是完全依赖黑盒神经网络，而是将：

- 物理先验模型
- 神经网络残差修正
- 数值积分求解

组合成一个混合动力学模型 `PANORAMA`，用于拟合理想物理模型无法完全解释的实验误差，并在较长时间范围内进行外推预测。

当前代码的主要功能包括：

- 从原始角度 CSV 构造标准化训练数据
- 基于角度序列构造状态 `[theta, omega]`
- 训练物理 + 神经残差的混合模型
- 对测试段做长时滚动预测
- 使用 Optuna 进行粗搜索与局部微调

---

## 模型结构

系统状态定义为：

- `theta`：摆角
- `omega`：角速度

整体动力学由两部分组成：

- `F_p`：物理白盒模型，对应单摆动力学与阻尼项
- `F_a`：神经网络残差项，用于修正未建模因素

最终形式：

```text
F = F_p + F_a
```

对应代码位置：

- 物理项：`src/models/physics.py`
- 残差项：`src/models/augmentation.py`
- 混合模型：`src/models/panorama.py`
- 数值积分：`src/utils/integrators.py`

模型在内部按配置中的原始 `fps` 进行时间推进，评估时输出仍然是原始帧率下的预测序列。

---

## 当前数据格式

### 推荐原始输入格式

当前推荐的原始 CSV 直接提供时间和摆角：

```csv
Time,Angle_rad
0,-0.263427688
0.0111,-0.263427688
0.0222,-0.26073796
...
```

支持的原始列名：

- 时间列：`Time` 或 `Time_Sec`
- 角度列：`Angle_rad` 或 `theta_rad`

### 预处理后输出格式

`scripts/01_preprocess.py` 会将原始数据统一转换为：

- `Frame`
- `Time_Sec`
- `theta_rad`
- `omega_rad_s`

其中：

- `theta_rad` 是角度序列
- `omega_rad_s` 由角度序列对时间求导得到

如果原始数据中没有 `Frame` 列，会自动按 `0, 1, 2, ...` 生成。

---

## 当前训练与评估逻辑

### 数据切分

当前默认按比例切分：

- 前 `75%` 作为训练段
- 后 `25%` 作为测试段

对于 4 分钟数据，等价于：

- 前 3 分钟训练
- 后 1 分钟测试

对应配置：

```yaml
data:
  train_ratio: 0.75
```

### 稀疏监督训练

当前训练逻辑支持：

- 模型仍按原始 `fps` 做积分与 rollout
- 训练时可以不对每一帧都计算 loss
- 通过 `train.supervision_downsample_to_fps` 控制监督频率

例如：

```yaml
train:
  supervision_downsample_to_fps: 50
```

表示：

- 模型仍按原始 `fps` 推演
- 但 loss 只在约 `50fps` 的监督点上计算

如果设置为：

```yaml
train:
  supervision_downsample_to_fps: null
```

则表示全帧监督。

### 评估逻辑

当前评估脚本 `scripts/04_evaluate.py`：

- 从测试段起点取初始状态
- 以原始 `fps` 对整个测试段做滚动预测
- 分别生成纯物理基线与 PANORAMA 预测
- 计算 RMSE
- 输出对比图

---

## 项目结构

```text
panorama_project/
├─ artifacts/                  # Optuna 微调等实验历史
├─ assets/
│  ├─ models/                  # 训练后的模型权重
│  └─ plots/                   # 评估输出图像
├─ configs/
│  └─ train_config.yaml        # 全局配置
├─ data/
│  ├─ raw/                     # 原始 CSV
│  └─ processed/               # 预处理后 CSV
├─ scripts/
│  ├─ 01_preprocess.py         # 数据预处理
│  ├─ 02_tune_optuna.py        # 粗搜索 Optuna
│  ├─ 03_train.py              # 正式训练
│  ├─ 04_evaluate.py           # 评估与可视化
│  └─ 05_finetune_optuna.py    # 局部微调 Optuna
├─ src/
│  ├─ dataset.py               # 数据集封装
│  ├─ trainer.py               # 训练循环
│  ├─ models/
│  │  ├─ augmentation.py
│  │  ├─ panorama.py
│  │  └─ physics.py
│  └─ utils/
│     ├─ curriculum.py
│     ├─ data.py
│     ├─ integrators.py
│     └─ metrics.py
├─ .gitignore
├─ README.md
└─ requirements.txt
```

---

## 配置文件说明

主配置文件为 `configs/train_config.yaml`。

### `system`

- `fps`：原始数据对应的帧率
- `device`：训练设备，如 `cuda` 或 `cpu`
- `seed`：随机种子

### `preprocess`

- `use_filter`：是否对角度序列做平滑
- `smooth_window`：Savitzky-Golay 平滑窗口
- `smooth_poly`：Savitzky-Golay 多项式阶数

### `data`

- `raw_path`：原始 CSV 目录
- `processed_path`：预处理输出目录
- `active_dataset`：当前训练/评估使用的数据文件
- `train_ratio`：训练段比例

### `physics`

- `g`
- `L`
- `m`
- `k1`
- `k2`

这些是白盒物理模型中的参数。

### `model`

- `hidden_dim`
- `input_scale`
- `residual_scale`
- `output_init_std`

### `train`

- `batch_size`
- `epochs`
- `lr`
- `weight_decay`
- `seq_len_seconds`
- `supervision_downsample_to_fps`
- `theta_loss_weight`
- `omega_loss_weight`
- `fa_reg_weight`

### `train.curriculum`

当前代码仍保留 curriculum 配置能力，但默认可以关闭：

```yaml
train:
  curriculum:
    enabled: false
```

### `tune`

粗搜索 Optuna 配置：

- `search_epochs`
- `validation_ratio`
- `validation_rollout_seconds`
- `stage1.n_trials`
- `stage1.seq_len_seconds_candidates`
- `stage2.n_trials`
- `stage2.lr_range`
- `stage2.weight_decay_range`
- `stage2.hidden_dim_choices`
- `stage2.batch_size_choices`

### `fine_tune`

局部微调 Optuna 配置，用于围绕当前 best params 做小范围再搜索：

- `n_trials`
- `search_epochs`
- `validation_ratio`
- `validation_rollout_seconds`
- `validation_interval_epochs`
- `seq_len_seconds_multipliers`
- `lr_multiplier_range`
- `weight_decay_multiplier_range`
- `fa_reg_weight_multiplier_range`
- `hidden_dim_choices`
- `batch_size_choices`

### `paths`

- `model_save`
- `plot_save`

---

## 运行流程

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备原始数据

将原始 CSV 放入：

```text
data/raw/
```

### 3. 执行预处理

```bash
python scripts/01_preprocess.py
```

执行后会在：

```text
data/processed/
```

生成标准化后的数据文件。

### 4. 粗搜索超参数（可选）

```bash
python scripts/02_tune_optuna.py
```

这个脚本主要做两阶段粗搜索：

1. 先找较合适的 `seq_len_seconds`
2. 再在该窗口下搜索其他训练超参

### 5. 局部微调超参数（可选）

```bash
python scripts/05_finetune_optuna.py
```

这个脚本会围绕当前 `train_config.yaml` 中的已有超参，在其附近继续精细搜索。运行历史会保存在：

```text
artifacts/optuna_finetune/
```

每次运行会生成：

- `config_snapshot.yaml`
- `best_params.yaml`
- `summary.json`
- `trials.csv`

### 6. 正式训练

```bash
python scripts/03_train.py
```

训练完成后，模型默认保存到：

```text
assets/models/panorama_model.pth
```

### 7. 评估模型

```bash
python scripts/04_evaluate.py
```

评估结果图默认保存到：

```text
assets/plots/panorama_test_result.png
```

---

## 常见使用示例

### 全帧监督训练

```yaml
train:
  supervision_downsample_to_fps: null
```

### 稀疏监督训练

例如原始 `200fps` 数据，只在约 `50fps` 的点上计算 loss：

```yaml
train:
  supervision_downsample_to_fps: 50
```

### 只做一次轻量微调

```bash
python scripts/05_finetune_optuna.py --trials 10 --epochs 5 --validation-rollout-seconds 10
```

---

## 常见问题

### 1. 预处理时报找不到角度列

请确认原始 CSV 至少包含：

- `Time` 或 `Time_Sec`
- `Angle_rad` 或 `theta_rad`

### 2. 训练时报找不到数据文件

请确认：

- 已执行 `python scripts/01_preprocess.py`
- `configs/train_config.yaml` 中的 `data.active_dataset` 路径正确

### 3. 评估时报找不到模型文件

请确认：

- 已执行 `python scripts/03_train.py`
- `paths.model_save` 路径正确

### 4. 没有 GPU 怎么办

将配置改为：

```yaml
system:
  device: "cpu"
```

---

## 当前实验建议

如果你当前主要关注不同帧率数据上的表现，建议优先控制这几个变量：

- `data.active_dataset`
- `system.fps`
- `train.seq_len_seconds`
- `train.supervision_downsample_to_fps`
- `train.lr`
- `train.weight_decay`

对于高帧率数据，常见做法是：

- 仍使用原始高帧率数据做积分与预测
- 训练时适当使用稀疏监督，减轻优化难度

---

## 适用场景

本项目适合用于：

- 单摆实验数据的长时预测
- 物理先验 + 神经残差混合建模
- 实验竞赛中的 AI 建模展示
- 对不同采样帧率条件下的动力学建模效果做比较
