# 基于非线性摆的 PANORAMA 混合建模系统

本项目为第十二届全国大学生物理实验竞赛（创新）命题六“AI+物理实验”的算法核心代码。基于高内聚低耦合原则，采用残差解耦机制，将物理学先验算子与深度学习神经算子相结合。

## 📁 项目目录结构说明

panorama_project/
├── .git/                    # Git 版本控制目录 (自动生成)
├── .gitignore               # Git 忽略规则 (忽略数据、日志、模型权重)
├── README.md                # 项目说明文档
├── configs/                 # 配置文件目录 (剥离硬编码参数)
│   └── train_config.yaml    # 超参数、物理常数、文件路径配置
├── data/                    # 数据存放目录 (不提交到 Git)
│   ├── raw/                 # YOLO-Pose 提取的原始 CSV 数据
│   └── processed/           # 经过清洗、平滑处理后的标准化数据
├── src/                     # 核心源码包 (高度模块化)
│   ├── __init__.py
│   ├── dataset.py           # 数据集封装 (ODEDataset)
│   ├── models/              # 模型定义层
│   │   ├── __init__.py
│   │   ├── physics.py       # 基础物理算子层 (F_p)
│   │   ├── augmentation.py  # 神经增强算子层 (F_a)
│   │   └── panorama.py      # 整合物理与神经网络的 ODE 求解器包装类
│   ├── utils/               # 工具函数
│   │   ├── integrators.py   # 数值积分器 (如 RK4 的独立实现)
│   │   └── metrics.py       # 误差计算 (MSE, RMSE 等)
│   └── trainer.py           # 核心训练逻辑 (动态乘子法、Loss 计算、反向传播)
├── scripts/                 # 执行脚本 (调用 src 中的模块)
│   ├── 01_preprocess.py     # 数据预处理脚本
│   ├── 02_tune_optuna.py    # 超参数搜索脚本
│   ├── 03_train.py          # 主训练脚本
│   └── 04_evaluate.py       # 模型测试与可视化出图脚本
└── requirements.txt         # 依赖清单 (如 torch, pandas, optuna)


## 📁 项目目录结构说明

\`\`\`text
PANORAMA_PROJECT/
├── assets/                  # 训练产出与静态资源 (Git 忽略具体文件)
│   ├── models/              # 保存的 .pth 模型最佳权重
│   └── plots/               # 评估脚本生成的测试集对比曲线图表
├── configs/                 # 全局配置中心
│   └── train_config.yaml    # 集中管理所有超参数、环境与物理常数
├── data/                    # 数据集目录 (Git 忽略具体文件)
│   ├── processed/           # 预处理与平滑后的标准化格式数据
│   └── raw/                 # YOLO-Pose 提取的原始离散轨迹数据
├── scripts/                 # 项目执行入口脚本 (流水线控制)
│   ├── 01_preprocess.py     # 数据清洗、平滑处理与格式统一
│   ├── 02_tune_optuna.py    # Optuna 贝叶斯超参数自动化寻优
│   ├── 03_train.py          # 核心训练脚本 (动态拉格朗日乘子法)
│   └── 04_evaluate.py       # 模型载入、长时序推演与 RMSE 误差评估
├── src/                     # 核心算法与工具包 (高内聚模块)
│   ├── models/              # 网络与物理算子定义层
│   │   ├── __init__.py
│   │   ├── augmentation.py  # 神经黑盒算子层 (Fa, 输出残差扰动)
│   │   ├── panorama.py      # 整合物理与神经的前向 ODE 推演图
│   │   └── physics.py       # 物理白盒算子层 (Fp, 理想动力学)
│   ├── utils/               # 通用底层工具组件
│   │   ├── __init__.py
│   │   ├── integrators.py   # 数值积分器实现 (如高精度 RK4)
│   │   └── metrics.py       # 误差计算与性能评估指标库
│   ├── __init__.py
│   ├── dataset.py           # 时序序列滑动窗口切割与 DataLoader 封装
│   └── trainer.py           # 训练循环控制与自适应乘子惩罚升级逻辑
├── .gitignore               # Git 忽略规则配置
├── README.md                # 项目说明文档 (本文档)
└── requirements.txt         # Python 依赖环境清单
\`\`\`

## 🚀 快速开始
1. 安装依赖环境：`pip install -r requirements.txt`
2. 将 `data.csv` 放入 `data/raw/` 目录。
3. 根据需要修改 `configs/train_config.yaml` 中的参数。
4. 运行训练脚本：`python scripts/03_train.py`