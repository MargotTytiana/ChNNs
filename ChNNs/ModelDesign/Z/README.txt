speaker_recognition_chaotic_nn/
├── data_loader/              # 数据加载与预处理模块
│   ├── __init__.py
│   └── audio_processor.py    # 音频加载、分帧、VAD、归一化等
│
├── feature_extraction/       # 混沌特征提取模块
│   ├── __init__.py
│   ├── phase_space.py        # 相空间重构（延迟嵌入）
│   ├── mlsa.py               # 多尺度Lyapunov谱分析
│   └── rqa.py                # 递归量化分析（RR, DET, LAM）
│
├── chaotic_models/           # 混沌系统模型定义
│   ├── __init__.py
│   ├── lorenz.py             # Lorenz系统（核心混沌模型）
│   ├── rossler.py            # Rossler系统（备选混沌模型）
│   └── bifurcation_control.py # 分叉控制模块
│
├── neural_networks/          # 混沌神经网络核心组件
│   ├── __init__.py
│   ├── chaotic_embedding.py  # 混沌嵌入层（驱动混沌系统）
│   ├── attractor_pooling.py  # 奇怪吸引子池化（拓扑不变量提取）
│   └── speaker_embedding.py  # 说话人嵌入层（特征压缩）
│
├── training/                 # 模型训练模块
│   ├── __init__.py
│   └── trainer.py            # 训练循环、损失函数、优化器
│
├── evaluation/               # 模型评估与可视化
│   ├── __init__.py
│   ├── evaluator.py          # 准确率、混淆矩阵等评估指标
│   └── visualizer.py         # 特征可视化（相空间轨迹、递归图）
│
├── utils/                    # 工具类（数学、IO、日志等）
│   ├── __init__.py
│   ├── math_utils.py         # 数学工具（FNN算法、QR分解、拓扑不变量计算）
│   ├── io_utils.py           # 文件读写（音频、模型、配置）
│   └── logger.py             # 日志管理
│
└── configs/                  # 配置文件
    ├── default_config.yaml    # 默认全局配置（数据路径、超参数）
    └── model_configs.json     # 模型专属配置（混沌系统参数、嵌入维度）