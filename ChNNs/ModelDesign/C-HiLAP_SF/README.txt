project/
├── data/                      # 数据集存储目录
│   ├── librispeech/           # LibriSpeech数据集
│   │   ├── train_clean_100/   # 训练集
│   │   ├── dev_clean/         # 验证集
│   │   ├── test_clean/        # 测试集
│   ├── noise/                 # 用于鲁棒性测试的噪声
│   └── ravdess/               # 用于情感鲁棒性测试（可选）
├── src/                      # 源代码目录
│   ├── preprocessing/        # 数据预处理模块
│   │   ├── librispeech_parser.py  # 解析LibriSpeech数据集结构
│   │   ├── audio_processing.py    # 音频处理（重采样、分帧等）
│   │   └── augmentation.py        # 数据增强（噪声、混响等）
│   ├── features/             # 特征提取模块
│   │   ├── conventional.py        # 传统特征（MFCC、Fbank等）
│   │   └── chaotic.py             # 混沌特征（MLE、递归图等）
│   ├── models/               # 模型实现模块
	# 基线模型
│   │   ├── ecapa_tdnn.py       # ECAPA-TDNN实现（或其它）
│   │   └── xvector.py          # x-vector实现
 	# 混沌神经网络模型
│   │   ├── variant_a.py        # 混沌特征增强型网络
│   │   ├── variant_b.py        # 层级混沌注入型网络
│   │   └── c_hilap.py          # C-HiLAP完整实现
│   ├── utils/                # 工具函数
│   │   ├── metrics.py           # 评估指标（EER、minDCF等）
│   │   ├── visualization.py      # 可视化工具
│   │   ├── chaos_math.py         # 混沌数学工具函数
│   │   └── librispeech_utils.py  # LibriSpeech特定工具函数
│   └── losses/               # 损失函数实现
│       ├── classification.py    # 分类损失（交叉熵等）
│       └── chaotic_reg.py       # 混沌正则化损失
├── experiments/              # 实验配置与结果
│   ├── configs/              # 实验配置文件
│   │   ├── baseline.yaml         # 基线模型配置
│   │   ├── variant_a.yaml        # 混沌特征增强型配置
│   │   ├── variant_b.yaml        # 层级混沌注入型配置
│   │   └── c_hilap.yaml          # C-HiLAP配置
│   └── results/              # 实验结果存储
│       ├── baseline/          # 基线模型结果
│       ├── variant_a/         # 混沌特征增强型结果
│       ├── variant_b/         # 层级混沌注入型结果
│       └── c_hilap/           # C-HiLAP结果
├── scripts/                  # 执行脚本
│   ├── train.py               # 训练脚本
│   ├── evaluate.py            # 评估脚本
│   ├── preprocess.py          # 数据预处理脚本
│   └── visualize.py           # 结果可视化脚本
└── notebooks/                # Jupyter notebooks用于分析
    ├── data_exploration.ipynb   # 数据探索
    ├── feature_analysis.ipynb   # 特征分析
    └── results_analysis.ipynb   # 结果分析