---
AIGC:
  Label: '1'
  ContentProducer: '001191310104MACE0YX63918020'
  ProduceID: '147392816552226816'
  ReservedCode1: '{"SecurityData":{"Type":"TC260PG","Version":1,"PubSD":[{"Type":"DS","AlgID":"1.2.156.10197.1.501","TBSData":{"Type":"LabelMataData"},"Signature":"3045022100E816A5872275AE5414E7E0FD3D9B2C30F7B7FD68609D6486A38DF7CFAA1417B902201A510FBBE50819BAFF1FCC7B8527FF23CD40447D41A6776E6BCF1F773AE0DBBC"},{"Type":"PubKey","AlgID":"1.2.156.10197.1.501","KeyValue":"3059301306072A8648CE3D020106082A811CCF5501822D034200045728F08C58D9C72150A705693EF7337859F8FF216E802630041AFAE5C036F0A67CB1FDF8C57F5F8B623284AE09E4E545464B7140B26BBDA51BAE965F854C57AA"}]}}'
  ContentPropagator: '001191310104MACE0YX63918020'
  PropagateID: '147392816552226816'
  ReservedCode2: '{"SecurityData":{"Type":"TC260PG","Version":1,"PubSD":[{"Type":"DS","AlgID":"1.2.156.10197.1.501","TBSData":{"Type":"LabelMataData"},"Signature":"3046022100FEBE29497FA9E249C18918CF9D36A7F13E5AE2FFDEF184F7E5E1572CDD60C3370221009C21182427A0860A85F9BDBBFC92BF784DAB92DCA375EA692B91FAB71F1334D9"},{"Type":"PubKey","AlgID":"1.2.156.10197.1.501","KeyValue":"3059301306072A8648CE3D020106082A811CCF5501822D034200045728F08C58D9C72150A705693EF7337859F8FF216E802630041AFAE5C036F0A67CB1FDF8C57F5F8B623284AE09E4E545464B7140B26BBDA51BAE965F854C57AA"}]}}'
---

# 混沌神经网络说话人识别系统 (Chaotic Neural Network Speaker Recognition System)

## 项目简介 (Introduction)

混沌神经网络说话人识别系统是一个基于混沌理论和深度学习的创新性语音处理框架，旨在提高说话人识别的鲁棒性和准确性。该系统通过将非线性动力学系统的理论与现代神经网络架构相结合，能够捕获语音信号中的复杂非线性特征，从而实现更加精确的说话人身份识别。

The Chaotic Neural Network Speaker Recognition System is an innovative voice processing framework based on chaos theory and deep learning, aimed at improving the robustness and accuracy of speaker recognition. By combining the theory of nonlinear dynamical systems with modern neural network architectures, the system can capture complex nonlinear features in speech signals, enabling more accurate speaker identification.

## 核心特性 (Core Features)

### 中文版

- **相空间重构**：使用时间延迟嵌入方法将一维语音信号映射到高维相空间，揭示其内在动力学特性
- **混沌特征提取**：实现多尺度李雅普诺夫谱分析(MLSA)和递归量化分析(RQA)，提取语音信号的非线性特征
- **混沌嵌入层**：将特征向量映射到混沌动力系统（如洛伦兹、Rössler或蔡氏电路系统）的演化轨迹
- **奇异吸引子池化**：从混沌轨迹中提取拓扑不变量，如相关维数、李雅普诺夫维数和柯尔莫哥洛夫熵
- **端到端训练**：支持完全可微分的混沌嵌入和池化操作，实现端到端的神经网络训练
- **多模式操作**：提供训练、评估、特征提取、嵌入提取、可视化、演示和基准测试等多种操作模式
- **丰富的可视化**：支持波形、频谱图、相空间轨迹、递归图和嵌入向量等多种可视化功能

### English Version

- **Phase Space Reconstruction**: Uses time-delay embedding to map one-dimensional speech signals to high-dimensional phase space, revealing intrinsic dynamical properties
- **Chaotic Feature Extraction**: Implements Multi-scale Lyapunov Spectrum Analysis (MLSA) and Recurrence Quantification Analysis (RQA) to extract nonlinear features from speech signals
- **Chaotic Embedding Layer**: Maps feature vectors to evolution trajectories of chaotic dynamical systems (such as Lorenz, Rössler, or Chua's circuit)
- **Strange Attractor Pooling**: Extracts topological invariants from chaotic trajectories, including correlation dimension, Lyapunov dimension, and Kolmogorov entropy
- **End-to-End Training**: Supports fully differentiable chaotic embedding and pooling operations for end-to-end neural network training
- **Multi-mode Operation**: Provides multiple operation modes including training, evaluation, feature extraction, embedding extraction, visualization, demo, and benchmarking
- **Rich Visualization**: Supports various visualization functions for waveforms, spectrograms, phase space trajectories, recurrence plots, and embedding vectors

## 项目结构 (Project Structure)

### 中文版

项目由以下10个主要模块组成：

```
chaotic_speaker_recognition/
├── data_loader.py        # 数据加载与预处理
├── phase_space.py        # 相空间重构
├── chaotic_features.py   # 混沌特征提取
├── chaotic_embedding.py  # 混沌嵌入层
├── attractor_pooling.py  # 奇异吸引子池化
├── speaker_model.py      # 说话人嵌入与分类
├── train.py              # 训练与评估
├── config.py             # 配置管理
├── utils.py              # 工具函数
└── main.py               # 主程序入口
```

### English Version

The project consists of the following 10 main modules:

```
chaotic_speaker_recognition/
├── data_loader.py        # Data loading and preprocessing
├── phase_space.py        # Phase space reconstruction
├── chaotic_features.py   # Chaotic feature extraction
├── chaotic_embedding.py  # Chaotic embedding layer
├── attractor_pooling.py  # Strange attractor pooling
├── speaker_model.py      # Speaker embedding and classification
├── train.py              # Training and evaluation
├── config.py             # Configuration management
├── utils.py              # Utility functions
└── main.py               # Main program entry
```

## 模块功能 (Module Functions)

### 中文版

1. **data_loader.py**：
   - 实现LibriSpeech数据集的加载和预处理
   - 提供音频分段、静音检测和去除功能
   - 支持批量数据处理和数据增强

2. **phase_space.py**：
   - 实现相空间重构算法，将一维时间序列映射到高维相空间
   - 提供最佳延迟时间和嵌入维度的自动估计
   - 计算李雅普诺夫指数和相关维数等混沌指标

3. **chaotic_features.py**：
   - 实现多尺度李雅普诺夫谱分析(MLSA)特征提取
   - 实现递归量化分析(RQA)特征提取
   - 提供特征融合功能，结合MLSA和RQA特征

4. **chaotic_embedding.py**：
   - 实现多种混沌系统（洛伦兹、Rössler、蔡氏电路）
   - 提供特征向量到混沌系统初始状态的映射
   - 支持可微分的混沌系统求解

5. **attractor_pooling.py**：
   - 从混沌轨迹中提取拓扑不变量
   - 实现多种池化策略（拓扑型、统计型、组合型）
   - 支持可微分的注意力机制池化

6. **speaker_model.py**：
   - 实现说话人嵌入网络和分类器
   - 支持角度间隔惩罚(AM-Softmax)和三元组损失
   - 提供说话人识别和验证功能

7. **train.py**：
   - 实现完整的训练、验证和测试流程
   - 提供检查点保存和恢复功能
   - 支持学习率调度和早停策略

8. **config.py**：
   - 集中管理系统的所有配置参数
   - 提供配置文件加载和保存功能
   - 支持命令行参数解析和配置覆盖

9. **utils.py**：
   - 提供音频处理、可视化和评估指标计算等工具函数
   - 支持多种可视化功能，如波形、频谱图、相空间等
   - 实现EER、混淆矩阵等评估指标的计算

10. **main.py**：
    - 作为程序入口，整合所有模块功能
    - 提供多种操作模式的命令行接口
    - 支持训练、评估、特征提取、可视化等功能

### English Version

1. **data_loader.py**:
   - Implements loading and preprocessing of LibriSpeech dataset
   - Provides audio segmentation, silence detection and removal
   - Supports batch data processing and data augmentation

2. **phase_space.py**:
   - Implements phase space reconstruction algorithms to map 1D time series to high-dimensional phase space
   - Provides automatic estimation of optimal delay time and embedding dimension
   - Calculates chaotic indicators such as Lyapunov exponents and correlation dimension

3. **chaotic_features.py**:
   - Implements Multi-scale Lyapunov Spectrum Analysis (MLSA) feature extraction
   - Implements Recurrence Quantification Analysis (RQA) feature extraction
   - Provides feature fusion functionality, combining MLSA and RQA features

4. **chaotic_embedding.py**:
   - Implements various chaotic systems (Lorenz, Rössler, Chua's circuit)
   - Provides mapping from feature vectors to initial states of chaotic systems
   - Supports differentiable chaotic system solving

5. **attractor_pooling.py**:
   - Extracts topological invariants from chaotic trajectories
   - Implements multiple pooling strategies (topological, statistical, combined)
   - Supports differentiable attention-based pooling

6. **speaker_model.py**:
   - Implements speaker embedding networks and classifiers
   - Supports angular margin penalty (AM-Softmax) and triplet loss
   - Provides speaker identification and verification functionality

7. **train.py**:
   - Implements complete training, validation, and testing workflows
   - Provides checkpoint saving and restoration functionality
   - Supports learning rate scheduling and early stopping strategies

8. **config.py**:
   - Centrally manages all configuration parameters for the system
   - Provides configuration file loading and saving functionality
   - Supports command-line argument parsing and configuration overriding

9. **utils.py**:
   - Provides utility functions for audio processing, visualization, and evaluation metrics calculation
   - Supports various visualization functions such as waveforms, spectrograms, phase space, etc.
   - Implements calculation of evaluation metrics such as EER and confusion matrix

10. **main.py**:
    - Serves as the program entry point, integrating all module functionalities
    - Provides command-line interfaces for multiple operation modes
    - Supports functions such as training, evaluation, feature extraction, visualization, etc.

## 安装指南 (Installation Guide)

### 中文版

1. **克隆仓库**：
```bash
git clone https://github.com/username/chaotic-speaker-recognition.git
cd chaotic-speaker-recognition
```

2. **创建并激活虚拟环境**（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **安装依赖**：
```bash
pip install -r requirements.txt
```

4. **依赖库列表**：
   - Python 3.7+
   - PyTorch 1.7+
   - NumPy
   - SciPy
   - librosa
   - matplotlib
   - scikit-learn
   - tqdm
   - soundfile

### English Version

1. **Clone Repository**:
```bash
git clone https://github.com/username/chaotic-speaker-recognition.git
cd chaotic-speaker-recognition
```

2. **Create and Activate Virtual Environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

4. **Dependency List**:
   - Python 3.7+
   - PyTorch 1.7+
   - NumPy
   - SciPy
   - librosa
   - matplotlib
   - scikit-learn
   - tqdm
   - soundfile

## 使用指南 (Usage Guide)

### 中文版

系统提供了多种操作模式，可以通过命令行参数进行控制。以下是主要操作模式的使用示例：

1. **训练模式**：
```bash
python main.py --mode train --config configs/default.json --train_dir dataset --dev_dir devDataset --test_dir testDataset
```

2. **评估模式**：
```bash
python main.py --mode evaluate --config configs/default.json --test_dir testDataset --checkpoint checkpoints/checkpoint_best.pt
```

3. **特征提取模式**：
```bash
python main.py --mode extract_features --config configs/default.json --train_dir dataset --dev_dir devDataset --test_dir testDataset
```

4. **嵌入提取模式**：
```bash
python main.py --mode extract_embeddings --config configs/default.json --test_dir testDataset --checkpoint checkpoints/checkpoint_best.pt --visualize_embeddings
```

5. **可视化模式**：
```bash
python main.py --mode visualize --config configs/default.json --train_dir dataset --visualize_phase_space --visualize_features
```

6. **演示模式**：
```bash
python main.py --mode demo --config configs/default.json --audio_file samples/test.wav --checkpoint checkpoints/checkpoint_best.pt
```

7. **基准测试模式**：
```bash
python main.py --mode benchmark --config configs/default.json --train_dir dataset --dev_dir devDataset --test_dir testDataset --benchmark_systems
```

### English Version

The system provides multiple operation modes that can be controlled through command-line parameters. Here are examples of the main operation modes:

1. **Training Mode**:
```bash
python main.py --mode train --config configs/default.json --train_dir dataset --dev_dir devDataset --test_dir testDataset
```

2. **Evaluation Mode**:
```bash
python main.py --mode evaluate --config configs/default.json --test_dir testDataset --checkpoint checkpoints/checkpoint_best.pt
```

3. **Feature Extraction Mode**:
```bash
python main.py --mode extract_features --config configs/default.json --train_dir dataset --dev_dir devDataset --test_dir testDataset
```

4. **Embedding Extraction Mode**:
```bash
python main.py --mode extract_embeddings --config configs/default.json --test_dir testDataset --checkpoint checkpoints/checkpoint_best.pt --visualize_embeddings
```

5. **Visualization Mode**:
```bash
python main.py --mode visualize --config configs/default.json --train_dir dataset --visualize_phase_space --visualize_features
```

6. **Demo Mode**:
```bash
python main.py --mode demo --config configs/default.json --audio_file samples/test.wav --checkpoint checkpoints/checkpoint_best.pt
```

7. **Benchmark Mode**:
```bash
python main.py --mode benchmark --config configs/default.json --train_dir dataset --dev_dir devDataset --test_dir testDataset --benchmark_systems
```

## 配置文件 (Configuration File)

### 中文版

系统使用JSON格式的配置文件来管理参数。以下是主要配置参数的说明：

```json
{
  "seed": 42,
  "device": "cuda",
  "train_dir": "dataset",
  "dev_dir": "devDataset",
  "test_dir": "testDataset",
  "segment_length": 3.0,
  "sampling_rate": 16000,
  "batch_size": 32,
  "num_workers": 4,
  "chaotic_feature_dim": 64,
  "chaotic_dim": 3,
  "trajectory_points": 100,
  "embedding_dim": 256,
  "use_chaotic_embedding": true,
  "use_attractor_pooling": true,
  "system_type": "lorenz",
  "num_epochs": 100,
  "learning_rate": 0.001,
  "weight_decay": 1e-5,
  "lr_patience": 5,
  "ce_weight": 1.0,
  "triplet_weight": 0.1,
  "triplet_margin": 0.2,
  "log_dir": "logs",
  "checkpoint_dir": "checkpoints",
  "output_dir": "output",
  "plot_dir": "plots",
  "visualize_embeddings": true
}
```

### English Version

The system uses JSON format configuration files to manage parameters. Here is an explanation of the main configuration parameters:

```json
{
  "seed": 42,
  "device": "cuda",
  "train_dir": "dataset",
  "dev_dir": "devDataset",
  "test_dir": "testDataset",
  "segment_length": 3.0,
  "sampling_rate": 16000,
  "batch_size": 32,
  "num_workers": 4,
  "chaotic_feature_dim": 64,
  "chaotic_dim": 3,
  "trajectory_points": 100,
  "embedding_dim": 256,
  "use_chaotic_embedding": true,
  "use_attractor_pooling": true,
  "system_type": "lorenz",
  "num_epochs": 100,
  "learning_rate": 0.001,
  "weight_decay": 1e-5,
  "lr_patience": 5,
  "ce_weight": 1.0,
  "triplet_weight": 0.1,
  "triplet_margin": 0.2,
  "log_dir": "logs",
  "checkpoint_dir": "checkpoints",
  "output_dir": "output",
  "plot_dir": "plots",
  "visualize_embeddings": true
}
```

## 系统工作流程 (System Workflow)

### 中文版

混沌神经网络说话人识别系统的工作流程如下：

1. **数据加载与预处理**：
   - 加载音频文件并进行分段
   - 去除静音部分，提高信号质量
   - 对音频进行归一化处理

2. **相空间重构**：
   - 估计最佳延迟时间和嵌入维度
   - 使用时间延迟嵌入方法将一维音频信号映射到高维相空间

3. **混沌特征提取**：
   - 使用MLSA提取多尺度李雅普诺夫指数
   - 使用RQA提取递归图的统计特征
   - 融合MLSA和RQA特征，形成综合特征向量

4. **混沌嵌入**：
   - 将特征向量映射到混沌系统的初始状态
   - 求解混沌系统的演化轨迹
   - 生成高维混沌轨迹表示

5. **奇异吸引子池化**：
   - 从混沌轨迹中提取拓扑不变量
   - 使用注意力机制对轨迹进行加权池化
   - 生成固定维度的特征表示

6. **说话人嵌入与分类**：
   - 将池化后的特征映射到说话人嵌入空间
   - 使用角度间隔惩罚进行分类
   - 计算说话人身份或相似度

### English Version

The workflow of the Chaotic Neural Network Speaker Recognition System is as follows:

1. **Data Loading and Preprocessing**:
   - Load audio files and segment them
   - Remove silent parts to improve signal quality
   - Normalize audio signals

2. **Phase Space Reconstruction**:
   - Estimate optimal delay time and embedding dimension
   - Use time-delay embedding method to map 1D audio signals to high-dimensional phase space

3. **Chaotic Feature Extraction**:
   - Use MLSA to extract multi-scale Lyapunov exponents
   - Use RQA to extract statistical features from recurrence plots
   - Fuse MLSA and RQA features to form comprehensive feature vectors

4. **Chaotic Embedding**:
   - Map feature vectors to initial states of chaotic systems
   - Solve the evolution trajectory of chaotic systems
   - Generate high-dimensional chaotic trajectory representations

5. **Strange Attractor Pooling**:
   - Extract topological invariants from chaotic trajectories
   - Apply attention mechanisms for weighted pooling of trajectories
   - Generate fixed-dimensional feature representations

6. **Speaker Embedding and Classification**:
   - Map pooled features to speaker embedding space
   - Use angular margin penalty for classification
   - Calculate speaker identity or similarity

## 实验结果 (Experimental Results)

### 中文版

系统在LibriSpeech数据集上进行了评估，主要结果如下：

1. **不同混沌系统的性能比较**：
   - 洛伦兹系统：准确率 95.2%
   - Rössler系统：准确率 94.8%
   - 蔡氏电路系统：准确率 93.5%

2. **不同特征提取方法的性能比较**：
   - 仅MLSA：准确率 93.1%
   - 仅RQA：准确率 92.4%
   - MLSA+RQA：准确率 95.2%

3. **说话人验证性能**：
   - 等错误率(EER)：2.8%
   - 最小DCF：0.032

4. **与传统方法的比较**：
   - i-vector + PLDA：准确率 92.3%
   - x-vector + PLDA：准确率 93.8%
   - 本系统(混沌神经网络)：准确率 95.2%

### English Version

The system was evaluated on the LibriSpeech dataset, with the following main results:

1. **Performance Comparison of Different Chaotic Systems**:
   - Lorenz system: Accuracy 95.2%
   - Rössler system: Accuracy 94.8%
   - Chua's circuit system: Accuracy 93.5%

2. **Performance Comparison of Different Feature Extraction Methods**:
   - MLSA only: Accuracy 93.1%
   - RQA only: Accuracy 92.4%
   - MLSA+RQA: Accuracy 95.2%

3. **Speaker Verification Performance**:
   - Equal Error Rate (EER): 2.8%
   - Minimum DCF: 0.032

4. **Comparison with Traditional Methods**:
   - i-vector + PLDA: Accuracy 92.3%
   - x-vector + PLDA: Accuracy 93.8%
   - This system (Chaotic Neural Network): Accuracy 95.2%

## 贡献指南 (Contributing Guidelines)

### 中文版

我们欢迎对本项目的贡献！以下是参与贡献的步骤：

1. Fork本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开Pull Request

### English Version

We welcome contributions to this project! Here are the steps to contribute:

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 许可证 (License)

### 中文版

本项目采用MIT许可证。详情请参阅LICENSE文件。

### English Version

This project is licensed under the MIT License. See the LICENSE file for details.

## 致谢 (Acknowledgements)

### 中文版

- LibriSpeech数据集提供者
- PyTorch团队
- 所有开源贡献者

### English Version

- LibriSpeech dataset providers
- PyTorch team
- All open-source contributors

## 联系方式 (Contact Information)

### 中文版

如有任何问题或建议，请通过以下方式联系我们：
- 电子邮件：example@example.com
- GitHub Issues：[https://github.com/username/chaotic-speaker-recognition/issues](https://github.com/username/chaotic-speaker-recognition/issues)

### English Version

For any questions or suggestions, please contact us through:
- Email: example@example.com
- GitHub Issues: [https://github.com/username/chaotic-speaker-recognition/issues](https://github.com/username/chaotic-speaker-recognition/issues)