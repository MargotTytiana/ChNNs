import os
class Config:
    # 数据加载器配置
    DEBUG_MODE = True
    DEBUG_SAMPLE_SIZE = 5000
    BASE_DIR = os.getcwd()
    LIBRISPEECH_PATH = os.path.join(BASE_DIR, "devDataset", "LibriSpeech")
    SAMPLE_RATE = 16000
    DURATION = 1.0
    MAX_SAMPLES = int(SAMPLE_RATE * DURATION)
    NOISE_TYPES = ["white", "babble"]
    SNR_LEVELS = [0, 5, 10]
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    VALID_RATIO = 0.1
    MAX_SEQ_LEN = 16000
    
    # 模型配置
    INPUT_DIM = 1
    HIDDEN_DIM = 256
    EMBEDDING_DIM = 128
    CHAOS_DIM = 64
    CHAOS_TIME_STEPS = 5
    ATTENTION_HEADS = 4
    
    # 训练配置
    EPOCHS = 500
    LR = 0.01
    LR_DECAY = 0.95
    WEIGHT_DECAY = 1e-5
    SAVE_INTERVAL = 10
    VAL_INTERVAL = 1
    CHECKPOINT_DIR = "../checkpoints"
    WARMUP_EPOCHS = 5
    GRAD_CLIP = 1.0
    CE_WEIGHT = 1.0
    PATIENCE = 10
    MIN_DELTA = 0.001
    GRADIENT_ACCUMULATION_STEPS = 4
    ENABLE_MIXED_PRECISION = False