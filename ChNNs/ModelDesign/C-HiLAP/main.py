#!/usr/bin/env python3
"""
C-HiLAP 说话人识别模型主运行脚本
解决了内存溢出和其他问题的版本
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import gc
import warnings

warnings.filterwarnings("ignore")

# 设置环境变量以优化内存使用
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def check_environment():
    """检查运行环境"""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)

    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    总内存: {props.total_memory / 1024 ** 3:.1f} GB")
            print(f"    多处理器数量: {props.multi_processor_count}")

    # 检查可用内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_free = torch.cuda.get_device_properties(0).total_memory
        print(f"GPU 可用内存: {memory_free / 1024 ** 3:.1f} GB")

    print()


def setup_device_and_memory():
    """设置设备和内存优化"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if torch.cuda.is_available():
        # 清空CUDA缓存
        torch.cuda.empty_cache()

        # 设置内存分配策略
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        print("CUDA 内存优化设置完成")

    return device


class OptimizedConfig:
    """优化后的配置参数"""
    # 模型参数
    INPUT_DIM = 1
    HIDDEN_DIM = 256  # 减少隐藏层维度
    EMBEDDING_DIM = 128  # 减少嵌入维度
    NUM_CLASSES = 50  # 减少类别数量以适应演示

    # 训练参数
    EPOCHS = 20  # 减少训练轮数
    BATCH_SIZE = 2  # 进一步减少批次大小
    LR = 0.001
    WEIGHT_DECAY = 1e-5

    # 数据参数
    MAX_SAMPLES = 200  # 限制总样本数量
    DURATION = 1.5  # 减少音频长度
    SAMPLE_RATE = 16000

    # 内存优化参数
    MAX_SEQ_LEN = 500  # 最大序列长度
    GRADIENT_ACCUMULATION_STEPS = 8  # 梯度累积
    ENABLE_MIXED_PRECISION = True

    # 混沌参数
    CHAOS_DIM = 3
    ATTENTION_HEADS = 4  # 减少注意力头数
    CHAOS_FEATURE_TYPE = 'mle'


def test_model_creation():
    """测试模型创建"""
    print("=" * 60)
    print("模型创建测试")
    print("=" * 60)

    try:
        # 动态导入修复后的模型
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

        # 更新配置
        from c_hilap_model import Config as ModelConfig
        ModelConfig.HIDDEN_DIM = OptimizedConfig.HIDDEN_DIM
        ModelConfig.EMBEDDING_DIM = OptimizedConfig.EMBEDDING_DIM
        ModelConfig.NUM_CLASSES = OptimizedConfig.NUM_CLASSES
        ModelConfig.MAX_SEQ_LEN = OptimizedConfig.MAX_SEQ_LEN
        ModelConfig.ATTENTION_HEADS = OptimizedConfig.ATTENTION_HEADS

        from c_hilap_model import CHiLAPModel, PhaseSynchronizationLoss

        # 创建模型
        device = setup_device_and_memory()
        model = CHiLAPModel(
            input_dim=OptimizedConfig.INPUT_DIM,
            hidden_dim=OptimizedConfig.HIDDEN_DIM,
            embedding_dim=OptimizedConfig.EMBEDDING_DIM,
            num_classes=OptimizedConfig.NUM_CLASSES
        ).to(device)

        print("模型创建成功!")

        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")

        # 测试前向传播
        print("\n测试前向传播...")
        batch_size = 2
        seq_len = 100

        with torch.no_grad():
            test_input = torch.randn(batch_size, seq_len, OptimizedConfig.INPUT_DIM).to(device)
            print(f"输入形状: {test_input.shape}")

            embeddings, logits = model(test_input)
            print(f"嵌入形状: {embeddings.shape}")
            print(f"输出形状: {logits.shape}")

            print("前向传播测试成功!")

        # 清理内存
        del model, test_input, embeddings, logits
        torch.cuda.empty_cache()
        gc.collect()

        return True

    except Exception as e:
        print(f"模型创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()


def test_data_loading():
    """测试数据加载"""
    print("=" * 60)
    print("数据加载测试")
    print("=" * 60)

    try:
        # 导入数据加载器
        from memory_efficient_data_loader import get_memory_efficient_dataloaders, Config as DataConfig

        # 更新数据配置
        DataConfig.DURATION = OptimizedConfig.DURATION
        DataConfig.BATCH_SIZE = OptimizedConfig.BATCH_SIZE
        DataConfig.MAX_SAMPLES = OptimizedConfig.MAX_SAMPLES

        # 创建数据加载器
        dataloaders = get_memory_efficient_dataloaders(
            dataset_name="demo",
            batch_size=OptimizedConfig.BATCH_SIZE,
            max_samples=OptimizedConfig.MAX_SAMPLES
        )

        if dataloaders is None:
            print("数据加载器创建失败")
            return False, None

        print("数据加载器创建成功!")

        # 测试每个数据分割
        for split_name, loader in dataloaders.items():
            print(f"{split_name}: {len(loader)} 批次")

            # 测试获取一个批次
            try:
                x, y = next(iter(loader))
                print(f"  批次形状: 音频 {x.shape}, 标签 {y.shape}")
                print(f"  数据范围: 音频 [{x.min():.3f}, {x.max():.3f}], 标签 [{y.min()}, {y.max()}]")
            except Exception as e:
                print(f"  获取 {split_name} 批次失败: {e}")
                return False, None

        print("数据加载测试成功!")
        return True, dataloaders

    except Exception as e:
        print(f"数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

    print()


def run_training_test(dataloaders):
    """运行训练测试"""
    print("=" * 60)
    print("训练测试")
    print("=" * 60)

    try:
        # 导入训练器
        from trainer import Trainer, Config as TrainerConfig

        # 更新训练器配置
        TrainerConfig.EPOCHS = 3  # 只训练3轮进行测试
        TrainerConfig.BATCH_SIZE = OptimizedConfig.BATCH_SIZE
        TrainerConfig.MAX_BATCH_SIZE = OptimizedConfig.BATCH_SIZE
        TrainerConfig.GRADIENT_ACCUMULATION_STEPS = OptimizedConfig.GRADIENT_ACCUMULATION_STEPS
        TrainerConfig.ENABLE_MIXED_PRECISION = OptimizedConfig.ENABLE_MIXED_PRECISION
        TrainerConfig.CHAOS_FEATURE_TYPE = OptimizedConfig.CHAOS_FEATURE_TYPE

        # 创建训练器
        trainer = Trainer(TrainerConfig)

        print("训练器创建成功!")
        print(f"使用设备: {trainer.device}")

        # 运行几个训练步骤
        print("\n开始训练测试...")

        # 获取训练和验证数据加载器
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']

        # 训练一个epoch
        try:
            train_loss = trainer.train_one_epoch(train_loader, epoch=1)
            print(f"训练损失: {train_loss:.4f}")

            # 验证
            val_loss, accuracy = trainer.validate(val_loader)
            print(f"验证损失: {val_loss:.4f}, 准确率: {accuracy:.2f}%")

            print("训练测试成功!")
            return True

        except Exception as e:
            print(f"训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()

    print()


def run_full_pipeline():
    """运行完整的训练流程"""
    print("=" * 60)
    print("完整训练流程")
    print("=" * 60)

    try:
        # 创建数据加载器
        from memory_efficient_data_loader import get_memory_efficient_dataloaders

        dataloaders = get_memory_efficient_dataloaders(
            dataset_name="demo",
            batch_size=OptimizedConfig.BATCH_SIZE,
            max_samples=OptimizedConfig.MAX_SAMPLES
        )

        if dataloaders is None:
            print("数据加载器创建失败")
            return False

        # 创建训练器
        from trainer import Trainer, Config as TrainerConfig

        # 配置训练参数
        TrainerConfig.EPOCHS = OptimizedConfig.EPOCHS
        TrainerConfig.BATCH_SIZE = OptimizedConfig.BATCH_SIZE
        TrainerConfig.LR = OptimizedConfig.LR
        TrainerConfig.GRADIENT_ACCUMULATION_STEPS = OptimizedConfig.GRADIENT_ACCUMULATION_STEPS
        TrainerConfig.ENABLE_MIXED_PRECISION = OptimizedConfig.ENABLE_MIXED_PRECISION

        trainer = Trainer(TrainerConfig)

        print("开始完整训练...")

        # 训练模型
        trainer.train(dataloaders['train'], dataloaders['val'])

        # 评估模型
        from trainer import SimpleEvaluator
        evaluator = SimpleEvaluator(trainer.model, trainer.chaos_feature_extractor)

        test_accuracy = evaluator.evaluate_accuracy(dataloaders['test'])
        print(f"最终测试准确率: {test_accuracy:.2f}%")

        return True

    except Exception as e:
        print(f"完整训练流程失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()


def main():
    """主函数"""
    print("C-HiLAP 说话人识别模型 - 优化版本")
    print("解决内存溢出和其他问题")

    # 检查环境
    check_environment()

    # 测试步骤
    tests = [
        ("模型创建", test_model_creation),
        ("数据加载", test_data_loading),
    ]

    dataloaders = None

    for test_name, test_func in tests:
        print(f"运行 {test_name} 测试...")

        if test_name == "数据加载":
            success, dataloaders = test_func()
        else:
            success = test_func()

        if not success:
            print(f"{test_name} 测试失败，停止执行")
            return
        else:
            print(f"{test_name} 测试通过！\n")

    # 运行训练测试
    if dataloaders is not None:
        print("运行训练测试...")
        if run_training_test(dataloaders):
            print("训练测试通过！\n")

            # 询问是否运行完整训练
            try:
                response = input("是否运行完整训练流程？(y/n): ").lower().strip()
                if response in ['y', 'yes', '是']:
                    run_full_pipeline()
                else:
                    print("跳过完整训练流程")
            except KeyboardInterrupt:
                print("\n用户中断")
        else:
            print("训练测试失败")

    print("程序执行完成")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 最终清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("内存清理完成")