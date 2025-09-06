import soundfile as sf
import numpy as np
import os
import sys


def test_soundfile_library():
    """测试 soundfile 库的功能"""
    print("测试 soundfile 库...")

    # 1. 检查库版本
    print(f"soundfile 版本: {sf.__version__}")
    print(f"libsndfile 版本: {sf.__libsndfile_version__}")

    # 2. 创建一个测试音频文件
    test_file = "test_audio.wav"
    sample_rate = 16000
    duration = 1.0  # 1秒
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz 正弦波

    try:
        # 3. 写入测试文件
        sf.write(test_file, signal, sample_rate)
        print(f"成功创建测试音频文件: {test_file}")

        # 4. 读取测试文件
        data, sr = sf.read(test_file)
        print(f"成功读取音频文件: 长度={len(data)}, 采样率={sr}")

        # 5. 验证读取的数据
        if np.allclose(data, signal, atol=1e-4):
            print("数据验证成功: 写入和读取的数据匹配")
        else:
            print("警告: 写入和读取的数据不匹配")
            diff = np.abs(data - signal)
            print(f"最大差异: {np.max(diff):.6f}")

        # 6. 清理测试文件
        os.remove(test_file)
        print("清理测试文件")

        return True
    except Exception as e:
        print(f"soundfile 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_soundfile_library()

    # 检查依赖项
    print("\n检查依赖项:")
    try:
        import librosa

        print(f"librosa 版本: {librosa.__version__}")
    except ImportError:
        print("未安装 librosa")

    try:
        import numpy

        print(f"numpy 版本: {np.__version__}")
    except ImportError:
        print("未安装 numpy")

    print(f"\n测试结果: {'成功' if success else '失败'}")
    sys.exit(0 if success else 1)