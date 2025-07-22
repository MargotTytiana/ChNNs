import torch
import tensorflow as tf
import platform


def check_pytorch():
    print("Checking PyTorch...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Is CUDA available (GPU): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"Is CPU available: {'cpu' in torch.device('cpu').type}\n")


def check_tensorflow():
    print("Checking TensorFlow...")
    print(f"TensorFlow version: {tf.__version__}")
    print("GPU devices:", tf.config.list_physical_devices('GPU'))
    print("CPU devices:", tf.config.list_physical_devices('CPU'))
    print()


def system_info():
    print("System Information:")
    print(f"Platform: {platform.system()}")
    print(f"Platform Version: {platform.version()}")
    print(f"Architecture: {platform.machine()}")
    print()


def main():
    system_info()
    check_pytorch()
    check_tensorflow()


if __name__ == "__main__":
    main()
