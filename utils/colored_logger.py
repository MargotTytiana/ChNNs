# colored_logger.py
import logging
from colorlog import ColoredFormatter


def setup_logging(log_file='processing.log', level=logging.INFO):
    """
    设置同时输出到控制台和文件的日志系统，控制台输出为彩色，日志文件为普通文本。

    Args:
        log_file (str): 日志文件路径
        level (int): 日志等级，如 logging.INFO、logging.DEBUG
    """
    # 日志格式（彩色控制台）
    color_formatter = ColoredFormatter(
        fmt="%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'blue',
            'CRITICAL': 'bold_red',
        }
    )

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(color_formatter)

    # 文件 handler（无颜色）
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 移除旧 handler，防止重复打印
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
