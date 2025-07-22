from utils.colored_logger import setup_logging
import logging

# 初始化日志系统
setup_logging("my_log.log")

logging.info("这是绿色的 info 信息")
logging.warning("这是黄色的 warning 信息")
logging.error("这是红色的 error 信息")
