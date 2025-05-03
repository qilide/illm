import logging
from illm.model.common import LOG_FILE_PATH

Logger = logging.getLogger(__name__)


def initialize_logger(log_file_path: str = LOG_FILE_PATH):
    """初始化日志配置"""
    Logger.setLevel(logging.DEBUG)

    # 控制台日志处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 文件日志处理器
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # 日志格式
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(log_formatter)
    file_handler.setFormatter(log_formatter)

    # 添加处理器
    Logger.addHandler(console_handler)
    Logger.addHandler(file_handler)
