import logging
import os
from datetime import datetime


def get_logger(log_dir, name="train_log"):
    """
    Args:
        log_dir (str/Path): 日志存放的目录 (如 output/All_crop_train)
        name (str): Logger 的名称
    """
    # 1. 确保目标文件夹存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 防止在多次调用 get_logger 时重复添加 Handler
    if not logger.handlers:
        # 2. 定义固定的日志文件名，方便“追加写入”
        # 如果你希望每次运行完全独立，可以保留时间戳；
        # 但为了方便“续写”，我们建议使用 main_train.log
        log_path = os.path.join(log_dir, "main_train.log")

        # 3. 创建文件处理器 (FileHandler)
        # mode='a' 表示追加模式 (Append)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")

        # 4. 创建控制台处理器 (StreamHandler)
        ch = logging.StreamHandler()

        # 5. 设置格式
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 6. 添加到 logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        # 防止日志信息传递到 root logger 导致重复打印
        logger.propagate = False

    return logger
